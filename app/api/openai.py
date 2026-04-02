"""OpenAI-compatible API endpoints for OpenWebUI integration."""

import time
import uuid
from typing import List, Optional, AsyncGenerator

import structlog
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse

from app.agents.graph import create_agent_graph
from app.agents.state import create_initial_state, GraphState
from app.config import get_settings
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    ChatMessageResponse,
    Choice,
    ChoiceChunk,
    Delta,
    Usage,
    OpenAIError,
    OpenAIErrorDetail,
)

logger = structlog.get_logger()

router = APIRouter()
settings = get_settings()


async def verify_api_key(request: Request) -> bool:
    """Verify API key if configured.

    Args:
        request: FastAPI request object

    Returns:
        True if authentication passes or is not required

    Raises:
        HTTPException: If authentication fails
    """
    if not settings.api_key:
        return True  # No auth required

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=OpenAIError(
                error=OpenAIErrorDetail(
                    message="Missing or invalid Authorization header",
                    type="invalid_request_error",
                    code="unauthorized",
                )
            ).model_dump(),
        )

    token = auth_header[7:]  # Remove "Bearer " prefix
    if token != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail=OpenAIError(
                error=OpenAIErrorDetail(
                    message="Invalid API key",
                    type="invalid_request_error",
                    code="unauthorized",
                )
            ).model_dump(),
        )

    return True


@router.get("/models")
async def list_models():
    """List available models.

    Returns a list containing the single virtual model that represents
    the GRAG pipeline.

    Returns:
        Model list response with grag-pipeline-v1 model
    """
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "grag",
                "permission": [],
            }
        ],
    }


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _auth: bool = Depends(verify_api_key),
) -> ChatCompletionResponse | StreamingResponse:
    """Handle chat completion requests.

    Supports both streaming and non-streaming responses. Routes to the
    appropriate handler based on the stream parameter.

    Args:
        request: Chat completion request with messages and model
        _auth: Authentication dependency (validated via verify_api_key)

    Returns:
        Non-streaming: ChatCompletionResponse
        Streaming: StreamingResponse with SSE chunks
    """
    # Validate model
    if request.model != settings.model_name:
        raise HTTPException(
            status_code=400,
            detail=OpenAIError(
                error=OpenAIErrorDetail(
                    message=f"Unknown model: {request.model}",
                    type="invalid_request_error",
                    param="model",
                    code="model_not_found",
                )
            ).model_dump(),
        )

    if request.stream:
        return StreamingResponse(
            stream_chat_completions(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        return await non_streaming_completion(request)


async def non_streaming_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion via LangGraph.

    Args:
        request: Chat completion request

    Returns:
        ChatCompletionResponse with the model's response
    """
    # Extract messages - find the last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail=OpenAIError(
                error=OpenAIErrorDetail(
                    message="No user message found",
                    type="invalid_request_error",
                    param="messages",
                    code="invalid_message",
                )
            ).model_dump(),
        )

    # Create session ID and initial state
    session_id = str(uuid.uuid4())
    initial_state = create_initial_state(user_query=user_message, session_id=session_id)

    logger.info(
        "chat_completion.pipeline.start",
        model=request.model,
        session_id=session_id,
    )

    # Invoke LangGraph pipeline
    try:
        app = create_agent_graph()
        config = {"configurable": {"thread_id": session_id}}

        final_state: GraphState = await app.ainvoke(initial_state, config)

        # Extract response components
        answer = final_state.get("answer", "")
        reasoning_steps = final_state.get("reasoning_steps", [])
        mermaid_path = final_state.get("mermaid_path", "")

        # Format the response with xAI content
        response_content = answer

        if reasoning_steps:
            response_content += "\n\n## Reasoning Steps\n"
            for i, step in enumerate(reasoning_steps, 1):
                response_content += f"\n{i}. {step}"

        if mermaid_path:
            response_content += (
                f"\n\n## Graph Reasoning Path\n```mermaid\n{mermaid_path}\n```"
            )

        logger.info(
            "chat_completion.pipeline.complete",
            session_id=session_id,
            answer_length=len(answer),
            reasoning_steps=len(reasoning_steps),
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessageResponse(
                        role="assistant", content=response_content
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=len(user_message.split()),
                completion_tokens=len(answer.split()),
                total_tokens=len(user_message.split()) + len(answer.split()),
            ),
        )

    except Exception as e:
        logger.error("pipeline.error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=OpenAIError(
                error=OpenAIErrorDetail(
                    message=f"Pipeline execution failed: {str(e)}",
                    type="internal_error",
                    code="pipeline_error",
                )
            ).model_dump(),
        )


async def stream_chat_completions(
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses with status updates + token chunks.

    Args:
        request: Chat completion request

    Yields:
        SSE-formatted data chunks with completion deltas
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Extract user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    # Create session ID and initial state
    session_id = str(uuid.uuid4())
    initial_state = create_initial_state(user_query=user_message, session_id=session_id)

    logger.info(
        "chat_completion.stream.start",
        model=request.model,
        session_id=session_id,
    )

    # Send initial status message
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[
            ChoiceChunk(
                index=0,
                delta=Delta(content="[GRAG is processing your request...]\n\n"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    try:
        # Invoke LangGraph pipeline
        app = create_agent_graph()
        config = {"configurable": {"thread_id": session_id}}

        final_state: GraphState = await app.ainvoke(initial_state, config)

        # Extract answer
        answer = final_state.get("answer", "")
        reasoning_steps = final_state.get("reasoning_steps", [])
        mermaid_path = final_state.get("mermaid_path", "")

        # Format full response
        full_response = answer

        if reasoning_steps:
            full_response += "\n\n## Reasoning Steps\n"
            for i, step in enumerate(reasoning_steps, 1):
                full_response += f"\n{i}. {step}"

        if mermaid_path:
            full_response += (
                f"\n\n## Graph Reasoning Path\n```mermaid\n{mermaid_path}\n```"
            )

        logger.info(
            "chat_completion.stream.pipeline_complete",
            session_id=session_id,
            answer_length=len(answer),
        )

        # Stream content token by token (word-by-word)
        words = full_response.split()
        for i, word in enumerate(words):
            chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=request.model,
                choices=[
                    ChoiceChunk(
                        index=0,
                        delta=Delta(content=word + (" " if i < len(words) - 1 else "")),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error("stream.pipeline.error", error=str(e))
        chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=Delta(content=f"\n[Error: {str(e)}]"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[ChoiceChunk(index=0, delta=Delta(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    logger.info("chat_completion.stream.complete", chunk_id=chunk_id)

    yield "data: [DONE]\n\n"


# Export router for inclusion in main.py
__all__ = ["router"]
