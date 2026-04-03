"""OpenAI-compatible API endpoints for OpenWebUI integration."""

import base64
import binascii
import time
import uuid
from typing import Any, List, Optional, AsyncGenerator

import structlog
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse

from app.agents.graph import create_agent_graph
from app.agents.state import create_initial_state, GraphState
from app.config import get_settings
from app.ingestion.image_processing import (
    build_image_fallback_text,
    extract_text_from_image,
    preprocess_image_bytes,
)
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

IMAGE_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/gif": "gif",
    "image/tiff": "tiff",
}


def _first_non_empty_string(*values: Any) -> str:
    """Return the first non-empty string value from inputs."""
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def _extract_image_url_from_part(part: dict[str, Any]) -> str:
    """Extract image URL from OpenAI/OpenWebUI multimodal content part."""
    image_obj = part.get("image_url") or part.get("image")
    if isinstance(image_obj, dict):
        return _first_non_empty_string(image_obj.get("url"), part.get("url"))
    if isinstance(image_obj, str):
        return image_obj.strip()
    return _first_non_empty_string(part.get("url"))


def _parse_user_content(raw_content: Any) -> tuple[str, list[str]]:
    """Parse user message content into text and image URLs.

    Supports OpenAI-compatible multimodal payloads where content is
    either a plain string or an array of typed parts.
    """
    text_parts: list[str] = []
    image_urls: list[str] = []

    if isinstance(raw_content, str):
        text = raw_content.strip()
        if text:
            text_parts.append(text)
        return "\n".join(text_parts).strip(), image_urls

    if isinstance(raw_content, dict):
        raw_content = [raw_content]

    if not isinstance(raw_content, list):
        fallback = str(raw_content).strip()
        if fallback:
            text_parts.append(fallback)
        return "\n".join(text_parts).strip(), image_urls

    for part in raw_content:
        if isinstance(part, str):
            text = part.strip()
            if text:
                text_parts.append(text)
            continue

        if not isinstance(part, dict):
            continue

        part_type = str(part.get("type", "")).strip().lower()

        if part_type in {"text", "input_text"}:
            text = _first_non_empty_string(part.get("text"), part.get("content"))
            if text:
                text_parts.append(text)
            continue

        if part_type in {"image_url", "input_image", "image"}:
            image_url = _extract_image_url_from_part(part)
            if image_url:
                image_urls.append(image_url)
            continue

        # Unknown part type: salvage any embedded text.
        text = _first_non_empty_string(part.get("text"), part.get("content"))
        if text:
            text_parts.append(text)

    return "\n".join(text_parts).strip(), image_urls


def _decode_data_uri_image(url: str) -> tuple[bytes, str] | None:
    """Decode data:image/...;base64,... URLs to bytes and MIME type."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None

    try:
        header, payload = url.split(",", 1)
    except ValueError:
        return None

    if ";base64" not in header:
        return None

    mime = header[5:].split(";", 1)[0].strip().lower() or "application/octet-stream"

    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None

    return decoded, mime


def _extract_image_urls_from_files(files: list[dict[str, Any]] | None) -> list[str]:
    """Extract image URLs/data URIs from OpenWebUI `files` payload."""
    if not files:
        return []

    urls: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue

        image_obj = item.get("image_url") or item.get("image")
        image_url = ""
        if isinstance(image_obj, dict):
            image_url = _first_non_empty_string(image_obj.get("url"))
        elif isinstance(image_obj, str):
            image_url = image_obj.strip()

        if not image_url:
            image_url = _first_non_empty_string(
                item.get("url"),
                item.get("file_url"),
                item.get("download_url"),
                item.get("content"),
            )

        if image_url:
            urls.append(image_url)

    return urls


def _build_image_filename(index: int, mime: str) -> str:
    """Build deterministic filename for chat image parts."""
    ext = IMAGE_MIME_TO_EXT.get(mime.lower(), "bin")
    return f"openwebui_upload_{index}.{ext}"


async def _prepare_chat_input(request: ChatCompletionRequest) -> tuple[str, str]:
    """Extract text query and image-ingestion text from request messages."""
    user_text = ""
    image_urls: list[str] = []

    for msg in reversed(request.messages):
        if msg.role != "user":
            continue
        user_text, image_urls = _parse_user_content(msg.content)
        if user_text or image_urls:
            break

    for file_url in _extract_image_urls_from_files(request.files):
        if file_url not in image_urls:
            image_urls.append(file_url)

    if not user_text and not image_urls:
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

    extracted_blocks: list[str] = []

    for idx, image_url in enumerate(image_urls, start=1):
        decoded = _decode_data_uri_image(image_url)
        if not decoded:
            logger.warning(
                "openai.chat.image.unsupported_url",
                index=idx,
                preview=image_url[:80],
            )
            continue

        raw_image, mime = decoded
        filename = _build_image_filename(idx, mime)

        image_metadata: dict[str, Any] = {
            "original_format": mime,
            "processed_width": None,
            "processed_height": None,
        }

        try:
            processed_image, image_metadata = preprocess_image_bytes(raw_image)
            extracted = await extract_text_from_image(
                processed_image,
                filename=filename,
            )
            block_text = extracted.strip() or build_image_fallback_text(
                filename,
                image_metadata,
            )
        except Exception as e:
            logger.warning(
                "openai.chat.image.extraction_failed",
                filename=filename,
                error=str(e),
            )
            block_text = build_image_fallback_text(filename, image_metadata)

        extracted_blocks.append(f"[Image: {filename}]\n{block_text}")

    if extracted_blocks:
        image_context = "\n\n".join(extracted_blocks)
        if user_text:
            final_query = (
                f"{user_text}\n\n"
                "Attached image transcription:\n"
                f"{image_context}"
            )
        else:
            final_query = (
                "Analyze the attached image and answer using this transcription:\n"
                f"{image_context}"
            )
        return final_query, image_context

    return user_text, ""


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


@router.post("/chat/completions", response_model=None)
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
    user_message, ingestion_text = await _prepare_chat_input(request)

    # Create session ID and initial state
    session_id = str(uuid.uuid4())
    initial_state = create_initial_state(user_query=user_message, session_id=session_id)
    if ingestion_text:
        initial_state["ingestion_text"] = ingestion_text

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
    """True streaming: runs pipeline → context, then streams explanation from Ollama.

    Flow:
    1. Emit "[GRAG is processing...]" SSE chunk immediately  →  user sees activity
    2. Run create_pipeline_graph() (all agents EXCEPT explanation)  →  ~1-3s
    3. Pipe Ollama's streaming /api/chat response directly as SSE chunks  →  real-time tokens
    4. Emit [DONE]

    This eliminates the fake word-splitting that destroyed markdown formatting.
    Tokens flow: Ollama cloud → FastAPI SSE → Open WebUI render  in real time.
    """
    from app.agents.explanation_agent import stream_explanation
    from app.agents.graph import create_pipeline_graph

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    session_id = str(uuid.uuid4())
    user_message = ""
    ingestion_text = ""

    def _chunk(content: str, finish: str | None = None) -> str:
        """Format an SSE data chunk in OpenAI streaming format."""
        c = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[ChoiceChunk(index=0, delta=Delta(content=content), finish_reason=finish)],
        )
        return f"data: {c.model_dump_json()}\n\n"

    # ── Step 1: Processing indicator ──────────────────────────────────────────
    yield _chunk("[GRAG is processing your request...]\n\n")

    try:
        user_message, ingestion_text = await _prepare_chat_input(request)
        initial_state = create_initial_state(user_query=user_message, session_id=session_id)
        if ingestion_text:
            initial_state["ingestion_text"] = ingestion_text

        logger.info(
            "chat_completion.stream.start",
            model=request.model,
            session_id=session_id,
            query_preview=user_message[:100],
            image_ingestion=bool(ingestion_text),
        )

        # ── Step 2: Run pipeline stages (query → search → context) ───────────
        pipeline = create_pipeline_graph()
        config = {"configurable": {"thread_id": session_id}}
        pipeline_state = await pipeline.ainvoke(initial_state, config)

        logger.info(
            "chat_completion.stream.pipeline_ready",
            session_id=session_id,
            context_length=len(pipeline_state.get("merged_context", "")),
            kr_paths=len(pipeline_state.get("kr_paths", [])),
        )

        # ── Step 3: Stream explanation tokens directly from Ollama ────────────
        async for token in stream_explanation(pipeline_state):
            yield _chunk(token)

    except Exception as e:
        logger.error("stream.pipeline.error", error=str(e), session_id=session_id)
        yield _chunk(f"\n\n[Pipeline error: {str(e)[:200]}]")

    # ── Step 4: Final stop chunk + DONE ───────────────────────────────────────
    stop = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[ChoiceChunk(index=0, delta=Delta(), finish_reason="stop")],
    )
    yield f"data: {stop.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    logger.info("chat_completion.stream.complete", chunk_id=chunk_id, session_id=session_id)




# Export router for inclusion in main.py
__all__ = ["router"]
