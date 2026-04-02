"""OpenAI-compatible API schemas for OpenWebUI integration.

Provides request/response models that match the OpenAI Chat API format
for compatibility with OpenWebUI and other OpenAI API clients.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    """Single message in a conversation."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Ignore extra parameters gracefully
    model_config = ConfigDict(extra="ignore")


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessageResponse(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """Individual choice in a non-streaming response."""

    index: int
    message: ChatMessageResponse
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """Full non-streaming chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage = Usage()


class Delta(BaseModel):
    """Delta content for streaming responses."""

    content: Optional[str] = None
    role: Optional[str] = None


class ChoiceChunk(BaseModel):
    """Individual choice in a streaming response."""

    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming response chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChoiceChunk]


class OpenAIErrorDetail(BaseModel):
    """Detail object within an OpenAI error response."""

    message: str
    type: str
    param: Optional[str] = None
    code: str


class OpenAIError(BaseModel):
    """OpenAI-compatible error response."""

    error: OpenAIErrorDetail


# Export all models for convenient importing
__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "Usage",
    "ChatMessageResponse",
    "Choice",
    "ChatCompletionResponse",
    "Delta",
    "ChoiceChunk",
    "ChatCompletionChunk",
    "OpenAIErrorDetail",
    "OpenAIError",
]
