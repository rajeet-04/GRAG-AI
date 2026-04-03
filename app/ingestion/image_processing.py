"""Image preprocessing and OCR/caption helpers for ingestion."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from io import BytesIO
from typing import Any, Dict, Tuple

import httpx
import structlog

try:
    from PIL import Image, ImageOps, UnidentifiedImageError

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive for partial environments
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    UnidentifiedImageError = Exception  # type: ignore[assignment]
    PIL_AVAILABLE = False

from app.config import get_settings
from app.llm.ollama_client import OllamaClient


logger = structlog.get_logger()

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


def is_supported_image_filename(filename: str) -> bool:
    """Check whether a filename looks like a supported image."""
    if not filename:
        return False
    lower = filename.lower().strip()
    return any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS)


def preprocess_image_bytes(raw: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Normalize uploaded image for inference/storage metadata.

    Steps:
    1. Apply EXIF orientation correction
    2. Convert to RGB
    3. Resize longest side to configured max dimension
    4. Re-encode as optimized JPEG
    """
    settings = get_settings()

    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for image preprocessing")

    with Image.open(BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img)
        original_format = (img.format or "UNKNOWN").upper()
        original_mode = img.mode
        original_width, original_height = img.size

        if img.mode != "RGB":
            img = img.convert("RGB")

        max_dimension = max(int(settings.ingestion_image_max_dimension), 256)
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        processed_width, processed_height = img.size

        output = BytesIO()
        quality = max(40, min(int(settings.ingestion_image_jpeg_quality), 95))
        img.save(output, format="JPEG", optimize=True, quality=quality)
        processed = output.getvalue()

    metadata = {
        "sha256": hashlib.sha256(processed).hexdigest(),
        "original_format": original_format,
        "original_mode": original_mode,
        "original_width": original_width,
        "original_height": original_height,
        "processed_format": "JPEG",
        "processed_width": processed_width,
        "processed_height": processed_height,
        "processed_size_bytes": len(processed),
    }

    return processed, metadata


def build_image_fallback_text(filename: str, metadata: Dict[str, Any]) -> str:
    """Build structured fallback text when vision extraction is unavailable."""
    return (
        f"Image file {filename}. "
        f"Processed resolution: {metadata.get('processed_width')}x{metadata.get('processed_height')}. "
        f"Original format: {metadata.get('original_format')}. "
        "No OCR text could be extracted from this image."
    )


async def _extract_text_via_openai_compat(
    *,
    prompt: str,
    encoded_image: str,
) -> str:
    """Fallback OCR extraction via Ollama OpenAI-compatible endpoint."""
    settings = get_settings()
    base = str(settings.ollama_base_url).rstrip("/")
    endpoint = f"{base}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if getattr(settings, "ollama_api_key", None):
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"

    payload = {
        "model": settings.ollama_vision_model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=25.0, headers=headers) as client:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


async def extract_text_from_image(
    image_bytes: bytes,
    *,
    filename: str,
) -> str:
    """Extract OCR/caption text from image via Ollama vision-capable model."""
    settings = get_settings()
    client = OllamaClient(use_cloud=False)
    available_models = await client.list_models()
    model_names = {str(m.get("name", "")).strip() for m in available_models}
    if not any(
        name == settings.ollama_vision_model
        or name.startswith(f"{settings.ollama_vision_model}:")
        for name in model_names
    ):
        raise RuntimeError(
            f"Vision model '{settings.ollama_vision_model}' is not installed in local Ollama"
        )

    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Extract readable text from this image exactly when possible. "
        "Then provide a concise factual description of relevant entities, objects, "
        "organizations, dates, and relationships visible in the image. "
        "Return plain text only."
    )

    try:
        response = await asyncio.wait_for(
            client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [encoded],
                    }
                ],
                temperature=0.1,
                max_tokens=500,
                think=False,
                model=settings.ollama_vision_model,
            ),
            timeout=25.0,
        )
        content = (response.get("content") or "").strip()
        if content:
            return content
        raise RuntimeError("Vision model returned empty content")
    except Exception as e:
        logger.warning(
            "image_processing.vision_native_failed",
            filename=filename,
            model=settings.ollama_vision_model,
            error=str(e),
        )
        try:
            compat_content = await _extract_text_via_openai_compat(
                prompt=prompt,
                encoded_image=encoded,
            )
            if compat_content:
                return compat_content
            raise RuntimeError("OpenAI-compatible vision endpoint returned empty content")
        except Exception as compat_err:
            combined_error = f"native={e}; openai_compat={compat_err}"
            logger.warning(
                "image_processing.vision_failed",
                filename=filename,
                model=settings.ollama_vision_model,
                error=combined_error,
            )
            raise RuntimeError(combined_error) from compat_err


def decode_image(raw: bytes) -> Image.Image:
    """Decode image bytes to PIL image for unit tests and validation."""
    if not PIL_AVAILABLE:
        raise ValueError("Pillow is not available")
    try:
        return Image.open(BytesIO(raw))
    except UnidentifiedImageError as e:
        raise ValueError("Unsupported or invalid image data") from e
