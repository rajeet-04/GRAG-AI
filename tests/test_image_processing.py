"""Tests for image preprocessing and vision extraction helpers."""

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest

from app.ingestion.image_processing import (
    build_image_fallback_text,
    is_supported_image_filename,
    preprocess_image_bytes,
    extract_text_from_image,
)


pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _make_png_bytes(width: int = 32, height: int = 32) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_is_supported_image_filename():
    assert is_supported_image_filename("photo.png") is True
    assert is_supported_image_filename("scan.JPG") is True
    assert is_supported_image_filename("doc.pdf") is False


def test_preprocess_image_bytes_outputs_metadata():
    raw = _make_png_bytes(120, 80)
    processed, metadata = preprocess_image_bytes(raw)

    assert len(processed) > 0
    assert metadata["processed_format"] == "JPEG"
    assert metadata["processed_width"] > 0
    assert metadata["processed_height"] > 0
    assert metadata["sha256"]


def test_build_image_fallback_text():
    text = build_image_fallback_text(
        "image.png",
        {
            "processed_width": 100,
            "processed_height": 80,
            "original_format": "PNG",
        },
    )
    assert "image.png" in text
    assert "100x80" in text


@pytest.mark.asyncio
async def test_extract_text_from_image_uses_vision_model():
    raw = _make_png_bytes(64, 64)

    with patch("app.ingestion.image_processing.OllamaClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.list_models = AsyncMock(return_value=[{"name": "qwen3.5:4b"}])
        mock_client.chat = AsyncMock(return_value={"content": "Detected text"})

        content = await extract_text_from_image(raw, filename="test.png")

    assert "Detected text" in content


@pytest.mark.asyncio
async def test_extract_text_from_image_falls_back_to_openai_compat():
    raw = _make_png_bytes(64, 64)

    with patch("app.ingestion.image_processing.OllamaClient") as mock_client_cls, patch(
        "app.ingestion.image_processing._extract_text_via_openai_compat",
        new=AsyncMock(return_value="Fallback OCR text"),
    ):
        mock_client = mock_client_cls.return_value
        mock_client.list_models = AsyncMock(return_value=[{"name": "qwen3.5:4b"}])
        mock_client.chat = AsyncMock(side_effect=RuntimeError("native failed"))

        content = await extract_text_from_image(raw, filename="test.png")

    assert content == "Fallback OCR text"
