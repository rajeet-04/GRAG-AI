"""Text chunking utilities for ingestion-time extraction."""

from __future__ import annotations

import re


def normalize_text_for_extraction(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    if not text:
        return ""

    cleaned = text.replace("\x00", " ").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def split_text_into_chunks(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split text into overlap-aware chunks with boundary preference.

    Uses a character-window strategy and backtracks to sentence/paragraph
    boundaries to reduce semantic breakage.
    """
    normalized = normalize_text_for_extraction(text)
    if not normalized:
        return []

    if max_chars <= 0:
        return [normalized]

    if overlap_chars < 0:
        overlap_chars = 0

    chunks: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + max_chars, length)

        if end < length:
            # Prefer splitting at natural boundaries near the end of the window.
            for sep in ("\n\n", ". ", "\n", " "):
                split_at = normalized.rfind(sep, start, end)
                if split_at > start + max_chars // 2:
                    end = split_at + len(sep)
                    break

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(0, end - overlap_chars)

    return chunks
