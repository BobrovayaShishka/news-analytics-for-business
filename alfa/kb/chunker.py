"""Нарезка текста аналитики окном токенов с перекрытием.

Граница чанка по возможности совпадает с абзацем. Если абзац длиннее окна,
он режется по словам. Токены: tiktoken, если установлен, иначе число слов.
Метаданные документа наследуются каждым чанком.
"""

from __future__ import annotations

import re
from datetime import datetime

from alfa.core.config import load_runtime_config
from alfa.core.ids import make_chunk_id
from alfa.core.schemas import AnalyticsChunk, Section, Verified

_PARAGRAPH = re.compile(r"\n\s*\n")


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        words = text.split()
        return len(words) if words else 0


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if size <= 0:
        raise ValueError("size_tokens must be positive")
    step = max(1, size - max(0, overlap))
    pieces: list[str] = []
    start = 0
    while start < len(words):
        piece = words[start : start + size]
        if not piece:
            break
        pieces.append(" ".join(piece))
        if start + size >= len(words):
            break
        start += step
    return pieces


def chunk_text(
    text: str,
    *,
    doc_id: str,
    page: int,
    published_at: datetime,
    size_tokens: int | None = None,
    overlap_tokens: int | None = None,
    section: str = "other",
    bbox_rel: list[float] | None = None,
    verified: str = "manual",
    ocr_confidence: float | None = 1.0,
    start_index: int = 0,
) -> list[AnalyticsChunk]:
    search = load_runtime_config()["search"]
    if size_tokens is None:
        size_tokens = int(search["size_tokens"])
    if overlap_tokens is None:
        overlap_tokens = int(search["overlap_tokens"])
    paragraphs = [part.strip() for part in _PARAGRAPH.split(text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    windows: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        windows.append("\n\n".join(buffer))
        if overlap_tokens <= 0:
            buffer = []
            buffer_tokens = 0
            return
        kept: list[str] = []
        kept_tokens = 0
        for paragraph in reversed(buffer):
            tokens = count_tokens(paragraph)
            if kept and kept_tokens + tokens > overlap_tokens:
                break
            if not kept and tokens > overlap_tokens:
                break
            kept.append(paragraph)
            kept_tokens += tokens
        buffer = list(reversed(kept))
        buffer_tokens = kept_tokens

    for paragraph in paragraphs:
        tokens = count_tokens(paragraph)
        if tokens > size_tokens:
            flush()
            buffer = []
            buffer_tokens = 0
            windows.extend(_split_words(paragraph, size_tokens, overlap_tokens))
            continue
        if buffer and buffer_tokens + tokens > size_tokens:
            flush()
        buffer.append(paragraph)
        buffer_tokens += tokens
    if buffer:
        joined = "\n\n".join(buffer)
        if not windows or windows[-1] != joined:
            windows.append(joined)

    box = bbox_rel or [0.0, 0.0, 1.0, 1.0]
    chunks: list[AnalyticsChunk] = []
    for offset, piece in enumerate(windows):
        n = start_index + offset
        chunks.append(
            AnalyticsChunk(
                chunk_id=make_chunk_id(doc_id, page, n),
                doc_id=doc_id,
                page=page,
                bbox_rel=box,
                text=piece,
                section=Section(section),
                ocr_confidence=ocr_confidence,
                verified=Verified(verified),
                published_at=published_at,
            )
        )
    return chunks
