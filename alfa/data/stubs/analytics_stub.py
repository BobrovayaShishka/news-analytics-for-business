"""Заглушка аналитики: карточки и чанки, набранные вручную из материалов банка."""

from __future__ import annotations

from pathlib import Path

from alfa.core.config import resolve_path
from alfa.core.schemas import AnalyticsCard, AnalyticsChunk, AnalyticsDocMeta
from alfa.data.stubs.news_stub import read_jsonl


class JsonlAnalyticsStore:
    def __init__(self, directory: str | Path):
        self.directory = resolve_path(directory)
        self._docs = [
            AnalyticsDocMeta.model_validate(row) for row in read_jsonl(self.directory / "docs.jsonl")
        ]
        self._chunks = [
            AnalyticsChunk.model_validate(row) for row in read_jsonl(self.directory / "chunks.jsonl")
        ]
        self._cards = [
            AnalyticsCard.model_validate(row) for row in read_jsonl(self.directory / "cards.jsonl")
        ]

    def list_docs(self) -> list[AnalyticsDocMeta]:
        return list(self._docs)

    def get_chunks(self, doc_id: str | None = None) -> list[AnalyticsChunk]:
        if doc_id is None:
            return list(self._chunks)
        return [item for item in self._chunks if item.doc_id == doc_id]

    def get_cards(self, doc_id: str | None = None) -> list[AnalyticsCard]:
        if doc_id is None:
            return list(self._cards)
        return [item for item in self._cards if item.doc_id == doc_id]
