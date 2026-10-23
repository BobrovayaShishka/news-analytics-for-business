"""Контракты источников. Реальный парсер подменяет заглушку через data.impl."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from alfa.core.schemas import AnalyticsCard, AnalyticsChunk, AnalyticsDocMeta, NewsDoc


class IngestError(RuntimeError):
    """Явная ошибка пользовательского забора. Пустой текст сюда не молчит."""


def eligible_for_extraction(doc: NewsDoc) -> bool:
    """Очередь извлечения: первичные документы вне дешёвого отсева и страниц без JS."""
    if not doc.is_primary:
        return False
    if doc.prefilter == "likely_irrelevant":
        return False
    if "needs_js" in doc.flags:
        return False
    if doc.extraction_status == "done":
        return False
    return True


@dataclass
class SourceHealth:
    name: str
    state: str
    last_success_at: str | None
    consecutive_errors: int
    empty_ratio: float
    new_docs: int
    poll_interval_s: int


@dataclass
class RunReport:
    run_id: str
    started_at: str
    finished_at: str
    found: int = 0
    downloaded: int = 0
    parsed: int = 0
    empty_or_short: int = 0
    flagged: int = 0
    clustered: int = 0
    new_docs: int = 0
    problems: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    duplicate_ratio: float = 0.0
    freshness_hours: float | None = None
    coverage: float = 0.0
    sample_clusters: list[dict[str, Any]] = field(default_factory=list)
    embedder: str = ""
    elapsed_s: float = 0.0


class NewsSource(Protocol):
    def list_news(
        self,
        since: datetime | None,
        until: datetime | None,
        filters: dict[str, Any] | None = None,
    ) -> list[NewsDoc]: ...

    def get_news(self, news_id: str) -> NewsDoc: ...

    def get_cluster(self, cluster_id: str) -> list[NewsDoc]: ...

    def ingest_url(self, url: str) -> NewsDoc: ...

    def ingest_text(self, text: str, title: str | None = None) -> NewsDoc: ...

    def source_health(self) -> list[SourceHealth]: ...

    def run_ingest(
        self,
        sources: list[str] | None = None,
        backfill_days: int | None = None,
    ) -> RunReport: ...

    def preview(self, news_id: str) -> dict[str, Any]: ...

    def pending_extraction(self) -> list[NewsDoc]: ...


class AnalyticsStore(Protocol):
    def list_docs(self) -> list[AnalyticsDocMeta]: ...

    def get_chunks(self, doc_id: str | None = None) -> list[AnalyticsChunk]: ...

    def get_cards(self, doc_id: str | None = None) -> list[AnalyticsCard]: ...
