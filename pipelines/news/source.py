"""NewsSource поверх sqlite. Полный текст остаётся в хранилище, preview отдаёт фрагмент."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from alfa.core.config import resolve_path
from alfa.core.schemas import NewsDoc
from alfa.data.interfaces import SourceHealth
from alfa.data.news_select import select_news
from alfa.data.stubs.analytics_stub import JsonlAnalyticsStore
from pipelines.news.dedup import build_embedder
from pipelines.news.http import HttpxTransport
from pipelines.news.pipeline import NewsPipeline
from pipelines.news.settings import load_sources, news_settings
from pipelines.news.store import NewsStore, row_to_doc


class LiveNewsSource:
    def __init__(self, pipeline: NewsPipeline):
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LiveNewsSource:
        settings = news_settings(config)
        settings["embedding_cache"] = (config.get("paths") or {}).get("embedding_cache", "data/cache/embeddings")
        settings["db_path"] = str(resolve_path(settings["db_path"]))
        settings["raw_dir"] = str(resolve_path(settings["raw_dir"]))
        sources = load_sources(settings["sources_path"])
        store = NewsStore(resolve_path(settings["db_path"]))
        return cls(NewsPipeline(settings, sources, store, HttpxTransport(settings), build_embedder(settings)))

    def list_news(
        self,
        since: datetime | None,
        until: datetime | None,
        filters: dict[str, Any] | None = None,
    ) -> list[NewsDoc]:
        docs = [row_to_doc(row) for row in self.pipeline.store.all_rows()]
        return select_news(docs, since, until, filters)

    def get_news(self, news_id: str) -> NewsDoc:
        doc = self.pipeline.store.get_doc(news_id)
        if doc is None:
            raise KeyError(news_id)
        return doc

    def get_cluster(self, cluster_id: str) -> list[NewsDoc]:
        return [row_to_doc(row) for row in self.pipeline.store.cluster_rows(cluster_id)]

    def ingest_url(self, url: str) -> NewsDoc:
        return self.pipeline.ingest_url(url)

    def ingest_text(self, text: str, title: str | None = None) -> NewsDoc:
        return self.pipeline.ingest_text(text, title)

    def source_health(self) -> list[SourceHealth]:
        states = {item["name"]: item for item in self.pipeline.store.all_states()}
        health: list[SourceHealth] = []
        for spec in self.pipeline.sources:
            state = states.get(spec.name) or {}
            health.append(
                SourceHealth(
                    name=spec.name,
                    state=str(state.get("state") or "ok"),
                    last_success_at=state.get("last_success_at"),
                    consecutive_errors=int(state.get("consecutive_errors") or 0),
                    empty_ratio=float(state.get("empty_ratio") or 0),
                    new_docs=int(state.get("new_docs") or 0),
                    poll_interval_s=int(state.get("poll_interval_s") or spec.poll_interval_s),
                )
            )
        return health

    def run_ingest(self, sources: list[str] | None = None, backfill_days: int | None = None):
        return self.pipeline.run_ingest(sources, backfill_days)

    def preview(self, news_id: str) -> dict[str, Any]:
        doc = self.get_news(news_id)
        limit = int(self.pipeline.settings["preview_chars"])
        published = doc.published_at.isoformat() if doc.published_at else None
        return {
            "id": doc.id,
            "title": doc.title,
            "snippet": doc.text[:limit],
            "url": doc.url,
            "source": doc.source,
            "published_at": published,
        }

    def pending_extraction(self) -> list[NewsDoc]:
        from alfa.data.interfaces import eligible_for_extraction

        return [row_to_doc(row) for row in self.pipeline.store.queue_rows() if eligible_for_extraction(row_to_doc(row))]


def news_source(config: dict[str, Any]) -> LiveNewsSource:
    return LiveNewsSource.from_config(config)


def analytics_store(config: dict[str, Any]) -> JsonlAnalyticsStore:
    return JsonlAnalyticsStore(config["paths"]["analytics_fixtures"])
