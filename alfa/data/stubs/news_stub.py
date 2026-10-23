"""Чтение JSONL-фикстур новостей. Путь к каталогу берётся из config.yaml."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alfa.core.config import resolve_path
from alfa.core.ids import make_news_id
from alfa.core.schemas import NewsDoc
from alfa.data.interfaces import IngestError, RunReport, SourceHealth, eligible_for_extraction
from alfa.data.news_select import select_news


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl_dir(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


class JsonlNewsSource:
    def __init__(self, directory: str | Path):
        self.directory = resolve_path(directory)
        self._docs = [NewsDoc.model_validate(row) for row in iter_jsonl_dir(self.directory)]
        self._by_id = {doc.id: doc for doc in self._docs}
        if len(self._by_id) != len(self._docs):
            raise ValueError(f"duplicate news ids in {self.directory}")

    def list_news(
        self,
        since: datetime | None,
        until: datetime | None,
        filters: dict[str, Any] | None = None,
    ) -> list[NewsDoc]:
        return select_news(self._docs, since, until, filters)

    def get_news(self, news_id: str) -> NewsDoc:
        try:
            return self._by_id[news_id]
        except KeyError as exc:
            raise KeyError(news_id) from exc

    def get_cluster(self, cluster_id: str) -> list[NewsDoc]:
        docs = [doc for doc in self._docs if doc.cluster_id == cluster_id]
        docs.sort(key=lambda doc: (not doc.is_primary, doc.published_at or datetime.max.replace(tzinfo=timezone.utc)))
        return docs

    def ingest_url(self, url: str) -> NewsDoc:
        raise IngestError("заглушка не ходит в сеть")

    def ingest_text(self, text: str, title: str | None = None) -> NewsDoc:
        body = (text or "").strip()
        if not body:
            raise IngestError("текст не удалось получить")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        url = "urn:user-text:" + hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
        doc = NewsDoc(
            id=make_news_id("user", url),
            source="user",
            url=url,
            canonical_url=url,
            published_at=None,
            title=title or body[:80],
            text=body,
            text_len=len(body),
            source_type="user_submitted",
            fetched_at=now,
            trust_tier="unverified",
            origin="on_demand",
            flags=["date_missing"],
            extraction_status="pending",
        )
        self._by_id[doc.id] = doc
        self._docs = [item for item in self._docs if item.id != doc.id]
        self._docs.append(doc)
        return doc

    def source_health(self) -> list[SourceHealth]:
        return []

    def run_ingest(self, sources: list[str] | None = None, backfill_days: int | None = None) -> RunReport:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return RunReport(run_id="stub", started_at=now, finished_at=now, embedder="stub")

    def preview(self, news_id: str) -> dict[str, Any]:
        doc = self.get_news(news_id)
        return {
            "id": doc.id,
            "title": doc.title,
            "snippet": doc.text[:400],
            "url": doc.url,
            "source": doc.source,
            "published_at": doc.published_at.isoformat() if doc.published_at else None,
        }

    def pending_extraction(self) -> list[NewsDoc]:
        return [doc for doc in self._docs if eligible_for_extraction(doc)]
