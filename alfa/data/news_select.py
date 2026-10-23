"""Общий отбор новостей для заглушки и живого sqlite-источника."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from alfa.core.schemas import NewsDoc
from alfa.core.timeutil import attach_utc


def news_matches(doc: NewsDoc, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    if filters.get("source") and doc.source != filters["source"]:
        return False
    hint = filters.get("topic_hint")
    if hint and hint not in doc.topic_hints:
        return False
    if filters.get("prefilter") and doc.prefilter != filters["prefilter"]:
        return False
    if filters.get("only_primary") and not doc.is_primary:
        return False
    return True


def select_news(
    docs: list[NewsDoc],
    since: datetime | None,
    until: datetime | None,
    filters: dict[str, Any] | None = None,
) -> list[NewsDoc]:
    since_dt = attach_utc(since) if since else None
    until_dt = attach_utc(until) if until else None
    selected: list[NewsDoc] = []
    for doc in docs:
        if not news_matches(doc, filters):
            continue
        if since_dt or until_dt:
            if doc.published_at is None:
                continue
            published = attach_utc(doc.published_at)
            if since_dt and published < since_dt:
                continue
            if until_dt and published > until_dt:
                continue
        selected.append(doc)
    return selected
