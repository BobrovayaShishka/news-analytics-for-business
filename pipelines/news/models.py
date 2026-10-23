"""Записи конвейера новостей. Это не схема выдачи: наружу уходит NewsDoc."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SourceSpec:
    name: str
    type: str
    feed_url: str = ""
    list_url: str = ""
    trust_tier: str = "major_media"
    topic_hints: list[str] = field(default_factory=list)
    timezone: str = "Europe/Moscow"
    poll_interval_s: int = 1800
    fallback_group: str = ""
    role: str = "primary"
    enabled: bool = True
    selectors: dict = field(default_factory=dict)
    strip_patterns: list[str] = field(default_factory=list)
    notes: str = ""
    access: dict = field(default_factory=dict)


@dataclass
class Draft:
    id: str
    source: str
    source_type: str
    trust_tier: str
    url: str
    canonical_url: str
    title: str
    text: str
    served_text: str
    text_len: int
    published_at: datetime | None
    fetched_at: datetime
    lang: str
    content_hash: str
    flags: list[str]
    prefilter: str
    origin: str
    topic_hints: list[str]
    text_hash: str
    simhash: str
    raw_html: bytes | None = None
    http_truncated: bool = False
