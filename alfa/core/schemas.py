"""Pydantic-модели дня 2. Все enum-коды строятся из taxonomy.yaml."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alfa.core.taxonomy import get_taxonomy


def _str_enum(name: str, group: str) -> type[Enum]:
    codes = get_taxonomy().codes(group)
    return Enum(name, {code: code for code in codes}, type=str)  # type: ignore[return-value]


EventType = _str_enum("EventType", "event_type")
Topic = _str_enum("Topic", "topic")
Stance = _str_enum("Stance", "stance")
ClaimType = _str_enum("ClaimType", "claim_type")
Section = _str_enum("Section", "section")
Verified = _str_enum("Verified", "verified")
EntityType = _str_enum("EntityType", "entity_type")
HitKind = _str_enum("HitKind", "hit_kind")
ExtractionStatus = _str_enum("ExtractionStatus", "extraction_status")


def _bbox(value: list[float]) -> list[float]:
    if len(value) != 4:
        raise ValueError("bbox_rel must be [x0, y0, x1, y1]")
    if any(item < 0 or item > 1 for item in value):
        raise ValueError("bbox_rel values must be in 0..1")
    return [float(item) for item in value]


class NewsDoc(BaseModel):
    """Документ новости. published_at пустой только вместе с флагом date_missing."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    url: str
    published_at: datetime | None = None
    title: str
    text: str
    source_type: str = "news"
    fetched_at: datetime
    lang: str = "ru"
    trust_tier: str = "major_media"
    canonical_url: str = ""
    text_len: int = 0
    updated_at: datetime | None = None
    revision: int = 1
    content_hash: str = ""
    raw_ref: str | None = None
    flags: list[str] = Field(default_factory=list)
    prefilter: str = "unknown"
    cluster_id: str | None = None
    is_primary: bool = True
    echo_count: int = 0
    echo_sources: list[str] = Field(default_factory=list)
    extraction_status: str = "pending"
    origin: str = "scheduled"
    topic_hints: list[str] = Field(default_factory=list)


class AnalyticsDocMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    published_at: datetime


class AnalyticsChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    page: int = Field(ge=1)
    bbox_rel: list[float]
    text: str
    section: Section
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    verified: Verified
    published_at: datetime

    @field_validator("bbox_rel")
    @classmethod
    def _check_bbox(cls, value: list[float]) -> list[float]:
        return _bbox(value)


class AnalyticsCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_id: str
    doc_id: str
    page: int = Field(ge=1)
    bbox_rel: list[float]
    topic: list[Topic] = Field(min_length=1)
    claim_type: ClaimType
    claim: str = Field(min_length=1)
    stance: Stance
    metric: str | None = None
    value: str | None = None
    unit: str | None = None
    horizon: str | None = None
    revision: str | None = None
    published_at: datetime
    verified: Verified

    @field_validator("bbox_rel")
    @classmethod
    def _check_bbox(cls, value: list[float]) -> list[float]:
        return _bbox(value)


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    type: EntityType


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    value: str | None = None
    unit: str | None = None
    date: str | None = None
    subject: str | None = None
    evidence_span: str = ""
    unverified: bool = False


class SpanNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_span: str = ""


class ExtractionValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unverified_numbers: list[str] = Field(default_factory=list)
    unverified_dates: list[str] = Field(default_factory=list)
    missing_spans: list[str] = Field(default_factory=list)


class EventExtraction(BaseModel):
    """event_type пустой только при extraction_status=failed: тип не выдумывается."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    event_type: EventType | None = None
    secondary_types: list[EventType] = Field(default_factory=list)
    is_business_relevant: bool | None = None
    entities: list[Entity] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    risks: list[SpanNote] = Field(default_factory=list)
    opportunities: list[SpanNote] = Field(default_factory=list)
    extraction_status: ExtractionStatus
    validation: ExtractionValidation = Field(default_factory=ExtractionValidation)
    model: str = ""
    prompt_version: str = ""
    trace_id: str = ""
    truncated: bool = False


class ScoreComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bm25: float | None = None
    dense: float | None = None


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: HitKind
    id: str
    doc_id: str
    page: int | None = None
    text: str
    score: float
    score_components: ScoreComponents
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    content: str | None = None
    parsed: Any = None
    error: str | None = None
    cache_hit: bool = False
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float | None = None
    model: str | None = None
    raw: str | None = None
