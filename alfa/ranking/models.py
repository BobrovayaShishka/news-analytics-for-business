"""Модели ранжирования. Идентификаторы контрагентов и членов группы наружу не отдаются."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FiredRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    mechanism_id: str
    contribution: float


class SignalSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    via: str
    mechanism_id: str


class LinkHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["payment_link_exposure"]
    text: str
    evidence: dict[str, Any]
    confidence: float


class ScoredPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    client_id: str
    score: float
    relevance: float
    importance: float
    components: dict[str, float]
    matched_rules: list[FiredRule]
    reliability: str
    priority: str
    flags: list[str]
    mechanism_ids: list[str] = Field(default_factory=list)
    cluster_id: str = ""
    revision: int = 1
    fact_signature: str = ""
    hypotheses: list[LinkHypothesis] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    score_confidence: float = 0.0
    config_version: str = ""
    weight_version: str = ""
    group_signal: SignalSource | None = None
    graph_signal: SignalSource | None = None
    analytics_refs: list[str] = Field(default_factory=list)
    above_threshold: bool = False


class FeedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    client_id: str
    score: float
    priority: str
    mechanism_ids: list[str]
    reasons: list[str]
    reliability: str
    analytics_refs: list[str]
    flags: list[str]
    hypotheses: list[LinkHypothesis] = Field(default_factory=list)


class Feed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    as_of: datetime
    items: list[FeedItem]
    empty_state: bool
    empty_reason: str | None = None
    config_version: str = ""
    weight_version: str = ""
    role: str | None = None


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    client_id: str
    fired_rules: list[FiredRule]
    components: dict[str, float]
    group_signal: SignalSource | None = None
    graph_signal: SignalSource | None = None
    missing_features: list[str]
    reliability: str
    score_confidence: float
    config_version: str
    weight_version: str
    score: float
    relevance: float
    importance: float
    priority: str
    flags: list[str]
    hypotheses: list[LinkHypothesis] = Field(default_factory=list)
    eligible: bool = False
    above_threshold: bool = False


@dataclass
class ClientRow:
    """Одна строка CLIENT. None = признак неизвестен, это не ноль."""

    client_id: str
    client_type: str = "CLIENT"
    conflicting_records: bool = False
    profile_version: str = "v1"
    has_credit: bool | None = None
    has_leasing: bool | None = None
    has_deposit: bool | None = None
    has_guarantees: bool | None = None
    has_factoring: bool | None = None
    has_payroll: bool | None = None
    has_rko: bool | None = None
    industry_group: str | None = None
    themes: list[str] | None = None
    okved_code: str | None = None
    region: str | None = None
    size_class: str | None = None
    workers_class: int | None = None
    revenue: float | None = None
    fx_exposure: str | None = None
    fx_confidence: str | None = None
    fts_countries: list[str] | None = None
    currencies: list[str] | None = None
    group_id: str | None = None
    group_size_reported: int | None = None
    key_owner_known: bool | None = None
    has_payment_data: bool | None = None
    industry_mix_in: dict[str, float] | None = None
    industry_mix_out: dict[str, float] | None = None
    fts_counterparty_volume_share: float | None = None
    known_counterparty_volume_share: float | None = None
    reliability: str | None = None
    n_edges: int | None = None
    intra_group_payment_share: float | None = None
    text: str = ""


@dataclass
class RankEvent:
    event_id: str
    event_type: str
    published_at: datetime
    extraction_status: str = "ok"
    is_business_relevant: bool = True
    is_primary: bool = True
    trust_tier: str = "major_media"
    cluster_id: str = ""
    revision: int = 1
    fact_signature: str = ""
    magnitude: float = 0.0
    n_sources: int = 1
    echo_count: int = 0
    entities: list[dict[str, str]] = field(default_factory=list)
    title: str = ""
    text: str = ""

    def signature(self) -> str:
        stamp = self.published_at.isoformat()
        return f"{self.revision}|{self.fact_signature}|{self.event_type}|{stamp}|{self.text}"


@dataclass
class CardView:
    card_id: str
    topics: list[str]
    published_at: datetime
    verified: str = "manual"


@dataclass
class LinkedEntity:
    kind: str
    value: str
    fuzzy: bool
    raw: str
