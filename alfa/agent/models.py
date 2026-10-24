"""Модели агентного слоя. Контракт инсайта — раздел 11 ТЗ."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from alfa.ranking.models import RelevanceSignal as RelevanceSignalView


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    allowed_company_ids: list[str]
    active_client_id: str
    role: Literal["ceo", "cfo", "default"] = "default"
    as_of: datetime | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    redactions: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    layer: Literal["fact", "alfa", "model", "hypothesis"]
    text: str
    refs: list[str] = Field(default_factory=list)
    feature_refs: list[str] = Field(default_factory=list)
    mechanism_id: str | None = None
    calc_refs: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
    diverges_from_bank_view: bool = False


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    logic: str
    plausibility: Literal["low", "medium", "high"]
    support_refs: list[str] = Field(default_factory=list)


class DraftRegistry(BaseModel):
    """Реестр интерпретации. Уровень влияния сюда не входит: его ставит рубрика."""

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim]
    headline: str
    what_happened: list[str] = Field(default_factory=list)
    why_it_matters: list[str] = Field(default_factory=list)
    what_to_watch: list[str] = Field(default_factory=list)
    impact_rationale: str = ""
    scenarios: list[Scenario] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class PlanStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = ""
    required: bool = False


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlanStepDraft]
    stop_conditions: list[str] = Field(default_factory=list)
    reasoning: str = ""


class RepairDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replacements: list[Claim] = Field(default_factory=list)


class CalcView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calc_id: str
    fn: str
    formula: str
    inputs: dict[str, Any]
    assumptions: list[str]
    result: float | None
    unit: str
    illustrative: bool
    text: str
    version: str = "calc-v1"


class FeatureUsed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    value: Any = None


class RelevanceView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = 0.0
    relevance: float = 0.0
    importance: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)
    reliability: str = "none"
    mechanism_ids: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    above_threshold: bool = False
    summary: str = ""
    signals: list[RelevanceSignalView] = Field(default_factory=list)


class UserFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str
    answered_at: str
    verified_by_bank: bool = False


class PendingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    text: str
    options: list[str]
    why_asked: str


class ModelInference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impact_level: Literal["low", "medium", "high"]
    impact_definition: str
    impact_rationale: str
    quantification: list[CalcView] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    diverges_from_bank_view: bool = False
    confidence: float = 0.0


class VerificationView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage1: dict[str, Any] = Field(default_factory=dict)
    stage2: dict[str, Any] = Field(default_factory=dict)
    iterations: int = 0
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    stale_claims: list[str] = Field(default_factory=list)
    layer_violations: list[str] = Field(default_factory=list)


class SourceRef(BaseModel):
    """Первоисточник вывода: url и позиция в тексте, не только id ссылки."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["external_news", "bank_analytics", "company_data", "calc", "external_stats"]
    title: str
    locator: str | None = None
    url: str | None = None
    evidence_span: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class Insight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    insight_id: str
    trace_id: str
    status: Literal["published", "published_with_caveats", "withheld"]
    event_id: str
    client_id: str
    as_of: str
    priority: Literal["urgent", "regular"]
    role: Literal["ceo", "cfo", "default"]
    headline: str
    what_happened: str
    why_it_matters: str
    what_to_watch: str
    claims: list[Claim] = Field(default_factory=list)
    confirmed_facts: list[Claim] = Field(default_factory=list)
    alfa_analytics: list[Claim] = Field(default_factory=list)
    model_inference: ModelInference
    hypotheses: list[Claim] = Field(default_factory=list)
    user_provided: list[UserFact] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    client_features_used: list[FeatureUsed] = Field(default_factory=list)
    relevance: RelevanceView = Field(default_factory=RelevanceView)
    missing_data: list[str] = Field(default_factory=list)
    verification: VerificationView = Field(default_factory=VerificationView)
    pending_question: PendingQuestion | None = None
    flags: list[str] = Field(default_factory=list)
    disclaimer: str
    sources: list[SourceRef] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    client_id: str
    user_id: str
    allowed_company_ids: list[str]
    role: Literal["ceo", "cfo", "default"] = "default"
    as_of: str | None = None
    active_client_id: str | None = None


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    allowed_company_ids: list[str]
    value: str
    field: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    allowed_company_ids: list[str]
    question: str
    cut: Literal["cost_side", "credit_portfolio", "counterparties", "group", "scenario", "source"] | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    allowed_company_ids: list[str]
    insight_id: str
    label: Literal["useful", "irrelevant", "inaccurate"]
    reason: str = ""


class CheckNewsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    allowed_company_ids: list[str]
    client_id: str
    text: str | None = None
    url: str | None = None
    title: str | None = None


class AgentError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message
