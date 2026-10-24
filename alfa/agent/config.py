"""Загрузка agent.yaml, mechanisms.yaml и рубрики влияния.

Критичные поля сверяются с structurally_missing и с requires карты чувствительности.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

from alfa.core.config import PACKAGE_ROOT
from alfa.ranking.config import load_ranking_config
from alfa.settings import load_profile_config

CONFIG_DIR = PACKAGE_ROOT / "config"
AGENT_PATH = CONFIG_DIR / "agent.yaml"
MECHANISMS_PATH = CONFIG_DIR / "mechanisms.yaml"
CONSEQUENCES_PATH = CONFIG_DIR / "consequences.yaml"
RUBRIC_PATH = CONFIG_DIR / "impact_rubric.yaml"

TOOL_NAMES = frozenset(
    {
        "get_profile_view",
        "get_group_context",
        "get_counterparties",
        "get_event",
        "search_alfa_analytics",
        "search_news",
        "external_stats",
        "calc",
        "explain_relevance",
    }
)

# Необязательные шаги, которые LLM может добавить к YAML-плану. Всё остальное — только из tools.
OPTIONAL_TOOL_NAMES = frozenset({"search_news", "external_stats", "get_group_context"})

# Слой alfa — только взгляд банка, не чужой факт ЦБ/Росстата под атрибуцией Альфа.
BANK_VIEW_TYPES = frozenset({"forecast", "opinion", "risk_assessment"})

TOPICS = frozenset(
    {
        "key_rate",
        "fx",
        "inflation",
        "sanctions_trade",
        "taxes",
        "commodities_gold",
        "labor",
        "consumer_demand",
        "sector_regulation",
        "ma",
        "other_econ",
        "from_event",
    }
)


def _read(path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


@dataclass
class ToolStep:
    tool: str
    args: dict[str, Any]
    required: bool = True


@dataclass
class QuestionSpec:
    field: str
    text: str
    why_asked: str
    options: list[dict[str, str]]
    short_label: str = ""
    unknown_label: str = ""
    known_phrases: dict[str, str] = field(default_factory=dict)
    value_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MechanismPresentation:
    headline: str = ""
    headlines: tuple[dict[str, Any], ...] = ()
    factor: str = ""
    factor_parts: tuple[dict[str, Any], ...] = ()
    effect: str = ""
    feature_ref: str = "industry_group"
    feature_ref_prefer: tuple[dict[str, Any], ...] = ()
    tags: tuple[str, ...] = ()


@dataclass
class MechanismSpec:
    mechanism_id: str
    directness: str
    sensitivity_requires: tuple[str, ...]
    critical_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    changes_conclusion: tuple[str, ...]
    risk_template: str
    opportunity_template: str
    watch_template: str
    tools: tuple[ToolStep, ...]
    optional_tools: tuple[ToolStep, ...]
    presentation: MechanismPresentation = field(default_factory=MechanismPresentation)

    @property
    def headline(self) -> str:
        return self.presentation.headline

    @property
    def headlines(self) -> tuple[dict[str, Any], ...]:
        return self.presentation.headlines

    @property
    def factor(self) -> str:
        return self.presentation.factor

    @property
    def factor_parts(self) -> tuple[dict[str, Any], ...]:
        return self.presentation.factor_parts

    @property
    def effect(self) -> str:
        return self.presentation.effect

    @property
    def feature_ref(self) -> str:
        return self.presentation.feature_ref

    @property
    def feature_ref_prefer(self) -> tuple[dict[str, Any], ...]:
        return self.presentation.feature_ref_prefer

    @property
    def tags(self) -> tuple[str, ...]:
        return self.presentation.tags


@dataclass
class AgentConfig:
    raw: dict[str, Any]
    mechanisms_raw: dict[str, Any]
    consequences: dict[str, Any]
    rubric: dict[str, Any]
    mechanisms: dict[str, MechanismSpec]
    questions: dict[str, QuestionSpec]
    version: str
    disclaimer: str
    max_steps: int
    max_repair: int
    max_replan: int
    max_seconds: float
    max_tokens: int
    session_max_tokens: int
    session_max_insights: int
    tool_timeout_s: float
    prompt_retention_days: int
    max_questions_per_insight: int
    max_questions_per_session: int
    min_priority: str
    stale_after_days: int
    context_tokens: int
    chars_per_token: int
    monitor_rate: float
    max_chat_turns: int
    max_scenarios: int
    max_calls: dict[str, int]
    roles: dict[str, str]
    prompt_versions: dict[str, str]
    rub_per_1k: float
    data_policy: dict[str, Any]
    base_tools: tuple[ToolStep, ...]
    optional_tools: tuple[ToolStep, ...] = field(default_factory=tuple)
    role_presentation: dict[str, Any] = field(default_factory=dict)
    presentation: dict[str, str] = field(default_factory=dict)

    def prompt_path(self, name: str):
        version = self.prompt_versions[name]
        return PACKAGE_ROOT / "agent" / "prompts" / f"{version}.md"


def _steps(items: list[dict[str, Any]] | None, *, required: bool) -> tuple[ToolStep, ...]:
    steps: list[ToolStep] = []
    for item in items or []:
        tool = str(item["tool"])
        if tool not in TOOL_NAMES:
            raise ValueError(f"unknown tool in config: {tool}")
        args = dict(item.get("args") or {})
        steps.append(ToolStep(tool=tool, args=args, required=bool(item.get("required", required))))
    return tuple(steps)


def _questions(raw: dict[str, Any]) -> dict[str, QuestionSpec]:
    output: dict[str, QuestionSpec] = {}
    for field_name, payload in (raw.get("questions") or {}).items():
        options = []
        for option in payload.get("options") or []:
            options.append({"label": str(option["label"]), "value": str(option["value"])})
        phrases = {
            str(key): str(value)
            for key, value in (payload.get("known_phrases") or {}).items()
        }
        value_labels = {
            str(key): str(value)
            for key, value in (payload.get("value_labels") or {}).items()
        }
        output[str(field_name)] = QuestionSpec(
            field=str(field_name),
            text=str(payload["text"]),
            why_asked=str(payload["why_asked"]),
            options=options,
            short_label=str(payload.get("short_label") or ""),
            unknown_label=str(payload.get("unknown_label") or ""),
            known_phrases=phrases,
            value_labels=value_labels,
        )
    return output


def _presentation(payload: dict[str, Any], name: str) -> dict[str, Any]:
    block = payload.get("presentation")
    if not isinstance(block, dict):
        raise ValueError(f"{name}: нужен блок presentation (headline, factor, effect, feature_ref)")
    headline = str(block.get("headline") or "").strip()
    factor = str(block.get("factor") or "").strip()
    effect = str(block.get("effect") or "").strip()
    feature_ref = str(block.get("feature_ref") or "").strip()
    if not headline or not factor or not effect or not feature_ref:
        raise ValueError(f"{name}: presentation требует headline, factor, effect, feature_ref")
    headlines = tuple(dict(item) for item in (block.get("headlines") or []) if isinstance(item, dict))
    factor_parts = tuple(dict(item) for item in (block.get("factor_parts") or []) if isinstance(item, dict))
    prefer = tuple(dict(item) for item in (block.get("feature_ref_prefer") or []) if isinstance(item, dict))
    for item in prefer:
        if not item.get("ref"):
            raise ValueError(f"{name}: feature_ref_prefer требует ref")
    tags = tuple(str(item) for item in (block.get("tags") or []))
    return {
        "headline": headline,
        "headlines": headlines,
        "factor": factor,
        "factor_parts": factor_parts,
        "effect": effect,
        "feature_ref": feature_ref,
        "feature_ref_prefer": prefer,
        "tags": tags,
    }


def _check_optional(steps: tuple[ToolStep, ...], where: str) -> None:
    bad = sorted({step.tool for step in steps if step.tool not in OPTIONAL_TOOL_NAMES})
    if bad:
        raise ValueError(f"{where}: optional_tools только {sorted(OPTIONAL_TOOL_NAMES)}, лишние: {bad}")


def _mechanisms(raw: dict[str, Any]) -> dict[str, MechanismSpec]:
    output: dict[str, MechanismSpec] = {}
    for name, payload in (raw.get("mechanisms") or {}).items():
        optional = _steps(payload.get("optional_tools"), required=False)
        _check_optional(optional, name)
        risk = str(payload.get("risk_template") or "").strip()
        opportunity = str(payload.get("opportunity_template") or "").strip()
        watch = str(payload.get("watch_template") or "").strip()
        if not risk or not opportunity or not watch:
            raise ValueError(f"{name}: нужны risk_template, opportunity_template и watch_template")
        present = _presentation(payload, name)
        output[str(name)] = MechanismSpec(
            mechanism_id=str(name),
            directness=str(payload.get("directness") or "direct"),
            sensitivity_requires=tuple(str(item) for item in payload.get("sensitivity_requires") or []),
            critical_fields=tuple(str(item) for item in payload.get("critical_fields") or []),
            optional_fields=tuple(str(item) for item in payload.get("optional_fields") or []),
            changes_conclusion=tuple(str(item) for item in payload.get("changes_conclusion") or []),
            risk_template=risk,
            opportunity_template=opportunity,
            watch_template=watch,
            tools=_steps(payload.get("tools"), required=True),
            optional_tools=optional,
            presentation=MechanismPresentation(
                headline=present["headline"],
                headlines=present["headlines"],
                factor=present["factor"],
                factor_parts=present["factor_parts"],
                effect=present["effect"],
                feature_ref=present["feature_ref"],
                feature_ref_prefer=present["feature_ref_prefer"],
                tags=present["tags"],
            ),
        )
    return output


def _require_consequences(consequences: dict[str, Any], mechanisms: dict[str, MechanismSpec]) -> None:
    by_field = consequences.get("by_field") or {}
    if not isinstance(by_field, dict):
        raise ValueError("consequences.yaml: by_field must be a mapping")
    for field_name, values in by_field.items():
        if not isinstance(values, dict):
            raise ValueError(f"consequences.yaml: by_field.{field_name} must be a mapping")
        for value, mechs in values.items():
            if not isinstance(mechs, dict):
                raise ValueError(f"consequences.yaml: by_field.{field_name}.{value} must be a mapping")
            for mechanism_id in mechs:
                if mechanism_id not in mechanisms:
                    raise ValueError(f"consequences.yaml: unknown mechanism in by_field: {mechanism_id}")
    matrix = consequences.get("event_matrix") or {}
    if not isinstance(matrix, dict):
        raise ValueError("consequences.yaml: event_matrix must be a mapping")
    for event_type, mechs in matrix.items():
        if not isinstance(mechs, dict):
            raise ValueError(f"consequences.yaml: event_matrix.{event_type} must be a mapping")
        for mechanism_id in mechs:
            if mechanism_id not in mechanisms:
                raise ValueError(f"consequences.yaml: unknown mechanism in event_matrix: {mechanism_id}")
    scenario_matrix = consequences.get("scenario_matrix") or {}
    if not isinstance(scenario_matrix, dict):
        raise ValueError("consequences.yaml: scenario_matrix must be a mapping")
    for field_name, values in scenario_matrix.items():
        if not isinstance(values, dict):
            raise ValueError(f"consequences.yaml: scenario_matrix.{field_name} must be a mapping")
        for value, items in values.items():
            if not isinstance(items, list):
                raise ValueError(f"consequences.yaml: scenario_matrix.{field_name}.{value} must be a list")
            for index, item in enumerate(items):
                path = f"scenario_matrix.{field_name}.{value}[{index}]"
                if not isinstance(item, dict):
                    raise ValueError(f"consequences.yaml: {path} must be a mapping")
                if not str(item.get("name") or "").strip():
                    raise ValueError(f"consequences.yaml: {path}.name required")
                if not str(item.get("logic") or "").strip():
                    raise ValueError(f"consequences.yaml: {path}.logic required")
                plausibility = str(item.get("plausibility") or "medium").strip()
                if plausibility not in {"low", "medium", "high"}:
                    raise ValueError(f"consequences.yaml: {path}.plausibility must be low|medium|high")


def _require_alignment(mechanisms: dict[str, MechanismSpec]) -> None:
    structural = set(load_profile_config()["structurally_missing"])
    ranking = load_ranking_config()
    expected = [str(item) for item in ranking.sensitivity.get("mechanisms") or []]
    if set(expected) != set(mechanisms):
        missing = sorted(set(expected) - set(mechanisms))
        extra = sorted(set(mechanisms) - set(expected))
        raise ValueError(f"mechanisms drift, missing={missing}, extra={extra}")
    requires: dict[str, set[str]] = {}
    for rule in ranking.rules:
        requires.setdefault(rule.mechanism_id, set()).update(rule.requires)
    for name, spec in mechanisms.items():
        if spec.directness not in {"direct", "indirect"}:
            raise ValueError(f"{name}: directness")
        if set(spec.changes_conclusion) != set(spec.critical_fields):
            raise ValueError(f"{name}: changes_conclusion must match critical_fields")
        unknown = set(spec.critical_fields) | set(spec.optional_fields)
        if not unknown <= structural:
            raise ValueError(f"{name}: fields outside structurally_missing: {sorted(unknown - structural)}")
        if set(spec.sensitivity_requires) != requires.get(name, set()):
            raise ValueError(
                f"{name}: sensitivity_requires {sorted(spec.sensitivity_requires)} "
                f"!= map {sorted(requires.get(name, set()))}"
            )


@lru_cache(maxsize=1)
def load_agent_config() -> AgentConfig:
    raw = _read(AGENT_PATH)
    mechanisms_raw = _read(MECHANISMS_PATH)
    consequences = _read(CONSEQUENCES_PATH)
    rubric = _read(RUBRIC_PATH)
    mechanisms = _mechanisms(mechanisms_raw)
    questions = _questions(mechanisms_raw)
    _require_alignment(mechanisms)
    _require_consequences(consequences, mechanisms)
    base_tools = _steps(raw.get("base_tools"), required=True)
    optional_tools = _steps(raw.get("optional_tools"), required=False)
    _check_optional(optional_tools, "agent.yaml")
    for spec in mechanisms.values():
        for field_name in spec.critical_fields:
            if field_name not in questions:
                raise ValueError(f"no question template for {field_name}")
    return AgentConfig(
        raw=raw,
        mechanisms_raw=mechanisms_raw,
        consequences=consequences,
        rubric=rubric,
        mechanisms=mechanisms,
        questions=questions,
        version=str(raw.get("version") or "agent"),
        disclaimer=str(raw["disclaimer"]),
        max_steps=int(raw["max_steps"]),
        max_repair=int(raw["max_repair"]),
        max_replan=int(raw.get("max_replan", 1)),
        max_seconds=float(raw["max_seconds"]),
        max_tokens=int(raw["max_tokens"]),
        session_max_tokens=int(raw["session_max_tokens"]),
        session_max_insights=int(raw["session_max_insights"]),
        tool_timeout_s=float(raw["tool_timeout_s"]),
        prompt_retention_days=int(raw["prompt_retention_days"]),
        max_questions_per_insight=int(raw["max_questions_per_insight"]),
        max_questions_per_session=int(raw["max_questions_per_session"]),
        min_priority=str(raw["min_priority"]),
        stale_after_days=int(raw["stale_after_days"]),
        context_tokens=int(raw["context_tokens"]),
        chars_per_token=int(raw["chars_per_token"]),
        monitor_rate=float(raw["monitor_rate"]),
        max_chat_turns=int(raw["max_chat_turns"]),
        max_scenarios=int(raw["max_scenarios"]),
        max_calls={str(key): int(value) for key, value in (raw.get("max_calls") or {}).items()},
        roles={str(key): str(value) for key, value in (raw.get("roles") or {}).items()},
        prompt_versions={str(key): str(value) for key, value in (raw.get("prompt_versions") or {}).items()},
        rub_per_1k=float((raw.get("cost") or {}).get("rub_per_1k_tokens") or 0),
        data_policy=dict(raw.get("data_policy") or {}),
        base_tools=base_tools,
        optional_tools=optional_tools,
        role_presentation=dict(raw.get("role_presentation") or {}),
        presentation={str(key): str(value) for key, value in (raw.get("presentation") or {}).items()},
    )
