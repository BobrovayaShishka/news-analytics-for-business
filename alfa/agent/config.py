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


@dataclass
class MechanismSpec:
    mechanism_id: str
    directness: str
    sensitivity_requires: tuple[str, ...]
    critical_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    changes_conclusion: tuple[str, ...]
    tools: tuple[ToolStep, ...]
    optional_tools: tuple[ToolStep, ...]


@dataclass
class AgentConfig:
    raw: dict[str, Any]
    mechanisms_raw: dict[str, Any]
    rubric: dict[str, Any]
    mechanisms: dict[str, MechanismSpec]
    questions: dict[str, QuestionSpec]
    version: str
    disclaimer: str
    max_steps: int
    max_repair: int
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
        output[str(field_name)] = QuestionSpec(
            field=str(field_name),
            text=str(payload["text"]),
            why_asked=str(payload["why_asked"]),
            options=options,
        )
    return output


def _mechanisms(raw: dict[str, Any]) -> dict[str, MechanismSpec]:
    output: dict[str, MechanismSpec] = {}
    for name, payload in (raw.get("mechanisms") or {}).items():
        output[str(name)] = MechanismSpec(
            mechanism_id=str(name),
            directness=str(payload.get("directness") or "direct"),
            sensitivity_requires=tuple(str(item) for item in payload.get("sensitivity_requires") or []),
            critical_fields=tuple(str(item) for item in payload.get("critical_fields") or []),
            optional_fields=tuple(str(item) for item in payload.get("optional_fields") or []),
            changes_conclusion=tuple(str(item) for item in payload.get("changes_conclusion") or []),
            tools=_steps(payload.get("tools"), required=True),
            optional_tools=_steps(payload.get("optional_tools"), required=False),
        )
    return output


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
    rubric = _read(RUBRIC_PATH)
    mechanisms = _mechanisms(mechanisms_raw)
    questions = _questions(mechanisms_raw)
    _require_alignment(mechanisms)
    for spec in mechanisms.values():
        for field_name in spec.critical_fields:
            if field_name not in questions:
                raise ValueError(f"no question template for {field_name}")
    return AgentConfig(
        raw=raw,
        mechanisms_raw=mechanisms_raw,
        rubric=rubric,
        mechanisms=mechanisms,
        questions=questions,
        version=str(raw.get("version") or "agent"),
        disclaimer=str(raw["disclaimer"]),
        max_steps=int(raw["max_steps"]),
        max_repair=int(raw["max_repair"]),
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
        base_tools=_steps(raw.get("base_tools"), required=True),
        optional_tools=_steps(raw.get("optional_tools"), required=False),
    )
