"""Тексты механизмов из YAML: заголовки, фактор, feature_ref, подписи полей."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alfa.agent.config import AgentConfig
from alfa.agent.state_view import mechanism_ids as state_mechanism_ids
from alfa.agent.state_view import user_provided as state_user_provided


def _match_when(when: dict[str, Any], *, features: dict[str, Any] | None = None, provided: dict[str, str] | None = None) -> bool:
    if not when:
        return True
    if "feature" in when:
        actual = (features or {}).get(str(when["feature"]))
        if "eq" in when:
            return actual == when["eq"]
        return bool(actual)
    if "field" in when:
        actual = (provided or {}).get(str(when["field"]))
        if "eq" in when:
            return actual == when["eq"]
        return bool(actual)
    return False


def feature_ref(mechanism_id: str, features: dict[str, Any], config: AgentConfig) -> str:
    spec = config.mechanisms.get(mechanism_id)
    if spec is None:
        return "industry_group"
    for rule in spec.feature_ref_prefer:
        if _match_when(rule, features=features):
            return str(rule["ref"])
    return spec.feature_ref or "industry_group"


def factor_phrase(mechanism_id: str, features: dict[str, Any], config: AgentConfig) -> str:
    spec = config.mechanisms.get(mechanism_id)
    if spec is None:
        return mechanism_id
    parts: list[str] = []
    for part in spec.factor_parts:
        if _match_when(part, features=features):
            label = str(part.get("label") or "").strip()
            if label:
                parts.append(label)
    if parts:
        return " и ".join(parts)
    return spec.factor or mechanism_id


def effect_clause(mechanism_id: str, reasons: dict[str, Any], config: AgentConfig) -> str:
    reason = str(reasons.get(mechanism_id) or "").strip().rstrip(".")
    if reason and reason != mechanism_id:
        return reason[0].lower() + reason[1:] if reason else reason
    spec = config.mechanisms.get(mechanism_id)
    if spec is not None and spec.effect:
        return spec.effect
    return mechanism_id


def headline_for(mechanism_id: str, provided: dict[str, str], config: AgentConfig) -> str:
    linked = str((config.presentation or {}).get("headline_linked") or "Событие связано с профилем компании")
    spec = config.mechanisms.get(mechanism_id)
    if spec is None:
        return linked
    for item in spec.headlines:
        when_block = item.get("when") if isinstance(item.get("when"), dict) else {}
        text = str(item.get("text") or "").strip()
        if text and _match_when(when_block, provided=provided):
            return text
    return spec.headline or linked


def field_short_label(field_name: str, config: AgentConfig) -> str:
    question = config.questions.get(field_name)
    if question is not None and question.short_label:
        return question.short_label
    return field_name


def field_value_label(field_name: str, value: str | None, config: AgentConfig) -> str:
    question = config.questions.get(field_name)
    if value in (None, "", "unknown"):
        if question is not None and question.unknown_label:
            return question.unknown_label
        return "неизвестно"
    if question is not None:
        if value in question.value_labels:
            return question.value_labels[str(value)]
        for option in question.options:
            if option.get("value") == value:
                return str(option.get("label") or value)
    return str(value)


def known_phrase(field_name: str, value: str, config: AgentConfig) -> str | None:
    question = config.questions.get(field_name)
    if question is None:
        return None
    phrase = question.known_phrases.get(value)
    return str(phrase) if phrase else None


def mechanisms_with_tag(config: AgentConfig, tag: str) -> tuple[str, ...]:
    return tuple(name for name, spec in config.mechanisms.items() if tag in spec.tags)


def role_hint(role: str | None, config: AgentConfig) -> str:
    block = config.role_presentation or {}
    key = role if role in {"cfo", "ceo"} else "default"
    return str(block.get(key) or block.get("default") or "Порядок: что произошло, почему важно, на что смотреть.")


def calc_mechanism(fn: str | None, state: dict, config: AgentConfig) -> str | None:
    target = str(fn or "")
    if not target:
        return None
    active = set(state_mechanism_ids(state))
    for mechanism_id, spec in config.mechanisms.items():
        for step in spec.tools:
            if step.tool == "calc" and str(step.args.get("fn") or "") == target:
                if mechanism_id in active:
                    return mechanism_id
    return None


def condition_phrase(state: dict, config: AgentConfig, mechanism_ids: list[str]) -> str:
    provided = state_user_provided(state)
    known: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for mechanism_id in mechanism_ids:
        spec = config.mechanisms.get(mechanism_id)
        if spec is None:
            continue
        for field_name in spec.critical_fields:
            if field_name in seen:
                continue
            seen.add(field_name)
            value = provided.get(field_name)
            label = field_short_label(field_name, config)
            if value and value != "unknown":
                phrase = known_phrase(field_name, value, config)
                known.append(phrase or f"уточнённого {label}")
            else:
                missing.append(label)
    if known:
        return "при условии " + " и ".join(known)
    if missing:
        return "при условии уточнения по " + " и ".join(missing[:2])
    return ""


def prioritize(
    ids: list[str],
    prefer: Callable[[str], bool],
    *,
    rank: Callable[[str], int] | None = None,
) -> list[str]:
    """Стабильно: сначала ids, где prefer=True (опционально по rank), затем остальные."""
    head = [claim_id for claim_id in ids if prefer(claim_id)]
    if rank is not None:
        head = sorted(head, key=rank)
    seen = set(head)
    return head + [claim_id for claim_id in ids if claim_id not in seen]
