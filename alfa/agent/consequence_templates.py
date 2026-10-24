"""Resolve/render шаблонов consequences: mechanisms.yaml + consequences.yaml."""

from __future__ import annotations

from typing import Any

from alfa.agent.config import AgentConfig
from alfa.agent.mechanism_text import factor_phrase, field_value_label
from alfa.agent.state_view import features as state_features
from alfa.agent.state_view import mechanism_ids as state_mechanism_ids
from alfa.agent.state_view import user_provided as state_user_provided
from alfa.ranking.config import RankingConfig, load_ranking_config

NEUTRAL = "Нейтральный эффект при текущих признаках"
_TEMPLATE_KEYS = ("risk_template", "opportunity_template", "watch_template")


class _SafeFormat(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _placeholder_map(
    state: dict,
    config: AgentConfig,
    mechanism_id: str,
    *,
    ranking: RankingConfig | None = None,
) -> dict[str, str]:
    ranking = ranking or load_ranking_config()
    profile = state_features(state)
    provided = state_user_provided(state)
    industry = profile.get("industry_group") or profile.get("industry_theme") or ""
    reasons = state.get("reasons") or {}
    reason = str(reasons.get(mechanism_id) or ranking.reason(mechanism_id) or mechanism_id)
    return {
        "factor": factor_phrase(mechanism_id, profile, config),
        "reason": reason,
        "industry_label": ranking.industry_label(str(industry)) if industry else "отрасли компании",
        "event_type": str(state.get("event_type") or ""),
        "debt_rate_type": field_value_label("debt_rate_type", provided.get("debt_rate_type"), config),
        "fx_expense_share": field_value_label("fx_expense_share", provided.get("fx_expense_share"), config),
        "fx_revenue_share": field_value_label("fx_revenue_share", provided.get("fx_revenue_share"), config),
        "debt_amount": field_value_label("debt_amount", provided.get("debt_amount"), config),
        "debt_maturity": field_value_label("debt_maturity", provided.get("debt_maturity"), config),
    }


def _render(template: str, mapping: dict[str, str]) -> str:
    body = (template or "").strip()
    if not body:
        return ""
    return body.format_map(_SafeFormat(mapping))


def _template_block(raw: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    output: dict[str, str] = {}
    for key in _TEMPLATE_KEYS:
        value = str(raw.get(key) or "").strip()
        if value:
            output[key] = value
    return output


def resolve_templates(state: dict, config: AgentConfig, mechanism_id: str) -> dict[str, str]:
    """Собрать шаблоны: by_field > event_matrix > mechanisms.yaml."""
    spec = config.mechanisms.get(mechanism_id)
    merged: dict[str, str] = {}
    if spec is not None:
        merged = {
            "risk_template": spec.risk_template,
            "opportunity_template": spec.opportunity_template,
            "watch_template": spec.watch_template,
        }
    event_type = str(state.get("event_type") or "")
    matrix = ((config.consequences.get("event_matrix") or {}).get(event_type) or {}).get(mechanism_id)
    merged.update(_template_block(matrix))
    provided = state_user_provided(state)
    by_field = config.consequences.get("by_field") or {}
    for field_name, values in by_field.items():
        value = provided.get(str(field_name))
        if value in (None, "", "unknown"):
            continue
        block = ((values.get(value) or {}).get(mechanism_id) if isinstance(values, dict) else None)
        merged.update(_template_block(block))
    return merged


def rendered_block(
    state: dict,
    config: AgentConfig,
    mechanism_id: str,
    *,
    ranking: RankingConfig | None = None,
) -> dict[str, str | None]:
    """Один проход: resolve + placeholders + render risk/opportunity/watch."""
    ranking = ranking or load_ranking_config()
    templates = resolve_templates(state, config, mechanism_id)
    mapping = _placeholder_map(state, config, mechanism_id, ranking=ranking)
    rendered: dict[str, str | None] = {}
    for key in _TEMPLATE_KEYS:
        text = _render(templates.get(key) or "", mapping)
        rendered[key] = text or None
    if not rendered["risk_template"]:
        profile = state_features(state)
        rendered["risk_template"] = (
            f"Событие может ухудшить показатель компании через "
            f"{factor_phrase(mechanism_id, profile, config)}."
        )
    return rendered


def rendered_for_mechanisms(
    state: dict,
    config: AgentConfig,
    *,
    ranking: RankingConfig | None = None,
) -> dict[str, dict[str, str | None]]:
    ranking = ranking or load_ranking_config()
    return {
        mechanism_id: rendered_block(state, config, mechanism_id, ranking=ranking)
        for mechanism_id in state_mechanism_ids(state)
    }
