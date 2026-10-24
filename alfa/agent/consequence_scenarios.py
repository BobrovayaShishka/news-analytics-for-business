"""Сценарии клиентских последствий: scenario_matrix + fallback по механизмам."""

from __future__ import annotations

from typing import Any

from alfa.agent.config import AgentConfig
from alfa.agent.consequence_templates import NEUTRAL, rendered_block
from alfa.agent.mechanism_text import factor_phrase
from alfa.agent.state_view import features as state_features
from alfa.agent.state_view import mechanism_ids as state_mechanism_ids
from alfa.agent.state_view import user_provided as state_user_provided
from alfa.ranking.config import RankingConfig, load_ranking_config


def _fact_refs(state: dict) -> list[str]:
    facts = ((state.get("evidence") or {}).get("event") or {}).get("facts") or []
    refs = [f"fact:{item.get('id') or f'f{index}'}" for index, item in enumerate(facts, start=1) if item.get("claim")]
    return refs or ["fact:f1"]


def _card_refs(state: dict) -> list[str]:
    return [str(card.get("card_id")) for card in (state.get("evidence") or {}).get("cards") or [] if card.get("card_id")]


def _calc_refs(state: dict) -> list[str]:
    return [str(item.get("calc_id")) for item in state.get("calc_results") or [] if item.get("calc_id")]


def support_refs(state: dict, refs: list[str] | None = None, *, prefer_calc: bool = False) -> list[str]:
    output: list[str] = []
    if refs:
        output.extend(str(item) for item in refs if item)
    if prefer_calc:
        output.extend(_calc_refs(state))
    output.extend(_fact_refs(state)[:1])
    output.extend(_card_refs(state)[:1])
    if not prefer_calc:
        output.extend(_calc_refs(state)[:1])
    seen: set[str] = set()
    unique: list[str] = []
    for ref in output:
        if ref and ref not in seen:
            seen.add(ref)
            unique.append(ref)
    return unique[:3] or ["fact:f1"]


def fact_refs(state: dict) -> list[str]:
    return _fact_refs(state)


def _mechanism_fields(state: dict, config: AgentConfig) -> list[str]:
    fields: list[str] = []
    for mechanism_id in state_mechanism_ids(state):
        spec = config.mechanisms.get(mechanism_id)
        if spec is None:
            continue
        for field_name in list(spec.critical_fields) + list(spec.optional_fields):
            if field_name not in fields:
                fields.append(field_name)
    return fields


def _scenario_branch(matrix: dict[str, Any], field_name: str, value: Any) -> list[dict[str, Any]]:
    branches = matrix.get(field_name)
    if not isinstance(branches, dict):
        return []
    if value in (None, "", "unknown"):
        key = "unknown"
    else:
        key = str(value)
    branch = branches.get(key)
    if branch is None:
        branch = branches.get("unknown")
    if not isinstance(branch, list):
        return []
    return [item for item in branch if isinstance(item, dict)]


def build_scenarios(
    state: dict,
    config: AgentConfig,
    support: list[str] | None = None,
    *,
    ranking: RankingConfig | None = None,
) -> list[dict[str, Any]]:
    """Сценарии из scenario_matrix; fallback — базовый / негативный / условный по механизмам."""
    provided = state_user_provided(state)
    refs = support_refs(state, support)
    calc_refs = support_refs(state, support, prefer_calc=True)
    max_n = max(1, int(config.max_scenarios))
    mechanisms = state_mechanism_ids(state)
    profile = state_features(state)
    ranking = ranking or load_ranking_config()
    scenarios: list[dict[str, Any]] = []

    def add(name: str, logic: str, plausibility: str, *, use_calc: bool = False) -> None:
        if len(scenarios) >= max_n:
            return
        scenarios.append(
            {
                "name": name,
                "logic": logic,
                "plausibility": plausibility,
                "support_refs": list(calc_refs if use_calc else refs),
            }
        )

    matrix = config.consequences.get("scenario_matrix") or {}
    if isinstance(matrix, dict):
        for field_name in _mechanism_fields(state, config):
            if len(scenarios) >= max_n:
                break
            for item in _scenario_branch(matrix, field_name, provided.get(field_name)):
                name = str(item.get("name") or "").strip()
                logic = str(item.get("logic") or "").strip()
                plausibility = str(item.get("plausibility") or "medium").strip() or "medium"
                if not name or not logic:
                    continue
                add(name, logic, plausibility, use_calc=bool(item.get("use_calc")))

    if not scenarios and mechanisms:
        primary = mechanisms[0]
        primary_risk = rendered_block(state, config, primary, ranking=ranking)["risk_template"]
        add(
            "базовый",
            f"Событие действует через {factor_phrase(primary, profile, config)}: {primary_risk}",
            "medium",
        )
        if len(mechanisms) > 1:
            second = mechanisms[1]
            second_risk = rendered_block(state, config, second, ranking=ranking)["risk_template"]
            add(
                "негативный",
                f"Параллельно усиливается {factor_phrase(second, profile, config)}: {second_risk}",
                "medium",
            )
        else:
            add(
                "негативный",
                f"Неблагоприятное развитие усиливает эффект через {factor_phrase(primary, profile, config)}.",
                "medium",
                use_calc=True,
            )
        missing = []
        for mechanism_id in mechanisms:
            spec = config.mechanisms.get(mechanism_id)
            if spec is None:
                continue
            for field_name in spec.critical_fields:
                value = provided.get(field_name)
                if value in (None, "", "unknown"):
                    missing.append(field_name)
        if missing:
            add(
                "условный",
                "Без ответа по полям "
                + ", ".join(sorted(set(missing)))
                + " вывод остаётся условным — масштаб станет конкретнее после уточнения.",
                "medium",
            )
        else:
            add(
                "условный",
                f"{NEUTRAL}, если признаки и критичные поля не меняются.",
                "high",
            )
    elif not scenarios:
        add(
            "базовый",
            "Без признаков механизма персональный масштаб эффекта остаётся неопределённым.",
            "low",
        )

    return scenarios[:max_n]
