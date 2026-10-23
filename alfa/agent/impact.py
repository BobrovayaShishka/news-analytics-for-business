"""Оценка влияния по рубрике. Уровень считает код, не модель."""

from __future__ import annotations

from alfa.agent.config import AgentConfig


def _user_map(state: dict) -> dict[str, str]:
    return {str(item["field"]): str(item["value"]) for item in state.get("user_provided") or []}


def missing_critical(state: dict, config: AgentConfig) -> list[str]:
    provided = _user_map(state)
    missing: list[str] = []
    for mechanism_id in state.get("mechanism_ids") or []:
        spec = config.mechanisms.get(mechanism_id)
        if spec is None:
            continue
        for field_name in spec.critical_fields:
            value = provided.get(field_name)
            if value is None or value == "" or value == "unknown":
                missing.append(field_name)
    return missing


def _downgrade(level: str, steps: int) -> str:
    order = ["low", "medium", "high"]
    index = order.index(level)
    return order[max(0, index - steps)]


def _band(score: float, rubric: dict) -> str:
    ranked = sorted(rubric["levels"].items(), key=lambda item: float(item[1]["min_score"]), reverse=True)
    for name, payload in ranked:
        if score >= float(payload["min_score"]):
            return name
    return "low"


def assess_impact(state: dict, config: AgentConfig) -> dict:
    rubric = config.rubric
    weights = rubric["weights"]
    mechanisms = list(state.get("mechanism_ids") or [])
    explanation = state.get("explanation") or {}
    direct = False
    if mechanisms:
        direct = any(config.mechanisms[item].directness == "direct" for item in mechanisms if item in config.mechanisms)
        if not explanation.get("rules") and (explanation.get("group_signal") or explanation.get("graph_signal")):
            direct = False
    base = float(weights["direct"] if direct else weights["indirect"])
    extra = min(
        float(weights["extra_mechanism_cap"]),
        float(weights["extra_mechanism"]) * max(0, len(mechanisms) - 1),
    )
    importance = float(explanation.get("importance") or 0.0)
    features = ((state.get("views") or {}).get("profile") or {}).get("features") or {}
    share = features.get("known_counterparty_volume_share")
    share_part = float(weights["payment_share"]) * float(share) if isinstance(share, (int, float)) else 0.0
    reliability = str(state.get("reliability") or explanation.get("reliability") or "none")
    reliability_part = float((weights.get("reliability") or {}).get(reliability) or 0.0)
    horizon = float((weights.get("horizon") or {}).get(state.get("event_type"), (weights.get("horizon") or {}).get("default", 0.0)) or 0.0)
    score = base + extra + float(weights["importance"]) * importance + share_part + reliability_part + horizon
    if not mechanisms:
        score = min(score, 0.2)
    level = _band(score, rubric)
    steps = 0
    low_band = set(rubric["downgrade"]["reliability_low"])
    if reliability in low_band:
        steps += 1
    gaps = missing_critical(state, config)
    if gaps and rubric["downgrade"].get("missing_critical"):
        steps += 1
    level = _downgrade(level, steps)
    definition = str(rubric["levels"][level]["definition"])
    rationale = f"{definition} Уровень посчитан по рубрике."
    return {
        "level": level,
        "definition": definition,
        "rationale": rationale,
        "score": round(score, 4),
        "directness": "direct" if direct else "indirect",
        "downgrade_steps": steps,
        "missing_critical": gaps,
        "reliability": reliability,
    }
