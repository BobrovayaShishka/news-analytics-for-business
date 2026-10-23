"""Сборка JSON-контракта для интерфейса."""

from __future__ import annotations

from typing import Any

from alfa.agent.config import AgentConfig
from alfa.agent.models import (
    CalcView,
    Claim,
    FeatureUsed,
    Insight,
    ModelInference,
    PendingQuestion,
    RelevanceView,
    Scenario,
    UserFact,
    VerificationView,
)
from alfa.agent.verify import contradictions

_CLAIM_FIELDS = {
    "id",
    "layer",
    "text",
    "refs",
    "feature_refs",
    "mechanism_id",
    "calc_refs",
    "evidence",
    "confidence",
    "diverges_from_bank_view",
}


def _claim(raw: dict) -> Claim:
    payload = {key: raw[key] for key in _CLAIM_FIELDS if key in raw}
    if payload.get("confidence") is not None:
        payload["confidence"] = float(payload["confidence"])
    payload["diverges_from_bank_view"] = bool(payload.get("diverges_from_bank_view") or False)
    return Claim.model_validate(payload)


def _contradictions(verification: dict, state: dict) -> list:
    stored = verification.get("contradictions")
    if isinstance(stored, list):
        return stored
    return contradictions((state.get("evidence") or {}).get("cards") or [])


def _join(ids: list[str], claims: list[dict]) -> str:
    by_id = {item["id"]: item["text"] for item in claims}
    return " ".join(by_id[item] for item in ids if item in by_id).strip()


def assemble_insight(state: dict, config: AgentConfig) -> dict[str, Any]:
    draft = dict(state.get("draft") or {})
    claims_raw = list(draft.get("claims") or [])
    claims = [_claim(item) for item in claims_raw]
    dumped = [item.model_dump() for item in claims]
    explanation = state.get("explanation") or {}
    features = ((state.get("views") or {}).get("profile") or {}).get("features") or {}
    used = []
    seen = set()
    for claim in dumped:
        for name in claim.get("feature_refs") or []:
            if name in seen:
                continue
            seen.add(name)
            used.append(FeatureUsed(feature=name, value=features.get(name)))
    calcs = []
    for item in state.get("calc_results") or []:
        if not item.get("calc_id"):
            continue
        calcs.append(CalcView.model_validate(item))
    scenarios = [Scenario.model_validate(item) for item in draft.get("scenarios") or []]
    level = draft.get("impact_level") if draft.get("impact_level") in {"low", "medium", "high"} else "low"
    definition = str(draft.get("impact_definition") or config.rubric["levels"][level]["definition"])
    verification = state.get("verification") or {}
    stage1 = verification.get("stage1") or {}
    stage2 = verification.get("stage2") or {}
    all_issues = list(stage1.get("issues") or []) + list(stage2.get("issues") or [])
    decision = verification.get("decision")
    flags = list(dict.fromkeys(state.get("flags") or []))
    if "access_denied" in flags or "unknown_event" in flags or "invalid_json" in flags:
        status = "withheld"
    elif decision == "pass":
        status = "published"
    elif decision == "caveats":
        status = "published_with_caveats"
    else:
        status = "withheld"
    pending = state.get("pending_question")
    question = PendingQuestion.model_validate(pending) if pending else None
    users = [
        UserFact(
            field=str(item["field"]),
            value=str(item["value"]),
            answered_at=str(item.get("answered_at") or ""),
            verified_by_bank=False,
        )
        for item in state.get("user_provided") or []
    ]
    structural = ((state.get("views") or {}).get("profile") or {}).get("structurally_missing") or []
    filled = {item.field for item in users if item.value not in {"", "unknown"}}
    missing = [name for name in structural if name not in filled]
    missing.extend(
        name
        for name in ((state.get("views") or {}).get("profile") or {}).get("missing_fields") or []
        if name not in missing
    )
    components = {str(key): float(value) for key, value in (explanation.get("components") or {}).items()}
    priority = state.get("priority") if state.get("priority") in {"urgent", "regular"} else "regular"
    role = state.get("role") if state.get("role") in {"ceo", "cfo", "default"} else "default"
    insight = Insight(
        insight_id=str(state["insight_id"]),
        trace_id=str(state["trace_id"]),
        status=status,
        event_id=str(state.get("event_id") or ""),
        client_id=str(state.get("client_id") or ""),
        as_of=str(state.get("as_of") or ""),
        priority=priority,
        role=role,
        headline=str(draft.get("headline") or "Инсайт не опубликован"),
        what_happened=_join(list(draft.get("what_happened") or []), dumped),
        why_it_matters=_join(list(draft.get("why_it_matters") or []), dumped),
        what_to_watch=_join(list(draft.get("what_to_watch") or []), dumped),
        claims=claims,
        confirmed_facts=[item for item in claims if item.layer == "fact"],
        alfa_analytics=[item for item in claims if item.layer == "alfa"],
        model_inference=ModelInference(
            impact_level=level,
            impact_definition=definition,
            impact_rationale=str(draft.get("impact_rationale") or definition),
            quantification=calcs,
            risks=list(draft.get("risks") or []),
            opportunities=list(draft.get("opportunities") or []),
            scenarios=scenarios,
            diverges_from_bank_view=bool(draft.get("diverges_from_bank_view") or False),
            confidence=float(draft.get("confidence") or 0.0),
        ),
        hypotheses=[item for item in claims if item.layer == "hypothesis"],
        user_provided=users,
        assumptions=list(draft.get("assumptions") or []),
        client_features_used=used,
        relevance=RelevanceView(
            score=float(explanation.get("score") or 0.0),
            relevance=float(explanation.get("relevance") or 0.0),
            importance=float(explanation.get("importance") or 0.0),
            components=components,
            reliability=str(explanation.get("reliability") or state.get("reliability") or "none"),
            mechanism_ids=list(state.get("mechanism_ids") or []),
            rules=list(explanation.get("rules") or []),
            above_threshold=bool(explanation.get("above_threshold") or False),
        ),
        missing_data=missing,
        verification=VerificationView(
            stage1=stage1,
            stage2=stage2,
            iterations=int(state.get("iteration") or 0),
            contradictions=_contradictions(verification, state),
            stale_claims=[item["claim_id"] for item in all_issues if item["type"] == "stale" and item["claim_id"]],
            layer_violations=[item["claim_id"] for item in all_issues if item["type"] == "layer_violation" and item["claim_id"]],
        ),
        pending_question=question,
        flags=flags,
        disclaimer=config.disclaimer,
    )
    return insight.model_dump(mode="json")
