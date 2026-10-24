"""Необязательный rerank. По умолчанию выключен и не ходит в сеть.

Rerank только переставляет уже отобранные пары и двигает score в пределах ±δ.
Новые пары и новые mechanism_id он не добавляет. Порог показа не пересматривается.
"""

from __future__ import annotations

import json
from typing import Any

from alfa.ranking.config import RankingConfig
from alfa.ranking.models import RankEvent, ScoredPair


def public_features(features: dict[str, Any]) -> dict[str, Any]:
    """Агрегаты без идентификаторов клиента, группы и контрагентов."""
    allowed = (
        "industry_group",
        "themes",
        "size_class",
        "region",
        "fx_known",
        "fts_countries",
        "has_credit",
        "has_leasing",
        "has_deposit",
        "has_payroll",
        "has_rko",
        "has_guarantees",
        "has_factoring",
        "has_group",
        "reliability",
        "market_rank",
        "market_peers",
        "market_okved2",
    )
    return {key: features[key] for key in allowed if key in features}


def rerank_payload(pair: ScoredPair, event: RankEvent, features: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(part for part in (event.title, event.text) if part)[:400]
    return {
        "event_type": event.event_type,
        "event_text": text,
        "features": public_features(features),
        "fired_rules": [item.rule_id for item in pair.matched_rules],
        "mechanisms": list(pair.mechanism_ids),
    }


def _stub_label(pair: ScoredPair) -> int:
    if pair.score >= 0.3:
        return 2
    if pair.above_threshold:
        return 1
    return 0


def apply_rerank(
    pairs: list[ScoredPair],
    config: RankingConfig,
    *,
    features_by_client: dict[str, dict] | None = None,
    events: dict[str, RankEvent] | None = None,
    llm: Any | None = None,
) -> list[ScoredPair]:
    block = config.scoring["rerank"]
    if not block.get("enabled"):
        return pairs
    delta = float(block["delta"])
    top_m = int(block["top_m"])
    head, tail = pairs[:top_m], pairs[top_m:]
    backend = str(block.get("backend") or "stub")
    adjusted: list[ScoredPair] = []
    for pair in head:
        label = _stub_label(pair)
        mechanism = pair.mechanism_ids[0] if pair.mechanism_ids else ""
        if backend == "llm" and llm is not None and events is not None:
            payload = rerank_payload(pair, events[pair.event_id], (features_by_client or {}).get(pair.client_id, {}))
            raw = llm.complete_rerank(payload)
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict):
                label = int(raw.get("relevance_label", label))
                suggested = str(raw.get("mechanism_id") or "")
                if suggested in pair.mechanism_ids:
                    mechanism = suggested
                label = min(2, max(0, label))
        shift = (label - 1) * delta * 0.5
        updated = pair.model_copy(deep=True)
        updated.score = float(min(pair.score + delta, max(pair.score - delta, pair.score + shift)))
        if pair.above_threshold:
            # Показ уже решён порогом признаков: rerank не прячет пару.
            floor = pair.score  # сохраняем не ниже исходного, если сдвиг отрицательный и пара была показана
            if shift < 0:
                updated.score = pair.score
            else:
                updated.score = float(min(pair.score + delta, pair.score + shift))
            _ = floor
        if mechanism and mechanism not in updated.mechanism_ids:
            updated.score = pair.score
        adjusted.append(updated)
    adjusted.sort(key=lambda item: (-item.score, item.event_id, item.client_id))
    return adjusted + tail
