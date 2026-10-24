"""Сопоставление невиданной новости с темами и механизмами без списка сценариев."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any

import yaml

from alfa.core.taxonomy import get_taxonomy
from alfa.kb.index import tokenize
from alfa.ranking.config import OPEN_WORLD_PATH, RankingConfig, RuleSpec, rule_from_raw
from alfa.ranking.models import RankEvent

CATCH_ALL_DEFAULT = "other__rko"
CLOSED_DEFAULT = frozenset({"other_econ", "irrelevant"})


@lru_cache(maxsize=1)
def load_open_world() -> dict[str, Any]:
    with OPEN_WORLD_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"open_world must be a mapping: {OPEN_WORLD_PATH}")
    return data


def is_open_topic(topic: str | None) -> bool:
    code = str(topic or "").strip()
    if not code:
        return True
    closed = set(load_open_world().get("closed_types") or CLOSED_DEFAULT)
    if code in closed:
        return True
    try:
        return code not in get_taxonomy().codes("event_type")
    except KeyError:
        return True


def _tokens(text: str) -> set[str]:
    stop = {str(item).casefold() for item in load_open_world().get("stopwords") or []}
    return {token for token in tokenize(text) if token not in stop and len(token) >= 4}


def bridge_rules() -> list[RuleSpec]:
    return [rule_from_raw(item) for item in load_open_world().get("bridge_rules") or []]


def _type_lexicon(code: str) -> tuple[set[str], set[str]]:
    taxonomy = get_taxonomy()
    meta = taxonomy.describe("event_type", code)
    tax_text = " ".join(str(part) for part in (code, meta.get("title") or "", meta.get("description") or "") if part)
    cues = load_open_world().get("cues") or {}
    cue_text = " ".join(str(item) for item in cues.get(code) or [])
    return _tokens(tax_text), _tokens(cue_text)


def type_scores(text: str) -> dict[str, float]:
    tokens = _tokens(text)
    lowered = (text or "").casefold().replace("ё", "е")
    closed = set(load_open_world().get("closed_types") or CLOSED_DEFAULT)
    cues = load_open_world().get("cues") or {}
    scores: dict[str, float] = {}
    for code in get_taxonomy().codes("event_type"):
        if code in closed:
            continue
        tax_tokens, _cue_tokens = _type_lexicon(code)
        cue_hits = sum(1 for cue in cues.get(code) or [] if str(cue).casefold().replace("ё", "е") in lowered)
        tax_hits = len(tokens & tax_tokens)
        score = cue_hits + 0.35 * tax_hits
        if score > 0:
            scores[code] = score
    return scores


def infer_event_types(event: RankEvent | None, *, text: str | None = None) -> list[str]:
    blob = text if text is not None else " ".join(
        part for part in ((event.title if event else ""), (event.text if event else "")) if part
    )
    scores = type_scores(blob)
    if not scores:
        return []
    cfg = load_open_world()
    prior_min = float(cfg.get("prior_min") or 1.0)
    relative = float(cfg.get("relative_keep") or 0.6)
    best = max(scores.values())
    ranked = [
        code
        for code, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score >= prior_min and score >= relative * best
    ]
    return ranked


def guess_event_type(text: str) -> str:
    ranked = infer_event_types(None, text=text)
    if not ranked:
        return "other_econ"
    promote = set(load_open_world().get("promote_types") or [])
    if ranked[0] in promote:
        return ranked[0]
    return "other_econ"


def catch_all_id() -> str:
    return str(load_open_world().get("catch_all_rule") or CATCH_ALL_DEFAULT)


def _rule_slot(rule: RuleSpec) -> tuple[str, tuple[str, ...]]:
    """Слот срабатывания: механизм + requires. Разные rule_id с одним слотом — один сигнал."""
    return (rule.mechanism_id, tuple(sorted(rule.requires)))


def applicable_rules(event: RankEvent, config: RankingConfig) -> tuple[list[RuleSpec], bool, list[str]]:
    """Правила для события. open_world=True — тип не из закрытой карты, правила по тексту.

    Sensitivity (с дисконтом) имеет приоритет; bridge только заполняет пустые слоты
    (mechanism_id + requires), чтобы не дублировать S_sens.
    """
    typed = [rule for rule in config.rules if event.event_type in rule.event_types and rule.rule_id != catch_all_id()]
    if typed:
        return typed, False, [event.event_type]
    inferred = infer_event_types(event)
    discount = float(load_open_world().get("weight_discount") or 0.85)
    selected: list[RuleSpec] = []
    occupied: set[tuple[str, tuple[str, ...]]] = set()
    wanted = set(inferred)
    for rule in config.rules:
        if rule.rule_id == catch_all_id():
            continue
        if rule.event_types & wanted:
            selected.append(replace(rule, weight=rule.weight * discount))
            occupied.add(_rule_slot(rule))
    for rule in bridge_rules():
        if _rule_slot(rule) in occupied:
            continue
        if rule.event_types & wanted:
            selected.append(rule)
            occupied.add(_rule_slot(rule))
    if selected:
        return selected, True, inferred
    fallback = [rule for rule in config.rules if event.event_type in rule.event_types]
    return fallback, False, inferred
