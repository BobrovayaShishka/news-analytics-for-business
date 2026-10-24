"""Векторный скоринг одного события на все строки CLIENT."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from alfa.core.timeutil import SECONDS_PER_DAY
from alfa.ranking.config import RankingConfig, RuleSpec
from alfa.ranking.entities import link_entities
from alfa.ranking.open_world import applicable_rules
from alfa.ranking.matrix import RELIABILITY_NAME, RELIABILITY_RANK, FeatureMatrix, embed_texts
from alfa.ranking.explain_text import human_explanation
from alfa.ranking.models import (
    CardView,
    FiredRule,
    LinkHypothesis,
    LinkedEntity,
    RankEvent,
    SignalSource,
)
from alfa.settings import norm_place

COMPONENT_ORDER = (
    "S_sens",
    "S_entity",
    "S_group",
    "S_graph",
    "S_sem",
    "S_analytics",
    "F_fresh",
    "T_src",
    "P_dup",
    "I_event",
)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def age_days(published: datetime, as_of: datetime) -> float:
    return max(0.0, (as_utc(as_of) - as_utc(published)).total_seconds() / SECONDS_PER_DAY)


def _calibration(config: RankingConfig, key: str):
    return config.scoring["calibration"][key]


@dataclass
class ScoreArrays:
    evidence: np.ndarray
    low_confidence: np.ndarray
    score: np.ndarray
    relevance: np.ndarray
    components: dict[str, np.ndarray]
    rule_fire: np.ndarray
    rules: list[RuleSpec]
    best_rule: np.ndarray
    donor_rule: np.ndarray
    hypothesis: np.ndarray
    hyp_industry: np.ndarray
    share: np.ndarray
    confidence: np.ndarray
    reliability_code: np.ndarray
    urgent: np.ndarray
    above: np.ndarray
    referenced: list[str]
    timings: dict[str, float]
    filtered: bool
    importance: float
    analytics_refs: list[str]
    linked: list[LinkedEntity]
    event_themes: set[str]
    flags_common: list[str] = field(default_factory=list)


def event_is_active(event: RankEvent, as_of: datetime, config: RankingConfig) -> bool:
    if event.extraction_status not in {"ok", "partial"}:
        return False
    if not event.is_business_relevant or not event.is_primary:
        return False
    if as_utc(event.published_at) > as_utc(as_of):
        return False
    return age_days(event.published_at, as_of) <= config.window_days(event.event_type)


def analytics_signal(
    event_type: str,
    as_of: datetime,
    cards: list[CardView],
    config: RankingConfig,
    *,
    topics: set[str] | None = None,
) -> tuple[float, list[str]]:
    block = config.scoring["analytics"]
    half = float(block["half_life_days"])
    wanted = set(topics or []) | {event_type}
    matching = [card for card in cards if wanted & set(card.topics) and as_utc(card.published_at) <= as_utc(as_of)]
    if not matching:
        return 0.0, []
    best = max(matching, key=lambda card: as_utc(card.published_at))
    age = age_days(best.published_at, as_of)
    strength = math.exp(-math.log(2) * age / half) if half else 0.0
    if age > float(block["stale_after_days"]):
        strength *= float(block["stale_multiplier"])
    return float(min(1.0, max(0.0, strength))), [best.card_id]


def _known(matrix: FeatureMatrix, name: str) -> np.ndarray:
    column = matrix.known.get(name)
    if column is None:
        return np.zeros(len(matrix), dtype=bool)
    return column


def _walk(cond: dict, found: set[str]) -> None:
    if not cond:
        return
    if "any" in cond:
        for child in cond["any"]:
            _walk(child, found)
    elif "all" in cond:
        for child in cond["all"]:
            _walk(child, found)
    elif "feature" in cond:
        found.add(str(cond["feature"]))


def _condition(cond: dict, matrix: FeatureMatrix, signals: dict, flag_true: float) -> np.ndarray:
    n = len(matrix)
    if not cond:
        return np.zeros(n, dtype=bool)
    if "any" in cond:
        out = np.zeros(n, dtype=bool)
        for child in cond["any"]:
            out |= _condition(child, matrix, signals, flag_true)
        return out
    if "all" in cond:
        out = np.ones(n, dtype=bool)
        for child in cond["all"]:
            out &= _condition(child, matrix, signals, flag_true)
        return out
    feature = str(cond["feature"])
    if feature == "event_country_in_fts":
        return signals["country_hit"] & _known(matrix, "fts_countries")
    if feature == "event_industry_match":
        return signals["industry_hit"] & _known(matrix, "industry_group")
    if feature == "industry_share":
        return signals["share_known"] & (signals["share"] >= float(cond["gte"]))
    if feature == "industry_theme":
        wanted = list(cond.get("in") or [cond.get("eq")])
        theme_columns = [matrix.theme_index[item] for item in wanted if item in matrix.theme_index]
        if not theme_columns:
            return np.zeros(n, dtype=bool)
        hit = matrix.theme_one_hot[:, theme_columns].sum(axis=1) > 0
        return hit & _known(matrix, "industry_theme")
    if feature == "industry_group":
        wanted = list(cond.get("in") or [cond.get("eq")])
        codes = [matrix.industry_index[item] for item in wanted if item in matrix.industry_index]
        if not codes:
            return np.zeros(n, dtype=bool)
        return np.isin(matrix.industry_code, np.asarray(codes, dtype=np.int16)) & _known(matrix, "industry_group")
    if feature == "region":
        wanted = [norm_place(str(item)) or str(item) for item in (cond.get("in") or [cond.get("eq")])]
        codes = [matrix.region_index[item] for item in wanted if item in matrix.region_index]
        if not codes:
            return np.zeros(n, dtype=bool)
        return np.isin(matrix.region_code, np.asarray(codes, dtype=np.int16)) & _known(matrix, "region")
    if feature == "size_class":
        if "gte" in cond:
            if cond["gte"] not in matrix.size_index:
                return np.zeros(n, dtype=bool)
            need = matrix.size_index[cond["gte"]]
            return _known(matrix, "size_class") & (matrix.size_code >= need)
        wanted = list(cond.get("in") or [cond.get("eq")])
        codes = [matrix.size_index[item] for item in wanted if item in matrix.size_index]
        if not codes:
            return np.zeros(n, dtype=bool)
        return np.isin(matrix.size_code, np.asarray(codes, dtype=np.int16)) & _known(matrix, "size_class")
    if feature == "fx_exposure" and cond.get("eq") == "known":
        return _known(matrix, "fx_exposure")
    if feature.startswith("group_any."):
        key = feature.split(".", 1)[1]
        column = matrix.group_any.get(key)
        mask = matrix.group_any_known.get(key)
        if column is None or mask is None:
            return np.zeros(n, dtype=bool)
        if cond.get("eq") is True:
            return mask & column
        if cond.get("eq") is False:
            return mask & ~column
        return mask
    mask = _known(matrix, feature)
    column = matrix.values.get(feature)
    if column is None:
        return np.zeros(n, dtype=bool)
    if cond.get("eq") is True:
        return mask & np.isfinite(column) & (column >= flag_true)
    if cond.get("eq") is False:
        return mask & np.isfinite(column) & (column < flag_true)
    if cond.get("eq") == "known":
        return mask
    if "gte" in cond:
        return mask & np.isfinite(column) & (column >= float(cond["gte"]))
    if "lte" in cond:
        return mask & np.isfinite(column) & (column <= float(cond["lte"]))
    return np.zeros(n, dtype=bool)


def _rule_mask(rule: RuleSpec, matrix: FeatureMatrix, signals: dict, event_themes: set[str], flag_true: float) -> np.ndarray:
    n = len(matrix)
    if rule.entity_themes and rule.entity_themes.isdisjoint(event_themes):
        return np.zeros(n, dtype=bool)
    if rule.weight <= 0:
        return np.zeros(n, dtype=bool)
    mask = _condition(rule.when, matrix, signals, flag_true)
    for name in rule.requires:
        mask &= _known(matrix, name)
    if rule.min_reliability:
        need = RELIABILITY_RANK[rule.min_reliability]
        mask &= _known(matrix, "reliability") & (matrix.reliability_rank >= need)
    return mask


def _group_signal(matrix: FeatureMatrix, sensitivity: np.ndarray, best_rule: np.ndarray, gamma: float, size_cap: float):
    n = len(matrix)
    s_group = np.zeros(n, dtype=np.float64)
    donor = np.full(n, -1, dtype=np.int16)
    valid = matrix.group_code >= 0
    if n == 0 or not np.any(valid):
        return s_group, donor
    frame = pd.DataFrame({"g": matrix.group_code, "s": sensitivity, "i": np.arange(n)})
    sub = frame.loc[valid].copy()
    sub.index = sub["i"].to_numpy()
    gmax = sub.groupby("g")["s"].transform("max")
    at = np.isclose(sub["s"].to_numpy(), gmax.to_numpy())
    sub = sub.assign(at=at, gmax=gmax.to_numpy())
    counts = sub.groupby("g")["at"].transform("sum").to_numpy()
    unique_holder = at & (counts == 1)
    s_rest = np.where(unique_holder, -1.0, sub["s"].to_numpy())
    second = pd.Series(s_rest, index=sub.index).groupby(sub["g"].to_numpy()).transform("max").to_numpy()
    s_other = np.where(unique_holder, np.maximum(second, 0.0), gmax.to_numpy())
    s_other = np.where(s_other < 0, 0.0, s_other)
    s_other_full = np.zeros(n, dtype=np.float64)
    s_other_full[sub["i"].to_numpy()] = s_other
    scale = np.minimum(1.0, float(size_cap) / np.maximum(matrix.group_member_count, 1.0))
    s_group = np.where(valid, float(gamma) * scale * s_other_full, 0.0)
    s_group = np.where(s_other_full > 0, s_group, 0.0)

    winners = sub.groupby("g")["s"].idxmax()
    rest = sub.drop(index=list(winners.values), errors="ignore")
    seconds = rest.groupby("g")["s"].idxmax() if not rest.empty else pd.Series(dtype="int64")
    width = int(matrix.group_code.max()) + 1
    winner_of = np.full(width, -1, dtype=np.int32)
    second_of = np.full(width, -1, dtype=np.int32)
    for group_code, client_index in winners.items():
        winner_of[int(group_code)] = int(client_index)
    for group_code, client_index in seconds.items():
        second_of[int(group_code)] = int(client_index)
    unique_full = np.zeros(n, dtype=bool)
    unique_full[sub["i"].to_numpy()] = unique_holder
    clipped = np.clip(matrix.group_code, 0, width - 1)
    winner_client = np.where(valid, winner_of[clipped], -1)
    second_client = np.where(valid, second_of[clipped], -1)
    donor_client = np.where(unique_full, second_client, winner_client)
    ok = donor_client >= 0
    donor[ok] = best_rule[donor_client[ok]]
    donor = np.where(s_group > 0, donor, -1).astype(np.int16)
    return s_group, donor


def _importance(event: RankEvent, config: RankingConfig) -> float:
    block = config.scoring["importance"]
    base = float(block["base"].get(event.event_type, block["unknown"]))
    official = float(block["official_bonus"]) if event.trust_tier == "official" else 0.0
    magnitude = float(block["magnitude_weight"]) * min(1.0, max(0.0, event.magnitude))
    extra_n = max(0, int(event.n_sources) - 1)
    extra = min(float(block["extra_source_cap"]), float(block["extra_source_bonus"]) * extra_n)
    return float(min(1.0, max(0.0, base + official + magnitude + extra)))


def _empty(n: int, config: RankingConfig, filtered: bool) -> ScoreArrays:
    zeros = np.zeros(n, dtype=np.float64)
    return ScoreArrays(
        evidence=np.zeros(n, dtype=bool),
        low_confidence=np.zeros(n, dtype=bool),
        score=zeros,
        relevance=zeros,
        components={key: zeros.copy() for key in COMPONENT_ORDER},
        rule_fire=np.zeros((n, 0), dtype=bool),
        rules=[],
        best_rule=np.full(n, -1, dtype=np.int16),
        donor_rule=np.full(n, -1, dtype=np.int16),
        hypothesis=np.zeros(n, dtype=bool),
        hyp_industry=np.full(n, -1, dtype=np.int16),
        share=zeros,
        confidence=np.ones(n, dtype=np.float64),
        reliability_code=np.zeros(n, dtype=np.int16),
        urgent=np.zeros(n, dtype=bool),
        above=np.zeros(n, dtype=bool),
        referenced=[],
        timings={key: 0.0 for key in ("rules", "group", "graph", "embeddings", "aggregation")},
        filtered=filtered,
        importance=0.0,
        analytics_refs=[],
        linked=[],
        event_themes=set(),
        flags_common=[],
    )


def score_arrays(
    matrix: FeatureMatrix,
    event: RankEvent,
    as_of: datetime,
    config: RankingConfig,
    cards: list[CardView],
    *,
    blocked: set[str] | None = None,
    enabled: set[str] | None = None,
    cache_dir=None,
) -> ScoreArrays:
    """Один проход по матрице. Цикл идёт по правилам, не по клиентам."""
    n = len(matrix)
    blocks = enabled or {"sens", "entity", "group", "graph", "sem", "analytics"}
    if n == 0 or not event_is_active(event, as_of, config):
        return _empty(n, config, filtered=True)
    import time

    timings = {key: 0.0 for key in ("rules", "group", "graph", "embeddings", "aggregation")}
    linked = link_entities(event.entities, config)
    event_themes = {item.value for item in linked if item.kind == "theme"}
    event_countries = [item.value for item in linked if item.kind == "country"]
    event_groups = [item.value for item in linked if item.kind == "industry"]
    event_regions = [item.value for item in linked if item.kind == "region"]
    event_currencies = {item.value for item in linked if item.kind == "currency"}

    started = time.perf_counter()
    flag_true = float(_calibration(config, "flag_true"))
    weight_cap = float(_calibration(config, "rule_weight_cap"))
    signal_epsilon = float(_calibration(config, "signal_epsilon"))
    if event_countries:
        country_columns = [matrix.country_index[code] for code in event_countries if code in matrix.country_index]
        country_hit = matrix.countries[:, country_columns].any(axis=1) if country_columns else np.zeros(n, dtype=bool)
    else:
        country_hit = np.zeros(n, dtype=bool)
    theme_columns = [matrix.theme_index[item.value] for item in linked if item.kind == "theme" and item.value in matrix.theme_index]
    group_codes = [matrix.industry_index[code] for code in event_groups if code in matrix.industry_index]
    industry_hit = np.zeros(n, dtype=bool)
    if theme_columns:
        industry_hit |= matrix.theme_one_hot[:, theme_columns].sum(axis=1) > 0
    if group_codes:
        industry_hit |= np.isin(matrix.industry_code, np.asarray(group_codes, dtype=np.int16))
    industry_hit &= _known(matrix, "industry_theme") | _known(matrix, "industry_group")
    if event_regions:
        region_codes = [matrix.region_index[code] for code in event_regions if code in matrix.region_index]
        region_hit = (
            np.isin(matrix.region_code, np.asarray(region_codes, dtype=np.int16)) & _known(matrix, "region")
            if region_codes
            else np.zeros(n, dtype=bool)
        )
    else:
        region_hit = np.zeros(n, dtype=bool)
    if event_currencies:
        currency_hit = np.fromiter(
            (bool(set(codes) & event_currencies) for codes in matrix.currencies),
            dtype=bool,
            count=n,
        )
    else:
        currency_hit = np.zeros(n, dtype=bool)
    if group_codes:
        industry_mix = np.maximum(matrix.mix_in[:, group_codes], matrix.mix_out[:, group_codes])
        share = industry_mix.max(axis=1).astype(np.float64)
        hyp_industry = np.asarray(group_codes, dtype=np.int16)[industry_mix.argmax(axis=1)]
    else:
        share = np.zeros(n, dtype=np.float64)
        hyp_industry = np.full(n, -1, dtype=np.int16)
    share_known = _known(matrix, "has_payment_data") & np.isfinite(matrix.values["has_payment_data"]) & (
        matrix.values["has_payment_data"] >= flag_true
    )
    share = np.where(share_known, share, 0.0)
    signals = {
        "country_hit": country_hit,
        "industry_hit": industry_hit,
        "share": share,
        "share_known": share_known,
    }
    timings["graph"] += time.perf_counter() - started

    started = time.perf_counter()
    applicable, open_world, inferred_types = applicable_rules(event, config)
    referenced: set[str] = set()
    for rule in applicable:
        _walk(rule.when, referenced)
        referenced.update(rule.requires)
    fire = np.zeros((n, len(applicable)), dtype=bool)
    uncovered = np.ones(n, dtype=np.float64)
    for index, rule in enumerate(applicable):
        mask = _rule_mask(rule, matrix, signals, event_themes, flag_true)
        fire[:, index] = mask
        if "sens" in blocks and rule.weight > 0:
            uncovered = np.where(mask, uncovered * (1.0 - min(rule.weight, weight_cap)), uncovered)
    sensitivity = 1.0 - uncovered if "sens" in blocks else np.zeros(n, dtype=np.float64)
    best_w = np.full(n, -1.0, dtype=np.float64)
    best_rule = np.full(n, -1, dtype=np.int16)
    order = sorted(range(len(applicable)), key=lambda i: (-applicable[i].weight, applicable[i].rule_id))
    for index in order:
        weight = applicable[index].weight
        selected = fire[:, index] & (weight > best_w)
        best_w = np.where(selected, weight, best_w)
        best_rule = np.where(selected, index, best_rule).astype(np.int16)
    timings["rules"] = time.perf_counter() - started

    started = time.perf_counter()
    group_cfg = config.scoring["group"]
    group_score, donor = _group_signal(matrix, sensitivity, best_rule, float(group_cfg["gamma"]), float(group_cfg["size_cap"]))
    if "group" not in blocks:
        group_score = np.zeros(n, dtype=np.float64)
        donor = np.full(n, -1, dtype=np.int16)
    timings["group"] = time.perf_counter() - started

    started = time.perf_counter()
    graph_score = np.where(share_known, share * matrix.reliability_coeff, 0.0) if "graph" in blocks else np.zeros(n)
    hyp_cfg = config.scoring["graph"]["hypothesis"]
    need = RELIABILITY_RANK[str(hyp_cfg["min_reliability"])]
    hypothesis = (
        share_known
        & (share >= float(hyp_cfg["share_min"]))
        & _known(matrix, "known_counterparty_volume_share")
        & np.isfinite(matrix.known_share)
        & (matrix.known_share >= float(hyp_cfg["known_share_min"]))
        & _known(matrix, "n_edges")
        & np.isfinite(matrix.n_edges)
        & (matrix.n_edges >= float(hyp_cfg["min_edges"]))
        & _known(matrix, "reliability")
        & (matrix.reliability_rank >= need)
        & (hyp_industry >= 0)
    )
    if "graph" not in blocks:
        hypothesis = np.zeros(n, dtype=bool)
    timings["graph"] += time.perf_counter() - started

    started = time.perf_counter()
    dim = int(config.scoring["embedding"]["dim"])
    event_text = " ".join(part for part in (event.title, event.text) if part) or event.event_type
    vector, _backend = embed_texts([event_text], cache_dir, dim)
    semantic = np.clip(matrix.embeddings @ vector[0], 0.0, 1.0).astype(np.float64)
    if "sem" not in blocks:
        semantic = np.zeros(n, dtype=np.float64)
    timings["embeddings"] = time.perf_counter() - started

    started = time.perf_counter()
    entity_w = config.scoring["entity"]
    entity_score = (
        country_hit.astype(np.float64) * float(entity_w["country"])
        + industry_hit.astype(np.float64) * float(entity_w["industry"])
        + region_hit.astype(np.float64) * float(entity_w["region"])
        + currency_hit.astype(np.float64) * float(entity_w["currency"])
    )
    boost = np.zeros(n, dtype=np.float64)
    for index, rule in enumerate(applicable):
        spec = rule.entity_boost or {}
        if spec.get("event_entity") == "country":
            boost = np.maximum(boost, np.where(fire[:, index] & country_hit, float(spec.get("boost", 0.0)), 0.0))
    if "entity" not in blocks:
        entity_score = np.zeros(n, dtype=np.float64)
    else:
        entity_score = np.clip(entity_score + boost, 0.0, 1.0)
    strength, analytics_refs = analytics_signal(
        event.event_type,
        as_of,
        cards,
        config,
        topics=set(inferred_types),
    )
    analytics_score = np.full(n, strength if "analytics" in blocks else 0.0, dtype=np.float64)
    weights = config.scoring["weights"]
    relevance = np.clip(
        float(weights["w_sens"]) * sensitivity
        + float(weights["w_entity"]) * entity_score
        + float(weights["w_group"]) * group_score
        + float(weights["w_graph"]) * graph_score
        + float(weights["w_sem"]) * semantic
        + float(weights["w_an"]) * analytics_score,
        0.0,
        1.0,
    )
    score_type = inferred_types[0] if open_world and inferred_types else event.event_type
    if score_type == event.event_type:
        importance = _importance(event, config)
    else:
        importance = _importance(
            RankEvent(
                event_id=event.event_id,
                event_type=score_type,
                published_at=event.published_at,
                trust_tier=event.trust_tier,
                magnitude=event.magnitude,
                n_sources=event.n_sources,
            ),
            config,
        )
    fresh = math.exp(-math.log(2) * age_days(event.published_at, as_of) / config.half_life(score_type))
    trust = config.trust(event.trust_tier)
    p_dup = np.zeros(n, dtype=np.float64)
    if blocked:
        p_dup[np.isin(matrix.client_ids, np.array(list(blocked), dtype=object))] = 1.0
    factor = float(config.scoring["score"]["a"]) + float(config.scoring["score"]["b"]) * importance
    score = relevance * factor * fresh * trust * (1.0 - p_dup)
    structural = np.zeros(n, dtype=bool)
    if "sens" in blocks:
        structural |= sensitivity > signal_epsilon
    if "entity" in blocks:
        structural |= entity_score > signal_epsilon
    if "group" in blocks:
        structural |= group_score > signal_epsilon
    if "graph" in blocks:
        structural |= graph_score > signal_epsilon
    sem_only = (~structural) & (semantic >= config.sem_only_min) if "sem" in blocks else np.zeros(n, dtype=bool)
    evidence = structural | sem_only
    if referenced:
        stack = np.vstack([_known(matrix, name).astype(np.float64) for name in sorted(referenced)])
        confidence = stack.mean(axis=0)
    else:
        confidence = np.ones(n, dtype=np.float64)
    reliability = _reliability_codes(
        fire.any(axis=1) if fire.size else np.zeros(n, dtype=bool),
        group_score,
        graph_score,
        analytics_score,
        sem_only,
        confidence,
        matrix,
        config,
    )
    thresholds = config.scoring["thresholds"]
    urgent = (relevance >= float(thresholds["tau_rel"])) & (importance >= float(thresholds["tau_imp"])) & (
        age_days(event.published_at, as_of) <= float(thresholds["tau_age_days"])
    )
    above = score >= config.tau_show(score_type)
    components = {
        "S_sens": sensitivity,
        "S_entity": entity_score,
        "S_group": group_score,
        "S_graph": graph_score,
        "S_sem": semantic,
        "S_analytics": analytics_score,
        "F_fresh": np.full(n, fresh),
        "T_src": np.full(n, trust),
        "P_dup": p_dup,
        "I_event": np.full(n, importance),
    }
    timings["aggregation"] = time.perf_counter() - started
    return ScoreArrays(
        evidence=evidence,
        low_confidence=sem_only,
        score=score,
        relevance=relevance,
        components=components,
        rule_fire=fire,
        rules=applicable,
        best_rule=best_rule,
        donor_rule=donor,
        hypothesis=hypothesis,
        hyp_industry=hyp_industry,
        share=share,
        confidence=confidence,
        reliability_code=reliability,
        urgent=urgent,
        above=above,
        referenced=sorted(referenced),
        timings=timings,
        filtered=False,
        importance=importance,
        analytics_refs=analytics_refs,
        linked=linked,
        event_themes=event_themes,
        flags_common=["open_world"] if open_world else [],
    )


def _reliability_codes(
    fired,
    group_score,
    graph_score,
    analytics_score,
    sem_only,
    confidence,
    matrix: FeatureMatrix,
    config: RankingConfig,
) -> np.ndarray:
    n = len(matrix)
    unused = int(_calibration(config, "reliability_unused"))
    high = float(_calibration(config, "confidence_high"))
    medium = float(_calibration(config, "confidence_medium"))
    epsilon = float(_calibration(config, "signal_epsilon"))
    used = np.full(n, unused, dtype=np.int16)
    profile = np.where(confidence >= high, 3, np.where(confidence >= medium, 2, 1)).astype(np.int16)
    used = np.where(fired, np.minimum(used, profile), used)
    used = np.where(group_score > epsilon, np.minimum(used, 2), used)
    used = np.where(graph_score > epsilon, np.minimum(used, matrix.reliability_rank), used)
    used = np.where(analytics_score > epsilon, np.minimum(used, 3), used)
    used = np.where(sem_only, np.minimum(used, 1), used)
    return np.where(used == unused, 0, used).astype(np.int16)


def _contributions(rules: list[RuleSpec], fired: np.ndarray, weight_cap: float) -> list[FiredRule]:
    indexes = [i for i, flag in enumerate(fired.tolist()) if flag]
    if not indexes:
        return []
    weights = [rules[i].weight for i in indexes]
    prod_all = 1.0
    for weight in weights:
        prod_all *= 1.0 - min(weight, weight_cap)
    total = 1.0 - prod_all
    output: list[FiredRule] = []
    for index, weight in zip(indexes, weights, strict=True):
        remain = prod_all / (1.0 - min(weight, weight_cap))
        without = 1.0 - remain
        output.append(
            FiredRule(
                rule_id=rules[index].rule_id,
                mechanism_id=rules[index].mechanism_id,
                contribution=float(total - without),
            )
        )
    output.sort(key=lambda item: (-item.contribution, item.rule_id))
    return output


def _hypothesis(matrix: FeatureMatrix, arrays: ScoreArrays, config: RankingConfig, index: int) -> LinkHypothesis | None:
    if not bool(arrays.hypothesis[index]):
        return None
    industry_index = int(arrays.hyp_industry[index])
    industry = matrix.industry_labels[industry_index] if industry_index >= 0 else ""
    share = float(arrays.share[index])
    coverage = float(matrix.known_share[index]) if np.isfinite(matrix.known_share[index]) else 0.0
    edges = int(matrix.n_edges[index]) if np.isfinite(matrix.n_edges[index]) else 0
    hyp_cfg = config.scoring["graph"]["hypothesis"]
    confidence = min(
        1.0,
        max(
            0.0,
            float(hyp_cfg["confidence_share"]) * share
            + float(hyp_cfg["confidence_coverage"]) * coverage
            + float(hyp_cfg["confidence_edges"]) * min(1.0, edges / float(hyp_cfg["edge_scale"])),
        ),
    )
    label = config.industry_label(industry)
    text = (
        f"Есть платёжная связь с отраслью «{label}»: "
        f"доля известного объёма {share:.0%}, покрытие {coverage:.0%}, рёбер {edges}."
    )
    return LinkHypothesis(
        type="payment_link_exposure",
        text=text,
        evidence={"share": share, "coverage": coverage, "n_edges": edges, "industry": industry},
        confidence=confidence,
    )


def explain_score(matrix: FeatureMatrix, arrays: ScoreArrays, config: RankingConfig, event: RankEvent, index: int) -> dict:
    weight_cap = float(_calibration(config, "rule_weight_cap"))
    epsilon = float(_calibration(config, "signal_epsilon"))
    fired = _contributions(
        arrays.rules,
        arrays.rule_fire[index] if arrays.rule_fire.shape[1] else np.zeros(0, dtype=bool),
        weight_cap,
    )
    components = {key: float(arrays.components[key][index]) for key in COMPONENT_ORDER}
    missing = [name for name in arrays.referenced if name in matrix.known and not bool(matrix.known[name][index])]
    group_signal = None
    donor = int(arrays.donor_rule[index]) if len(arrays.donor_rule) else -1
    if components["S_group"] > epsilon and 0 <= donor < len(arrays.rules):
        group_signal = SignalSource(via="group_member", mechanism_id=arrays.rules[donor].mechanism_id)
    graph_signal = None
    if components["S_graph"] > epsilon:
        hypothesis_cfg = config.scoring["graph"]["hypothesis"]
        mechanism = str(hypothesis_cfg["default_mechanism"])
        allowed = set(hypothesis_cfg["mechanisms"])
        for rule in fired:
            if rule.mechanism_id in allowed:
                mechanism = rule.mechanism_id
                break
        graph_signal = SignalSource(via="payment_neighborhood", mechanism_id=mechanism)
    hypothesis = _hypothesis(matrix, arrays, config, index)
    flags: list[str] = list(arrays.flags_common)
    if bool(arrays.low_confidence[index]):
        flags.append("low_confidence")
    fuzzy = any(item.fuzzy for item in arrays.linked)
    if fuzzy and (components["S_entity"] > 0 or components["S_sens"] > 0):
        flags.append("entity_fuzzy")
    if group_signal is not None:
        flags.append("group_signal")
    if graph_signal is not None:
        flags.append("graph_signal")
    if hypothesis is not None:
        flags.append("hypothesis")
    if arrays.filtered:
        flags.append("event_filtered")
    mechanism_ids: list[str] = []
    for rule in fired:
        if rule.mechanism_id not in mechanism_ids:
            mechanism_ids.append(rule.mechanism_id)
    priority = "urgent" if bool(arrays.urgent[index]) else "regular"
    hypotheses = [hypothesis] if hypothesis else []
    row = matrix.rows[index]
    summary, signals = human_explanation(
        row,
        event,
        fired,
        config,
        group_signal=group_signal,
        graph_signal=graph_signal,
        hypotheses=hypotheses,
    )
    return {
        "score": float(arrays.score[index]),
        "relevance": float(arrays.relevance[index]),
        "importance": float(arrays.importance),
        "components": components,
        "matched_rules": fired,
        "reliability": RELIABILITY_NAME[int(arrays.reliability_code[index])],
        "priority": priority,
        "flags": sorted(flags),
        "mechanism_ids": mechanism_ids,
        "hypotheses": hypotheses,
        "missing_features": missing,
        "score_confidence": float(arrays.confidence[index]),
        "group_signal": group_signal,
        "graph_signal": graph_signal,
        "analytics_refs": list(arrays.analytics_refs),
        "eligible": bool(arrays.evidence[index]) and not arrays.filtered,
        "above_threshold": bool(arrays.above[index]) and not arrays.filtered,
        "summary": summary,
        "signals": signals,
    }
