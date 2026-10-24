"""Контекст-пакет и черновик реестра утверждений."""

from __future__ import annotations

import os
import re
from typing import Any

from alfa.agent.config import BANK_VIEW_TYPES, AgentConfig
from alfa.agent.consequences import (
    apply_company_consequences,
    consequences_pack_section,
)
from alfa.agent.impact import assess_impact, missing_critical
from alfa.agent.mechanism_text import (
    calc_mechanism,
    condition_phrase,
    effect_clause,
    factor_phrase,
    feature_ref,
    field_short_label,
    headline_for,
    mechanisms_with_tag,
    prioritize,
)
from alfa.agent.security import redact_text, wrap_data
from alfa.agent.state_view import features as state_features
from alfa.agent.state_view import mechanism_ids as state_mechanism_ids
from alfa.agent.state_view import user_provided as state_user_provided
from alfa.core import textcheck

_CLOSED = {"revenue", "region", "fts_countries", "workers_class", "known_counterparty_volume_share", "fx_confidence"}
_PARTY = re.compile(r"\bпоставщик\w*\b|\bпокупател(?:ь|я|и|ем)\b", re.IGNORECASE)
TAIL_MARK = " [хвост удалён]"


def _pick_alfa_card(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cards:
        return None
    preferred = [card for card in cards if str(card.get("claim_type") or "") in BANK_VIEW_TYPES]
    return (preferred or cards)[0]


def _headline_for(state: dict, config: AgentConfig) -> str:
    mechanisms = state_mechanism_ids(state)
    unlinked = str((config.presentation or {}).get("headline_unlinked") or "Связь с профилем компании не подтверждена")
    if not mechanisms:
        return unlinked
    return headline_for(str(mechanisms[0]), state_user_provided(state), config)


def _impact_rationale_text(impact: dict[str, Any], state: dict, config: AgentConfig) -> str:
    parts = [str(impact.get("definition") or "").strip()]
    reasons = state.get("reasons") or {}
    mechanisms = list(state.get("mechanism_ids") or [])
    features = state_features(state)
    factors: list[str] = []
    for mechanism_id in mechanisms[:3]:
        clause = effect_clause(mechanism_id, reasons, config)
        factors.append(f"{factor_phrase(mechanism_id, features, config)} — {clause}")
    cards = ((state.get("evidence") or {}).get("cards") or [])
    if cards:
        factors.append("есть карточка Альфа-Банка по теме")
    if state.get("calc_results"):
        factors.append("есть расчёт чувствительности")
    if factors:
        parts.append("Ключевые факторы: " + "; ".join(factors) + ".")
    gaps = list(impact.get("missing_critical") or [])
    if gaps:
        labels = [field_short_label(name, config) for name in gaps[:3]]
        parts.append("Не хватает данных по " + ", ".join(labels) + ", поэтому масштаб условный.")
    elif mechanisms:
        parts.append("Критичные поля для прямого вывода заполнены или не требуются.")
    return " ".join(part for part in parts if part)


def _model_why_claims(
    state: dict, config: AgentConfig, features: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    mechanisms = list(state.get("mechanism_ids") or [])
    reasons = state.get("reasons") or {}
    if not mechanisms:
        return [], []
    claims: list[dict[str, Any]] = []
    why_ids: list[str] = []
    primary = mechanisms[0]
    partners = mechanisms[:2]
    factors = " и ".join(factor_phrase(item, features, config) for item in partners)
    effects = " и ".join(effect_clause(item, reasons, config) for item in partners)
    condition = condition_phrase(state, config, partners)
    lead = f"Из-за {factors}"
    if condition:
        lead += f" {condition}"
    lead += f" {effects}."
    lead = lead[0].upper() + lead[1:] if lead else lead
    claim_id = f"c_model_{primary}"
    claims.append(
        {
            "id": claim_id,
            "layer": "model",
            "text": lead,
            "refs": [],
            "feature_refs": [feature_ref(primary, features, config)],
            "mechanism_id": primary,
        }
    )
    why_ids.append(claim_id)
    if len(mechanisms) >= 2:
        second = mechanisms[1]
        second_text = (
            f"Отдельно из-за {factor_phrase(second, features, config)} "
            f"{effect_clause(second, reasons, config)}."
        )
        second_text = second_text[0].upper() + second_text[1:]
        second_id = f"c_model_{second}"
        claims.append(
            {
                "id": second_id,
                "layer": "model",
                "text": second_text,
                "refs": [],
                "feature_refs": [feature_ref(second, features, config)],
                "mechanism_id": second,
            }
        )
        why_ids.append(second_id)
    covered = {primary} | ({mechanisms[1]} if len(mechanisms) >= 2 else set())
    for mechanism_id in mechanisms:
        if mechanism_id in covered:
            continue
        claims.append(
            {
                "id": f"c_model_{mechanism_id}",
                "layer": "model",
                "text": (
                    f"Также через {factor_phrase(mechanism_id, features, config)} "
                    f"{effect_clause(mechanism_id, reasons, config)}."
                ),
                "refs": [],
                "feature_refs": [feature_ref(mechanism_id, features, config)],
                "mechanism_id": mechanism_id,
            }
        )
    return claims, why_ids


def _closed_ok(config: AgentConfig) -> bool:
    policy = config.data_policy
    if str(policy.get("channel") or "local") != "openrouter":
        return True
    env_name = str(policy.get("organizer_key_env") or "ALFA_ORGANIZER_KEY")
    return bool(os.environ.get(env_name))


def _public_features(features: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    if _closed_ok(config):
        return dict(features)
    output = {}
    for key, value in features.items():
        if key in _CLOSED:
            continue
        if isinstance(value, (bool, str)) or value is None:
            output[key] = value
    return output


def clip_text(text: str, limit: int) -> str:
    body = text or ""
    if len(body) <= limit:
        return body
    keep = max(0, limit - len(TAIL_MARK))
    return body[:keep] + TAIL_MARK


def _fact_ref(fact: dict[str, Any]) -> str:
    return f"fact:{fact.get('id') or 'f1'}"


def _primary_fact_refs(facts: list[dict[str, Any]]) -> list[str]:
    refs = [_fact_ref(fact) for fact in facts if fact.get("claim")]
    return refs or ["fact:f1"]


_THEME_CUES = ("ставк", "курс", "золот", "ключ", "нефть", "тариф", "налог")


def _theme_tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {cue for cue in _THEME_CUES if cue in lowered}


def _pack_factor_lines(state: dict, config: AgentConfig, features: dict[str, Any]) -> list[str]:
    reasons = state.get("reasons") or {}
    provided = state_user_provided(state)
    lines: list[str] = []
    for mechanism_id in state_mechanism_ids(state):
        feat = feature_ref(mechanism_id, features, config)
        reason = str(reasons.get(mechanism_id) or mechanism_id).strip()
        spec = config.mechanisms.get(mechanism_id)
        gaps: list[str] = []
        if spec is not None:
            for field_name in spec.critical_fields:
                value = provided.get(field_name)
                if value is None or value == "" or value == "unknown":
                    gaps.append(field_name)
        gap_text = ",".join(gaps) if gaps else "—"
        lines.append(f"- {mechanism_id}; feature={feat}; reason={reason}; missing_critical={gap_text}")
    if not lines:
        gaps = missing_critical(state, config)
        gap_text = ",".join(gaps) if gaps else "—"
        lines.append(f"- нет сработавших механизмов; missing_critical={gap_text}")
    return lines


def _pack_news_consequences(event: dict[str, Any]) -> str:
    risks = [str(item).strip() for item in (event.get("risks") or []) if str(item).strip()]
    opps = [str(item).strip() for item in (event.get("opportunities") or []) if str(item).strip()]
    risk_line = "; ".join(wrap_data(item) for item in risks[:3]) if risks else "не названы"
    opp_line = "; ".join(wrap_data(item) for item in opps[:3]) if opps else "не названы"
    return (
        "Уже известные риски/возможности из новости (evidence, не готовый ответ клиенту):\n"
        f"риски: {risk_line}\n"
        f"возможности: {opp_line}\n"
        "Нужно интерпретировать для клиента через признаки/механизмы, а не копировать."
    )


def _pack_ban(event: dict[str, Any]) -> str:
    title = str(event.get("title") or "").strip()
    summary = str(event.get("summary") or "").strip()
    lines = [
        "Запрет: не копировать title/summary в model-claims без нового смысла "
        "(связка событие → признак/механизм → следствие для компании)."
    ]
    if title:
        lines.append(f"title: {wrap_data(clip_text(title, 120))}")
    if summary:
        lines.append(f"summary: {wrap_data(clip_text(summary, 160))}")
    return "\n".join(lines)


def _pack_competing_signals(evidence: dict[str, Any]) -> list[str]:
    """До 2 конкурирующих сигналов: echo disagreement и банк vs новость."""
    signals: list[str] = []
    event = evidence.get("event") or {}
    for item in (event.get("echo_disagreements") or [])[:2]:
        primary = ",".join(str(num) for num in (item.get("primary_numbers") or []))
        echo = ",".join(str(num) for num in (item.get("echo_numbers") or []))
        publisher = item.get("echo_publisher") or "echo"
        signals.append(
            f"- echo disagreement: fact:{item.get('primary_id')} nums=[{primary}] "
            f"vs {publisher} nums=[{echo}]"
        )
        if len(signals) >= 2:
            return signals
    facts = list(event.get("facts") or [])
    for card in evidence.get("cards") or []:
        card_text = str(card.get("claim") or "")
        card_nums = textcheck.numbers_in(card_text)
        card_themes = _theme_tokens(card_text)
        if not card_nums or not card_themes:
            continue
        for fact in facts:
            fact_text = str(fact.get("claim") or "")
            fact_nums = textcheck.numbers_in(fact_text)
            if not fact_nums or fact_nums == card_nums:
                continue
            if not (card_themes & _theme_tokens(fact_text)):
                continue
            signals.append(
                f"- банк vs новость: card:{card.get('card_id')} nums=[{','.join(sorted(card_nums))}] "
                f"vs fact:{fact.get('id') or 'f1'} nums=[{','.join(sorted(fact_nums))}]"
            )
            break
        if len(signals) >= 2:
            break
    return signals[:2]


def build_context_pack(state: dict, config: AgentConfig) -> str:
    evidence = state.get("evidence") or {}
    event = evidence.get("event") or {}
    budget = config.context_tokens * config.chars_per_token
    raw_features = state_features(state)
    features = _public_features(raw_features, config)
    synthetic = not _closed_ok(config)
    fact_lines = []
    for fact in event.get("facts") or []:
        claim = wrap_data(str(fact.get("claim") or ""))
        span = str(fact.get("evidence_span") or "")
        locator = []
        if fact.get("url"):
            locator.append(str(fact["url"]))
        if fact.get("char_start") is not None and fact.get("char_end") is not None:
            locator.append(f"chars {fact['char_start']}-{fact['char_end']}")
        pubs = fact.get("echo_publishers") or []
        if pubs:
            locator.append("echo=" + ",".join(pubs))
        line = f"[{_fact_ref(fact)}] {claim}"
        if span and span != claim.rstrip("."):
            line += f"\n  span: {wrap_data(span)}"
        if locator:
            line += f"\n  loc: {'; '.join(locator)}"
        fact_lines.append(line)
    profile_note = "Профиль синтетический: суммы и идентификаторы сняты." if synthetic else "Признаки компании."
    factor_body = "\n".join(_pack_factor_lines(state, config, raw_features))
    sections = [
        "Факты события:\n" + ("\n".join(fact_lines) or "нет"),
        f"{profile_note} Признаки: {features}.",
        "Факторы для интерпретации:\n" + factor_body,
        _pack_news_consequences(event),
        _pack_ban(event),
    ]
    competing = _pack_competing_signals(evidence)
    if competing:
        sections.append(
            "Конкурирующие сигналы (риски vs возможности, не усреднять):\n" + "\n".join(competing)
        )
    cards = []
    for card in evidence.get("cards") or []:
        page = card.get("page")
        bbox = card.get("bbox_rel")
        loc = f" page={page}" if page else ""
        if bbox:
            loc += f" bbox={bbox}"
        cards.append(wrap_data(str(card.get("claim") or "")) + f" [{card.get('card_id')}]{loc}")
    if cards:
        sections.append("Карточки Альфа-Банка:\n" + "\n".join(cards))
    news_bits = []
    for item in evidence.get("news") or []:
        news_bits.append(
            f"{item.get('title') or item.get('event_id')}: {wrap_data(str(item.get('summary') or item.get('text') or '')[:400])}"
            + (f" url={item.get('url')}" if item.get("url") else "")
        )
    if news_bits:
        sections.append("Смежные новости:\n" + "\n".join(news_bits))
    stats_bits = []
    for item in evidence.get("stats") or []:
        stats_bits.append(
            f"{item.get('indicator')}: value={item.get('value')} {item.get('unit') or ''} "
            f"({item.get('source')}; {item.get('note') or ''})"
        )
    if stats_bits:
        sections.append("Внешняя статистика:\n" + "\n".join(stats_bits))
    calcs = [str(item.get("text") or "") for item in state.get("calc_results") or []]
    if calcs:
        sections.append("Расчёты:\n" + "\n".join(calcs))
    views = state.get("views") or {}
    group = views.get("group")
    graph = views.get("counterparties")
    if group:
        reliability = group.get("reliability", "none") if isinstance(group, dict) else "none"
        sections.append(f"Группа, reliability={reliability}: {group}")
    if graph:
        reliability = graph.get("reliability", "none") if isinstance(graph, dict) else "none"
        sections.append(f"Платёжное окружение, reliability={reliability}: {graph}")
    sections.append(consequences_pack_section(state, config))
    share = max(1, budget // len(sections))
    body = [clip_text(section, share) for section in sections]
    text, _hits = redact_text("\n\n".join(body))
    return text


def template_draft(state: dict, config: AgentConfig) -> dict[str, Any]:
    features = state_features(state)
    evidence = state.get("evidence") or {}
    event = evidence.get("event") or {}
    facts = [item for item in (event.get("facts") or []) if str(item.get("claim") or "").strip()]
    impact = assess_impact(state, config)
    claims: list[dict[str, Any]] = []
    for index, fact in enumerate(facts, start=1):
        refs = [f"fact:{fact.get('id') or f'f{index}'}"]
        for extra in fact.get("source_ids") or []:
            if extra not in refs:
                refs.append(str(extra))
        claims.append(
            {
                "id": f"c_fact_{fact.get('id') or index}",
                "layer": "fact",
                "text": str(fact.get("claim") or "").strip(),
                "refs": refs,
                "evidence": refs[:1],
            }
        )
    if not claims:
        claims.append(
            {
                "id": "c_fact_f1",
                "layer": "fact",
                "text": "Событие получено.",
                "refs": ["fact:f1"],
                "evidence": ["fact:f1"],
            }
        )
    fact_refs = _primary_fact_refs(facts)
    cards = list(evidence.get("cards") or [])
    # Несколько карточек: bank-view + факт/прогноз по теме, без схлопывания в одну.
    used_cards: list[dict[str, Any]] = []
    bank = _pick_alfa_card(cards)
    if bank is not None:
        used_cards.append(bank)
    for card in cards:
        if card in used_cards:
            continue
        if str(card.get("claim_type") or "") not in BANK_VIEW_TYPES:
            used_cards.append(card)
            break
    for offset, card in enumerate(used_cards):
        claim_text = str(card["claim"] or "").strip()
        lowered = claim_text.lower()
        if lowered.startswith("альфа-банк ") or lowered.startswith("альфа-банка "):
            rest = claim_text.split(" ", 1)[1]
            alfa_text = f"По оценке Альфа-Банка, {rest}"
        else:
            alfa_text = f"По оценке Альфа-Банка, {claim_text}"
        claims.append(
            {
                "id": "c_alfa" if offset == 0 else f"c_alfa_{offset}",
                "layer": "alfa",
                "text": alfa_text,
                "refs": [str(card["card_id"])],
            }
        )
    model_claims, why_model_ids = _model_why_claims(state, config, features)
    claims.extend(model_claims)
    extra_sources = {}
    calc_why_ids: list[str] = []
    for item in state.get("calc_results") or []:
        mechanism_id = calc_mechanism(item.get("fn"), state, config)
        if mechanism_id is None:
            continue
        calc_id = f"c_{item['calc_id'].replace(':', '_')}"
        claims.append(
            {
                "id": calc_id,
                "layer": "model",
                "text": item["text"],
                "refs": [],
                "feature_refs": [feature_ref(mechanism_id, features, config)],
                "mechanism_id": mechanism_id,
                "calc_refs": [item["calc_id"]],
            }
        )
        calc_why_ids.append(calc_id)
        extra_sources[item["calc_id"]] = item["text"]
    hypotheses = ((state.get("views") or {}).get("explanation") or state.get("explanation") or {}).get("hypotheses") or []
    if hypotheses and not _unsafe(str(hypotheses[0].get("text") or "")):
        text = str(hypotheses[0]["text"])
        extra_sources["graph:summary"] = text
        claims.append(
            {
                "id": "c_hyp_graph",
                "layer": "hypothesis",
                "text": text,
                "refs": ["graph:summary"],
                "evidence": ["graph:summary"],
                "confidence": float(hypotheses[0].get("confidence") or 0.4),
            }
        )
    else:
        claims.append(
            {
                "id": "c_hyp",
                "layer": "hypothesis",
                "text": "Если недостающие сведения появятся, масштаб эффекта станет конкретнее. Пока это предположение, а не подтверждённый факт.",
                "refs": list(fact_refs[:1]),
                "evidence": list(fact_refs[:1]),
                "confidence": 0.35,
            }
        )
    assumptions = []
    if impact["missing_critical"]:
        assumptions.append("Критичное поле в данных банка не заполнено, поэтому формулировка условная.")
    if not cards:
        assumptions.append("По этой теме нет карточки Альфа-Банка.")
    if any(card.get("stale") for card in cards):
        assumptions.append("Есть устаревшая карточка аналитики, она помечена.")
    if any(item.get("illustrative") for item in state.get("calc_results") or []):
        assumptions.append("Количественная оценка иллюстративная и не относится к сумме по компании.")
    headline = _headline_for(state, config)
    linked = str((config.presentation or {}).get("headline_linked") or "Событие связано с профилем компании")
    unlinked = str((config.presentation or {}).get("headline_unlinked") or "Связь с профилем компании не подтверждена")
    if not state.get("mechanism_ids"):
        summary = str(((state.get("views") or {}).get("explanation") or state.get("explanation") or {}).get("summary") or "").strip()
        if summary:
            claims.append(
                {
                    "id": "c_model_profile",
                    "layer": "model",
                    "text": summary,
                    "refs": [],
                    "feature_refs": ["industry_group"],
                }
            )
            why_model_ids = ["c_model_profile"]
            headline = linked
        else:
            claims.append(
                {
                    "id": "c_model_none",
                    "layer": "model",
                    "text": "Персональный вывод ограничен: в профиле нет признаков, которые связывают событие с механизмом компании.",
                    "refs": [],
                    "feature_refs": ["industry_group"],
                }
            )
            why_model_ids = ["c_model_none"]
            headline = unlinked
    by_layer = lambda layer: [item["id"] for item in claims if item["layer"] == layer]
    why = list(why_model_ids) + calc_why_ids + by_layer("alfa")
    seen_why: set[str] = set()
    why = [claim_id for claim_id in why if not (claim_id in seen_why or seen_why.add(claim_id))]
    watch = by_layer("hypothesis")
    role = str(state.get("role") or "")
    role_cfg = config.role_presentation or {}
    by_id = {item["id"]: item for item in claims}
    if role == "cfo":
        tag = str(role_cfg.get("cfo_why_tag") or "cfo_priority")
        rank_map = {name: index for index, name in enumerate(mechanisms_with_tag(config, tag))}
        why = prioritize(
            why,
            lambda claim_id: (by_id.get(claim_id) or {}).get("mechanism_id") in rank_map,
            rank=lambda claim_id: rank_map.get(
                str((by_id.get(claim_id) or {}).get("mechanism_id") or ""),
                len(rank_map),
            ),
        )
    elif role == "ceo":
        tag = str(role_cfg.get("ceo_watch_tag") or "revenue")
        tagged = set(mechanisms_with_tag(config, tag))
        watch = prioritize(
            watch,
            lambda claim_id: (
                str((by_id.get(claim_id) or {}).get("mechanism_id") or "") in tagged
                or "выруч" in str((by_id.get(claim_id) or {}).get("text") or "").lower()
            ),
        )
    draft = {
        "claims": claims,
        "headline": headline,
        "what_happened": by_layer("fact"),
        "why_it_matters": why,
        "what_to_watch": watch,
        "impact_rationale": _impact_rationale_text(impact, state, config),
        "impact_level": impact["level"],
        "impact_definition": impact["definition"],
        "impact_score": impact["score"],
        "scenarios": [],
        "assumptions": assumptions,
        "risks": [],
        "opportunities": [],
        "bank_block_ids": by_layer("alfa"),
        "disclaimer": config.disclaimer,
        "diverges_from_bank_view": False,
        "confidence": 0.42 if impact["missing_critical"] else 0.62,
        "extra_sources": extra_sources,
    }
    return apply_company_consequences(draft, state, config)


def _unsafe(text: str) -> bool:
    return bool(_PARTY.search(text or ""))


def llm_failure_kind(error: str | None) -> str:
    text = error or ""
    markers = ("JSON", "ValidationError", "validation error", "схем", "Input should", "literal_error", "extra_forbidden")
    if any(token in text for token in markers):
        return "invalid_json"
    return "unavailable"
