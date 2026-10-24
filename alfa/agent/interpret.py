"""Контекст-пакет и черновик реестра утверждений."""

from __future__ import annotations

import os
import re
from typing import Any

from alfa.agent.config import BANK_VIEW_TYPES, AgentConfig
from alfa.agent.impact import assess_impact
from alfa.agent.security import redact_text, wrap_data

_HEADLINES = {
    "debt_service_cost": "Событие касается стоимости вашего долга",
    "deposit_yield": "Событие касается доходности размещённых средств",
    "fx_cost": "Событие касается валютной стоимости операций",
    "import_restrictions": "Событие касается внешнего торгового контура",
    "labor_cost": "Событие касается расходов на персонал",
    "demand_shift": "Событие касается выручки в вашей отрасли",
    "input_cost": "Событие касается себестоимости и входящих ресурсов",
    "commodity_revenue": "Событие касается профильной выручки",
    "tax_burden": "Событие касается налоговой нагрузки на компанию",
    "regulatory_cost": "Событие касается стоимости деятельности в отрасли",
    "guarantee_risk": "Событие касается гарантий и расчётов",
}
_CONDITIONAL = {
    "debt_rate_type": "Тип ставки в данных банка неизвестен: если ставка плавающая, платёж может измениться, если фиксированная — нет.",
    "fx_expense_share": "Доля валютных расходов в данных банка неизвестна: если она заметна, курс сильнее отражается на затратах.",
    "fx_revenue_share": "Доля валютной выручки в данных банка неизвестна.",
    "debt_amount": "Сумма кредита в данных банка неизвестна.",
    "debt_maturity": "Срок кредита в данных банка неизвестен.",
}
_ROLE = {
    "default": "Порядок блоков: что произошло, почему это важно компании, на что смотреть.",
    "cfo": "В блоке «почему важно» сначала стоимость денег и расходы. Факты и вывод не меняй.",
    "ceo": "В блоке «на что смотреть» сначала устойчивость выручки. Факты и вывод не меняй.",
}
_CLOSED = {"revenue", "region", "fts_countries", "workers_class", "known_counterparty_volume_share", "fx_confidence"}
_PARTY = re.compile(r"\bпоставщик\w*\b|\bпокупател(?:ь|я|и|ем)\b", re.IGNORECASE)
TAIL_MARK = " [хвост удалён]"
_CFO_FIRST = ("debt_service_cost", "fx_cost", "labor_cost")
_REVENUE_MECHANISMS = {"demand_shift", "commodity_revenue"}


def _pick_alfa_card(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cards:
        return None
    preferred = [card for card in cards if str(card.get("claim_type") or "") in BANK_VIEW_TYPES]
    return (preferred or cards)[0]


def _features(state: dict) -> dict[str, Any]:
    return ((state.get("views") or {}).get("profile") or {}).get("features") or {}


def _user(state: dict) -> dict[str, str]:
    return {str(item["field"]): str(item["value"]) for item in state.get("user_provided") or []}


def feature_ref(mechanism_id: str, features: dict[str, Any]) -> str:
    if mechanism_id == "debt_service_cost":
        if features.get("has_credit") is True:
            return "has_credit"
        if features.get("has_leasing") is True:
            return "has_leasing"
        return "has_credit"
    if mechanism_id == "import_restrictions":
        if features.get("fx_exposure") == "known":
            return "fx_exposure"
        return "fts_countries"
    if mechanism_id == "demand_shift":
        if features.get("has_factoring") is True:
            return "has_factoring"
        if features.get("has_rko") is True:
            return "has_rko"
        return "industry_group"
    if mechanism_id == "input_cost" and features.get("has_credit") is True:
        return "has_credit"
    return {
        "deposit_yield": "has_deposit",
        "fx_cost": "fx_exposure",
        "labor_cost": "has_payroll",
        "commodity_revenue": "industry_group",
        "tax_burden": "has_rko",
        "regulatory_cost": "industry_group",
        "guarantee_risk": "has_guarantees",
        "input_cost": "industry_group",
        "demand_shift": "industry_group",
    }.get(mechanism_id, "industry_group")


def _profile_sentence(mechanism_id: str, features: dict[str, Any]) -> str:
    if mechanism_id == "debt_service_cost":
        parts = []
        if features.get("has_credit") is True:
            parts.append("кредит")
        if features.get("has_leasing") is True:
            parts.append("лизинг")
        if parts:
            return "В профиле есть " + " и ".join(parts) + "."
        return "В профиле есть признак долгового контура."
    if mechanism_id == "deposit_yield":
        return "В профиле есть депозит."
    if mechanism_id == "fx_cost":
        return "Валютный контур в профиле отмечен."
    if mechanism_id == "import_restrictions":
        return "В профиле отмечен контур внешней торговли."
    if mechanism_id == "labor_cost":
        return "В профиле есть фонд оплаты труда."
    if mechanism_id == "tax_burden":
        return "В профиле есть расчётный контур."
    if mechanism_id == "guarantee_risk":
        return "В профиле есть гарантии."
    if mechanism_id == "commodity_revenue":
        return "Отрасль компании относится к профильному сырью."
    if mechanism_id == "demand_shift":
        return "Отрасль компании связана со спросом."
    if mechanism_id == "regulatory_cost":
        return "Отрасль компании попадает под отраслевые правила."
    return "Себестоимость компании связана с входящими ресурсами."


def _user_sentence(field_name: str, value: str) -> str:
    if field_name == "debt_rate_type" and value == "floating":
        return "По данным, которые вы указали и банк не проверял, ставка плавающая, поэтому платёж может измениться вместе с ключевой ставкой."
    if field_name == "debt_rate_type" and value == "fixed":
        return "По данным, которые вы указали и банк не проверял, ставка фиксированная, поэтому само решение по ключевой ставке платёж не пересчитывает."
    if field_name in {"fx_expense_share", "fx_revenue_share"}:
        return "По данным, которые вы указали и банк не проверял, валютная доля относится к выбранному диапазону."
    return "По данным, которые вы указали и банк не проверял, уточнение учтено в условном выводе."


def _tail(mechanism_id: str, state: dict, config: AgentConfig) -> str:
    spec = config.mechanisms.get(mechanism_id)
    if spec is None or not spec.critical_fields:
        return ""
    provided = _user(state)
    parts = []
    for field_name in spec.critical_fields:
        value = provided.get(field_name)
        if value and value != "unknown":
            parts.append(_user_sentence(field_name, value))
        else:
            parts.append(_CONDITIONAL.get(field_name, "Критичное поле в данных банка неизвестно."))
    return " ".join(parts)


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


def build_context_pack(state: dict, config: AgentConfig) -> str:
    evidence = state.get("evidence") or {}
    budget = config.context_tokens * config.chars_per_token
    features = _public_features(_features(state), config)
    synthetic = not _closed_ok(config)
    fact_lines = []
    for fact in (evidence.get("event") or {}).get("facts") or []:
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
    mechanisms = ", ".join(state.get("mechanism_ids") or []) or "нет"
    profile_note = "Профиль синтетический: суммы и идентификаторы сняты." if synthetic else "Признаки компании."
    sections = [
        "Факты события:\n" + ("\n".join(fact_lines) or "нет"),
        f"Механизмы: {mechanisms}. {profile_note} Признаки: {features}.",
    ]
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
    share = max(1, budget // len(sections))
    body = [clip_text(section, share) for section in sections]
    text, _hits = redact_text("\n\n".join(body))
    return text


def template_draft(state: dict, config: AgentConfig) -> dict[str, Any]:
    features = _features(state)
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
    reasons = state.get("reasons") or {}
    for mechanism_id in state.get("mechanism_ids") or []:
        sentence = _profile_sentence(mechanism_id, features)
        reason = reasons.get(mechanism_id) or mechanism_id
        tail = _tail(mechanism_id, state, config)
        text = " ".join(part for part in (sentence, reason + ".", tail) if part)
        claims.append(
            {
                "id": f"c_model_{mechanism_id}",
                "layer": "model",
                "text": text,
                "refs": [],
                "feature_refs": [feature_ref(mechanism_id, features)],
                "mechanism_id": mechanism_id,
            }
        )
    extra_sources = {}
    for item in state.get("calc_results") or []:
        mechanism_id = _calc_mechanism(item.get("fn"), state)
        if mechanism_id is None:
            continue
        claims.append(
            {
                "id": f"c_{item['calc_id'].replace(':', '_')}",
                "layer": "model",
                "text": item["text"],
                "refs": [],
                "feature_refs": [feature_ref(mechanism_id, features)],
                "mechanism_id": mechanism_id,
                "calc_refs": [item["calc_id"]],
            }
        )
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
    event_risks = [str(item).strip() for item in (event.get("risks") or []) if str(item).strip()]
    event_opportunities = [str(item).strip() for item in (event.get("opportunities") or []) if str(item).strip()]
    assumptions = []
    if not event_risks:
        assumptions.append("Отдельные риски в тексте события не названы.")
    if not event_opportunities:
        assumptions.append("Отдельные возможности в тексте события не названы.")
    if impact["missing_critical"]:
        assumptions.append("Критичное поле в данных банка не заполнено, поэтому формулировка условная.")
    if not cards:
        assumptions.append("По этой теме нет карточки Альфа-Банка.")
    if any(card.get("stale") for card in cards):
        assumptions.append("Есть устаревшая карточка аналитики, она помечена.")
    if any(item.get("illustrative") for item in state.get("calc_results") or []):
        assumptions.append("Количественная оценка иллюстративная и не относится к сумме по компании.")
    headline = _HEADLINES.get((state.get("mechanism_ids") or [""])[0], "Событие связано с профилем компании")
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
            headline = "Событие связано с профилем компании"
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
            headline = "Связь с профилем компании не подтверждена"
    by_layer = lambda layer: [item["id"] for item in claims if item["layer"] == layer]
    why = by_layer("model") + by_layer("alfa")
    watch = by_layer("hypothesis")
    role = str(state.get("role") or "")
    if role == "cfo":
        why = _prioritize_mechanisms(claims, why, _CFO_FIRST)
    elif role == "ceo":
        watch = _revenue_first(claims, watch)
    return {
        "claims": claims,
        "headline": headline,
        "what_happened": by_layer("fact"),
        "why_it_matters": why,
        "what_to_watch": watch,
        "impact_rationale": impact["rationale"],
        "impact_level": impact["level"],
        "impact_definition": impact["definition"],
        "impact_score": impact["score"],
        "scenarios": [
            {
                "name": "условный масштаб",
                "logic": "Если недостающие данные появятся, масштаб эффекта станет конкретнее.",
                "plausibility": "medium",
                "support_refs": list(fact_refs[:1]),
            }
        ],
        "assumptions": assumptions,
        "risks": event_risks,
        "opportunities": event_opportunities,
        "bank_block_ids": by_layer("alfa"),
        "disclaimer": config.disclaimer,
        "diverges_from_bank_view": False,
        "confidence": 0.42 if impact["missing_critical"] else 0.62,
        "extra_sources": extra_sources,
    }


def _calc_mechanism(fn: str | None, state: dict) -> str | None:
    mapping = {
        "rate_sensitivity": "debt_service_cost",
        "fx_sensitivity": "fx_cost",
        "payroll_cost_change": "labor_cost",
    }
    mechanism_id = mapping.get(fn or "")
    if mechanism_id in (state.get("mechanism_ids") or []):
        return mechanism_id
    return None


def _prioritize_mechanisms(claims: list[dict[str, Any]], ids: list[str], order: tuple[str, ...]) -> list[str]:
    by_id = {item["id"]: item for item in claims}
    head: list[str] = []
    for mechanism_id in order:
        for claim_id in ids:
            claim = by_id.get(claim_id) or {}
            if claim.get("mechanism_id") == mechanism_id and claim_id not in head:
                head.append(claim_id)
    return head + [claim_id for claim_id in ids if claim_id not in head]


def _revenue_first(claims: list[dict[str, Any]], ids: list[str]) -> list[str]:
    by_id = {item["id"]: item for item in claims}

    def is_revenue(claim_id: str) -> bool:
        claim = by_id.get(claim_id) or {}
        text = str(claim.get("text") or "").lower()
        mechanism = str(claim.get("mechanism_id") or "")
        return mechanism in _REVENUE_MECHANISMS or "выруч" in text

    head = [claim_id for claim_id in ids if is_revenue(claim_id)]
    return head + [claim_id for claim_id in ids if claim_id not in head]


def _unsafe(text: str) -> bool:
    return bool(_PARTY.search(text or ""))


def llm_failure_kind(error: str | None) -> str:
    text = error or ""
    markers = ("JSON", "ValidationError", "validation error", "схем", "Input should", "literal_error", "extra_forbidden")
    if any(token in text for token in markers):
        return "invalid_json"
    return "unavailable"
