"""Человекочитаемое объяснение релевантности без жаргона весов и компонентов."""

from __future__ import annotations

from typing import Any

from alfa.ranking.config import RankingConfig, RuleSpec
from alfa.ranking.models import ClientRow, FiredRule, LinkHypothesis, RankEvent, RelevanceSignal, SignalSource
from alfa.ranking.open_world import bridge_rules

# requires / feature attr → путь в профиле (как в relevance.example)
_ATTR_FEATURE: dict[str, str] = {
    "has_credit": "bank_products.loans",
    "has_leasing": "bank_products.leasing",
    "has_deposit": "bank_products.deposits",
    "has_guarantees": "bank_products.guarantees",
    "has_factoring": "bank_products.factoring",
    "has_payroll": "bank_products.payroll",
    "has_rko": "bank_products.rko",
    "fx_exposure": "trade.fts_flag",
    "workers_class": "scale.workers_class",
    "market_rank": "scale.market_position.rank",
    "industry_group": "activity.industry_group",
    "industry_theme": "activity.industry_theme",
    "size_class": "scale.size_class",
    "has_payment_data": "activity.industry_theme",
}

# Короткий префикс к scoring.reasons[mechanism_id]
_ATTR_LEAD: dict[str, str] = {
    "has_credit": "Есть кредитный продукт",
    "has_leasing": "Есть лизинг",
    "has_deposit": "Есть депозит",
    "has_guarantees": "Есть гарантии",
    "has_factoring": "Есть факторинг",
    "has_payroll": "Есть зарплатный проект",
    "has_rko": "Есть РКО",
    "fx_exposure": "Внешнеэкономический контур известен",
    "workers_class": "Класс численности известен",
    "market_rank": "Крупный оборот внутри ОКВЭД {okved2}",
    "industry_group": "Отрасль клиента связана с темой события",
    "industry_theme": "Отрасль чувствительна к теме события",
    "size_class": "Класс размера известен",
    "has_payment_data": "Есть платёжные данные по отрасли",
}


def _attr(row: ClientRow, name: str) -> Any:
    return getattr(row, name, None)


def _rule_by_id(rule_id: str, config: RankingConfig) -> RuleSpec | None:
    for rule in config.rules:
        if rule.rule_id == rule_id:
            return rule
    for rule in bridge_rules():
        if rule.rule_id == rule_id:
            return rule
    return None


def _signal_attr(spec: RuleSpec | None) -> str | None:
    if spec is None:
        return None
    for name in spec.requires:
        if name in _ATTR_FEATURE:
            return name
    when = spec.when or {}
    feature = when.get("feature")
    if isinstance(feature, str) and feature in _ATTR_FEATURE:
        return feature
    return None


def _context_bits(row: ClientRow, config: RankingConfig) -> list[str]:
    bits: list[str] = []
    if row.industry_group:
        bits.append(f"отрасль «{config.industry_label(row.industry_group)}»")
    if row.size_class:
        bits.append(f"класс размера {row.size_class}")
    if row.market_rank is not None and row.market_peers is not None:
        okved = row.market_okved2 or "—"
        bits.append(f"ранг {row.market_rank} из {row.market_peers} по выручке в ОКВЭД {okved}")
    products = [
        name.removeprefix("has_")
        for name in ("has_credit", "has_leasing", "has_deposit", "has_guarantees", "has_factoring", "has_payroll", "has_rko")
        if _attr(row, name) is True
    ]
    if products:
        bits.append("продукты: " + ", ".join(products))
    if row.fx_exposure == "known":
        countries = ", ".join(row.fts_countries or []) or "страна не указана"
        bits.append(f"ВЭД известен ({countries})")
    if row.region:
        bits.append(f"регион {row.region}")
    return bits


def _signal_for_rule(rule: FiredRule, row: ClientRow, config: RankingConfig) -> RelevanceSignal | None:
    spec = _rule_by_id(rule.rule_id, config)
    attr = _signal_attr(spec)
    if attr is None:
        return RelevanceSignal(
            feature=f"mechanism.{rule.mechanism_id}",
            value=True,
            reason=config.reason(rule.mechanism_id) or f"Сработало правило «{rule.rule_id}».",
        )
    feature = _ATTR_FEATURE[attr]
    value = _attr(row, attr)
    if attr == "fx_exposure" and value == "known":
        value = True
    if attr == "industry_theme" and value is None:
        value = _attr(row, "industry_group")
    lead = _ATTR_LEAD.get(attr, "").format(okved2=row.market_okved2 or "клиента")
    mech = config.reason(rule.mechanism_id).rstrip(".")
    if lead and mech:
        reason = f"{lead}: {mech}."
    elif lead:
        reason = f"{lead}."
    else:
        reason = f"{mech}." if mech else f"Сработало правило «{rule.rule_id}»."
    return RelevanceSignal(feature=feature, value=value, reason=reason)


def human_explanation(
    row: ClientRow,
    event: RankEvent,
    fired: list[FiredRule],
    config: RankingConfig,
    *,
    group_signal: SignalSource | None = None,
    graph_signal: SignalSource | None = None,
    hypotheses: list[LinkHypothesis] | None = None,
) -> tuple[str, list[RelevanceSignal]]:
    """Две–три фразы: контекст клиента и почему событие значимо. Без S_* и весов."""
    signals: list[RelevanceSignal] = []
    seen_features: set[str] = set()
    for rule in fired:
        signal = _signal_for_rule(rule, row, config)
        if signal is None or signal.feature in seen_features:
            continue
        seen_features.add(signal.feature)
        signals.append(signal)

    why: list[str] = [item.reason.rstrip(".") for item in signals[:3]]
    for mechanism_id in {rule.mechanism_id for rule in fired}:
        text = config.reason(mechanism_id)
        if text and text not in why and len(why) < 3:
            if not any(text.casefold() in item.casefold() for item in why):
                why.append(text.rstrip("."))

    if group_signal is not None and len(why) < 3:
        why.append("признак усилен за счёт компании из той же группы")
    if graph_signal is not None and len(why) < 3:
        why.append("платёжное окружение клиента связано с темой события")
    if hypotheses:
        for item in hypotheses[:1]:
            if len(why) >= 3:
                break
            why.append(item.text.rstrip("."))

    context = _context_bits(row, config)
    context_text = "; ".join(context) if context else "профиль частично известен"
    event_label = event.title or event.event_type
    if why:
        why_text = "; ".join(why) + "."
        summary = f"Контекст: {context_text}. Событие «{event_label}» значимо, потому что {why_text}"
    else:
        summary = (
            f"Контекст: {context_text}. "
            f"Событие «{event_label}» отобрано по совпадению признаков профиля с темой публикации."
        )
    return summary, signals
