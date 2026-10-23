"""Предупреждение, если у прогноза нет metric или horizon."""

from __future__ import annotations

from alfa.core.logging import get_logger
from alfa.core.schemas import AnalyticsCard

logger = get_logger("alfa.kb.cards")


def validate_cards(cards: list[AnalyticsCard], *, strict_forecast: bool = False) -> list[str]:
    warnings: list[str] = []
    for card in cards:
        if str(card.claim_type.value) != "forecast":
            continue
        missing = [name for name in ("metric", "horizon") if not getattr(card, name)]
        if not missing:
            continue
        message = f"{card.card_id}: forecast without {', '.join(missing)}"
        warnings.append(message)
        logger.warning(message)
        if strict_forecast:
            raise ValueError(message)
    return warnings


def card_search_text(card: AnalyticsCard) -> str:
    parts = [card.claim]
    for value in (card.metric, card.value, card.unit, card.horizon):
        if value:
            parts.append(str(value))
    parts.extend(str(topic.value) for topic in card.topic)
    return "\n".join(parts)
