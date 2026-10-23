"""Единицы времени. Сутки и час не размазаны по модулям как 86400 и 3600."""

from __future__ import annotations

from datetime import datetime, timezone

SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


def attach_utc(value: datetime) -> datetime:
    """Наивной дате ставит UTC. Уже осознанную не переводит в другой пояс."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
