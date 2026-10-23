"""Ранжирование клиент × событие и персональная лента.

Публичные функции: build_feature_matrix, score_event, build_feed,
clients_for_event, explain. score_event и clients_for_event — пакетный контур.
build_feed и explain читают client_id только из сессии.
"""

from alfa.ranking.service import (
    build_feature_matrix,
    build_feed,
    clients_for_event,
    explain,
    score_event,
    score_incremental,
)

__all__ = [
    "build_feature_matrix",
    "score_event",
    "build_feed",
    "clients_for_event",
    "explain",
    "score_incremental",
]
