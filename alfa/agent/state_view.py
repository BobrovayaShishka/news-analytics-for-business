"""Чтение общих срезов agent state без дублирования accessors."""

from __future__ import annotations

from typing import Any


def features(state: dict) -> dict[str, Any]:
    return ((state.get("views") or {}).get("profile") or {}).get("features") or {}


def user_provided(state: dict) -> dict[str, str]:
    return {str(item["field"]): str(item["value"]) for item in state.get("user_provided") or []}


def mechanism_ids(state: dict) -> list[str]:
    return [str(item) for item in state.get("mechanism_ids") or [] if item]
