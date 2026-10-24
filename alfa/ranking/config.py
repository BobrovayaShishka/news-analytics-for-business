"""Загрузка scoring.yaml, sensitivity.yaml и словарей. Версии — хэш байтов файла."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from alfa.core.config import PACKAGE_ROOT, REPO_ROOT

CONFIG_DIR = PACKAGE_ROOT / "config"
SCORING_PATH = CONFIG_DIR / "scoring.yaml"
SENSITIVITY_PATH = CONFIG_DIR / "sensitivity.yaml"
SYNONYM_PATH = CONFIG_DIR / "entity_synonyms.yaml"
THEME_PATH = CONFIG_DIR / "okved_to_theme.yaml"
OPEN_WORLD_PATH = CONFIG_DIR / "open_world.yaml"


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def file_token(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return digest


@dataclass
class RuleSpec:
    rule_id: str
    event_types: frozenset[str]
    when: dict[str, Any]
    weight: float
    mechanism_id: str
    requires: tuple[str, ...]
    confidence_base: float
    entity_themes: frozenset[str]
    entity_boost: dict[str, Any] | None
    min_reliability: str | None


@dataclass
class RankingConfig:
    scoring: dict[str, Any]
    sensitivity: dict[str, Any]
    synonyms: dict[str, Any]
    themes: dict[str, Any]
    rules: list[RuleSpec]
    config_version: str
    weight_version: str
    reasons: dict[str, str] = field(default_factory=dict)

    def weight(self, name: str) -> float:
        return float(self.scoring["weights"][name])

    def tau_show(self, event_type: str) -> float:
        thresholds = self.scoring["thresholds"]
        critical = set(thresholds.get("critical_event_types") or [])
        if event_type in critical:
            return float(thresholds["tau_show_critical"])
        return float(thresholds["tau_show"])

    @property
    def sem_only_min(self) -> float:
        return float(self.scoring["thresholds"]["sem_only_min"])

    def half_life(self, event_type: str) -> float:
        block = self.scoring["freshness"]
        return float(block["by_event_type"].get(event_type, block["default_half_life_days"]))

    def window_days(self, event_type: str) -> float:
        block = self.scoring["windows_days"]
        return float(block.get(event_type, block["default"]))

    def trust(self, tier: str) -> float:
        table = self.scoring["trust"]
        return float(table.get(tier, table["unverified"]))

    def reason(self, mechanism_id: str) -> str:
        return self.reasons.get(mechanism_id, mechanism_id)

    def industry_label(self, code: str) -> str:
        labels = self.scoring.get("industry_labels") or {}
        return str(labels.get(code, code))


def rule_from_raw(item: dict[str, Any]) -> RuleSpec:
    """Единый парсер правила из sensitivity.yaml / open_world bridge_rules."""
    boost = item.get("entity_boost")
    return RuleSpec(
        rule_id=str(item["rule_id"]),
        event_types=frozenset(str(code) for code in item.get("event_types") or []),
        when=dict(item.get("when") or {}),
        weight=float(item["weight"]),
        mechanism_id=str(item["mechanism_id"]),
        requires=tuple(str(name) for name in item.get("requires") or []),
        confidence_base=float(item.get("confidence_base", 1.0)),
        entity_themes=frozenset(str(code) for code in item.get("entity_themes") or []),
        entity_boost=dict(boost) if isinstance(boost, dict) else None,
        min_reliability=(str(item["min_reliability"]) if item.get("min_reliability") else None),
    )


def _rules(raw: dict[str, Any]) -> list[RuleSpec]:
    return [rule_from_raw(item) for item in raw.get("rules") or []]


def _version(label: str, paths: list[Path]) -> str:
    blob = "|".join(file_token(path) for path in paths)
    return f"{label}+{blob}"


@lru_cache(maxsize=1)
def load_ranking_config() -> RankingConfig:
    scoring = _read(SCORING_PATH)
    sensitivity = _read(SENSITIVITY_PATH)
    synonyms = _read(SYNONYM_PATH)
    themes = _read(THEME_PATH)
    return RankingConfig(
        scoring=scoring,
        sensitivity=sensitivity,
        synonyms=synonyms,
        themes=themes,
        rules=_rules(sensitivity),
        config_version=_version(
            str(sensitivity.get("version", "sensitivity")),
            [SENSITIVITY_PATH, SYNONYM_PATH, THEME_PATH, OPEN_WORLD_PATH],
        ),
        weight_version=_version(str(scoring.get("version", "scoring")), [SCORING_PATH]),
        reasons={str(key): str(value) for key, value in (scoring.get("reasons") or {}).items()},
    )


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
