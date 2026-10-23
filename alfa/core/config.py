"""Загрузка day-2 конфига. Секреты только из окружения, не из файла."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "config.yaml"
TAXONOMY_PATH = PACKAGE_ROOT / "config" / "taxonomy.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@lru_cache(maxsize=1)
def load_runtime_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {CONFIG_PATH}")
    return data


def runtime_config_hash(path: Path | None = None) -> str:
    raw = (path or CONFIG_PATH).read_bytes()
    return hashlib.sha256(raw).hexdigest()


def data_label(config: dict[str, Any] | None = None) -> str:
    """«данные: заглушка» или «данные: реальные» по data.origin / data.impl."""
    cfg = config or load_runtime_config()
    block = cfg.get("data") or {}
    origin = str(block.get("origin") or block.get("impl") or "stub").lower()
    if origin in {"real", "реальные", "live"}:
        return "данные: реальные"
    return "данные: заглушка"
