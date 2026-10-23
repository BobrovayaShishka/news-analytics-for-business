from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
OKVED_PATH = Path(__file__).resolve().parent / "okved_to_theme.yaml"

_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‑": "-"})


def load_profile_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def profile_config_hash(config: dict) -> str:
    blob = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return "cfg-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def output_path(config: dict, key: str) -> Path:
    return ROOT / config["outputs"][key]


def _norm_core(value: str) -> str:
    text = str(value).strip().translate(_DASHES)
    return re.sub(r"\s+", " ", text).strip()


def norm_label(value: str | None) -> str:
    if value is None:
        return ""
    text = _norm_core(value).lower().replace("ё", "е")
    return re.sub(r"\s*-\s*", "-", text)


def norm_place(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = _norm_core(value).replace("ё", "е").casefold()
    return text or None


def split_countries(value: str | None, separators: list[str]) -> list[str]:
    if value is None or not str(value).strip():
        return []
    pattern = "|".join(re.escape(item) for item in separators)
    parts = re.split(pattern, str(value))
    seen: list[str] = []
    for part in parts:
        country = norm_place(part)
        if country and country not in seen:
            seen.append(country)
    return seen


def load_okved_map() -> dict[str, str]:
    with OKVED_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return {str(key): str(value) for key, value in raw.items()}


def longest_prefix(code: str, mapping: dict) -> object | None:
    """Самый длинный префикс кода. Карты ОКВЭД разные, правило одно."""
    best = None
    best_len = -1
    for prefix, value in mapping.items():
        key = str(prefix)
        if code == key or code.startswith(key + "."):
            if len(key) > best_len:
                best = value
                best_len = len(key)
    return best


def industry_group(okved_code: str | None, mapping: dict[str, str]) -> str:
    if not okved_code:
        return "unknown"
    return str(longest_prefix(okved_code.strip(), mapping) or "other")


def size_band(revenue: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels, strict=False):
        if revenue <= edge:
            return label
    return labels[-1]


def payment_reliability(known_share: float | None, edges: int, rules: dict) -> str:
    if edges <= 0 or known_share is None:
        return "none"
    if known_share < float(rules["medium_known_share"]) or edges < int(rules["min_edges"]):
        return "low"
    if known_share >= float(rules["high_known_share"]):
        return "high"
    return "medium"


def okved_section(okved_code: str | None) -> str | None:
    if not okved_code:
        return None
    head = okved_code.split(".", 1)[0]
    if not head.isdigit():
        return None
    return f"{int(head):02d}"
