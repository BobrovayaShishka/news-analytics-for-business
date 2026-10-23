"""Настройки новостей и реестр sources.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from alfa.core.config import resolve_path
from alfa.core.ids import SLUG_RE
from pipelines.news.models import SourceSpec
from pipelines.news.textutil import (
    BGE_ENCODE_BATCH,
    BGE_MAX_SEQ_LENGTH,
    CYRILLIC_LANG_SHARE,
    LATIN_LANG_SHARE,
    SIMHASH_BITS,
    SIMHASH_GRAMS,
)

DEFAULTS: dict = {
    "sources_path": "pipelines/news/sources.yaml",
    "db_path": "data/news/news.sqlite",
    "raw_dir": "data/news/raw",
    "raw_ttl_days": 14,
    "min_text_chars": 400,
    "max_text_chars": 20000,
    "hard_text_chars": 100000,
    "preview_chars": 400,
    "target_langs": ["ru"],
    "workers": 6,
    "timeout_s": 20,
    "max_retries": 3,
    "backoff_s": 0.4,
    "max_redirects": 5,
    "max_response_bytes": 2_000_000,
    "per_domain_rps": 1.0,
    "max_items_per_source": 40,
    "user_agent": "alfa-hackaton-news/1.0 (educational hackathon; local research)",
    "title_sources": ["РБК", "Интерфакс", "Ведомости", "Коммерсантъ", "Минфин", "Банк России"],
    "strip_patterns": [
        r"(?i)\bчитайте также\b[^\n]*",
        r"(?i)\bфото:\s*[^\n]*",
        r"(?i)(?:^|\n)\s*реклама\s*(?=\n|$)",
        r"(?i)\bподписывайтесь\b[^\n]*",
        r"(?i)\bматериал носит информационный характер\b[^\n]*",
    ],
    "paywall_markers": [
        "доступно по подписке",
        "оформите подписку",
        "только для подписчиков",
        "платный доступ",
        "этот материал доступен только",
    ],
    "suspicious_patterns": [
        r"игнорир\w*\s+предыдущ\w*\s+инструкц",
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"забудь\s+(все\s+)?инструкц",
        r"you\s+are\s+now\s+",
        r"system\s+prompt",
    ],
    "prefilter": {
        "url_parts": ["/sport", "/sports", "/culture", "/kino", "/style", "/auto/"],
        "blacklist": ["футбол", "хоккей", "биатлон", "кинопремьера", "сериал", "оскар"],
    },
    "lang": {
        "cyrillic_share": CYRILLIC_LANG_SHARE,
        "latin_share": LATIN_LANG_SHARE,
    },
    "dedup": {
        "simhash_bits": SIMHASH_BITS,
        "simhash_grams": SIMHASH_GRAMS,
        "simhash_max_distance": 3,
        "semantic_threshold": 0.92,
        "semantic_window_hours": 72,
        "embedder": "bge-m3",
        "bge_max_seq_length": BGE_MAX_SEQ_LENGTH,
        "bge_batch_size": BGE_ENCODE_BATCH,
    },
    "health": {
        "degraded_errors": 2,
        "down_errors": 4,
        "empty_ratio_degraded": 0.8,
        "low_volume_ratio": 0.2,
        "fallback_boost": 0.5,
    },
    "report_sample_size": 8,
}


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def news_settings(config: dict | None) -> dict:
    block = {}
    if config:
        raw = config.get("news") or {}
        if isinstance(raw, dict):
            block = raw
    return _merge(DEFAULTS, block)


def load_sources(path: str | Path) -> list[SourceSpec]:
    file_path = resolve_path(path)
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    rows = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"в {file_path} нет списка sources")
    specs: list[SourceSpec] = []
    seen_names: set[str] = set()
    seen_feeds: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("запись источника должна быть словарём")
        name = str(row.get("name") or "")
        if not SLUG_RE.fullmatch(name):
            raise ValueError(f"имя источника должно быть слагом [A-Za-z0-9_-]+, получено {name!r}")
        if name in seen_names:
            raise ValueError(f"повторяется источник {name}")
        seen_names.add(name)
        kind = str(row.get("type") or "rss")
        if kind not in {"rss", "html", "api"}:
            raise ValueError(f"{name}: неизвестный type {kind}")
        trust = str(row.get("trust_tier") or "major_media")
        if trust not in {"official", "major_media", "industry", "unverified"}:
            raise ValueError(f"{name}: неизвестный trust_tier {trust}")
        feed = str(row.get("feed_url") or "")
        if feed:
            if feed in seen_feeds:
                raise ValueError(f"повторяется feed_url {feed}")
            seen_feeds.add(feed)
        role = str(row.get("role") or "primary")
        if role not in {"primary", "fallback"}:
            raise ValueError(f"{name}: role должен быть primary или fallback")
        specs.append(
            SourceSpec(
                name=name,
                type=kind,
                feed_url=feed,
                list_url=str(row.get("list_url") or ""),
                trust_tier=trust,
                topic_hints=[str(item) for item in row.get("topic_hints") or []],
                timezone=str(row.get("timezone") or "Europe/Moscow"),
                poll_interval_s=int(row.get("poll_interval_s") or (900 if trust == "official" else 1800)),
                fallback_group=str(row.get("fallback_group") or ""),
                role=role,
                enabled=bool(row.get("enabled", True)),
                selectors=dict(row.get("selectors") or {}),
                strip_patterns=[str(item) for item in row.get("strip_patterns") or []],
                notes=str(row.get("notes") or ""),
                access=dict(row.get("access") or {}),
            )
        )
    return specs
