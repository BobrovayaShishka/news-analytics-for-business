"""Канонизация сущностей события. Точное совпадение, иначе косинус hashing выше порога."""

from __future__ import annotations

import re

import numpy as np

from alfa.kb.index import HashingEmbedder
from alfa.ranking.config import RankingConfig
from alfa.ranking.models import LinkedEntity
from alfa.settings import longest_prefix, norm_label, norm_place


def themes_for_okved(okved_code: str | None, industry_group: str | None, themes: dict) -> list[str]:
    """Префикс ОКВЭД важнее отраслевой группы. Золотодобыча не равна опту металлов."""
    prefixes = themes.get("okved_prefixes") or {}
    code = (okved_code or "").strip()
    if code:
        matched = longest_prefix(code, prefixes)
        if matched:
            return [str(item) for item in matched]
    table = themes.get("industry_group") or {}
    if industry_group and industry_group not in {"unknown", "other", ""}:
        mapped = table.get(industry_group)
        if mapped:
            return [str(item) for item in mapped]
    return []


def _embedder(dim: int) -> HashingEmbedder:
    return HashingEmbedder(dim=dim)


def link_text(text: str, table: dict, *, dim: int, threshold: float) -> tuple[str, bool] | None:
    """Возвращает (канон, fuzzy). Канон словаря может быть строкой или списком/словарём — наружу строка-ключ."""
    key = norm_label(text)
    if not key:
        return None
    normalized = {norm_label(str(name)): name for name in table}
    if key in normalized:
        return str(normalized[key]), False
    labels = list(table)
    if not labels:
        return None
    embedder = _embedder(dim)
    matrix = embedder.embed([key, *[norm_label(str(item)) for item in labels]])
    query = matrix[0]
    scores = matrix[1:] @ query
    best = int(np.argmax(scores))
    if float(scores[best]) < threshold:
        return None
    return str(labels[best]), True


def _canonical_industry(payload: object) -> tuple[str, list[str]]:
    if isinstance(payload, dict):
        group = str(payload.get("industry_group") or "")
        themes = [str(item) for item in payload.get("themes") or []]
        return group, themes
    return str(payload), []


def link_entities(raw_entities: list[dict], config: RankingConfig) -> list[LinkedEntity]:
    synonyms = config.synonyms
    dim = int(config.scoring["embedding"]["dim"])
    threshold = float(config.scoring["embedding"]["fuzzy_threshold"])
    linked: list[LinkedEntity] = []
    for entity in raw_entities:
        text = str(entity.get("text") or "")
        kind = str(entity.get("type") or "")
        if kind == "country":
            table = synonyms.get("countries") or {}
            hit = link_text(text, table, dim=dim, threshold=threshold)
            if hit:
                label, fuzzy = hit
                linked.append(LinkedEntity("country", str(table[label]), fuzzy, text))
        elif kind == "industry":
            table = synonyms.get("industries") or {}
            hit = link_text(text, table, dim=dim, threshold=threshold)
            if hit:
                label, fuzzy = hit
                group, themes = _canonical_industry(table[label])
                linked.append(LinkedEntity("industry", group, fuzzy, text))
                for theme in themes:
                    linked.append(LinkedEntity("theme", theme, fuzzy, text))
        elif kind == "region":
            table = synonyms.get("regions") or {}
            hit = link_text(text, table, dim=dim, threshold=threshold)
            if hit:
                label, fuzzy = hit
                linked.append(LinkedEntity("region", str(table[label]), fuzzy, text))
            else:
                place = norm_place(text)
                if place:
                    linked.append(LinkedEntity("region", place, False, text))
        elif kind == "currency":
            table = synonyms.get("currencies") or {}
            hit = link_text(text, table, dim=dim, threshold=threshold)
            if hit:
                label, fuzzy = hit
                linked.append(LinkedEntity("currency", str(table[label]), fuzzy, text))
    return linked


def country_code(value: str, config: RankingConfig) -> str:
    table = config.synonyms.get("countries") or {}
    key = norm_label(value)
    for name, code in table.items():
        if norm_label(str(name)) == key:
            return str(code)
    if re.fullmatch(r"[A-Za-z]{2,3}", value.strip()):
        return value.strip().upper()
    return key
