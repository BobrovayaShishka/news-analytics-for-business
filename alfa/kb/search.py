"""Поиск по аналитике. Индекс можно передать снаружи, чтобы не грузить bge-m3."""

from __future__ import annotations

from datetime import datetime

from alfa.core.config import load_runtime_config
from alfa.core.schemas import SearchHit
from alfa.data.factory import build_sources
from alfa.kb.index import AnalyticsIndex, HashingEmbedder, build_index, load_embedder


def build_default_index(*, prefer_real: bool = False) -> AnalyticsIndex:
    cfg = load_runtime_config()
    _, store = build_sources(cfg)
    if prefer_real:
        embedder, _note = load_embedder(cfg, prefer_real=True)
    else:
        embedder = HashingEmbedder()
    return build_index(
        store.get_chunks(),
        store.get_cards(),
        embedder,
        rrf_k=int(cfg["search"]["rrf_k"]),
    )


def search_analytics(
    query: str,
    *,
    topics: list[str] | None = None,
    as_of: datetime | None = None,
    k: int = 5,
    mode: str = "hybrid",
    include_cards: bool = True,
    include_chunks: bool = True,
    index: AnalyticsIndex | None = None,
) -> list[SearchHit]:
    active = index or build_default_index(prefer_real=False)
    return active.search(
        query,
        topics=topics,
        as_of=as_of,
        k=k,
        mode=mode,
        include_cards=include_cards,
        include_chunks=include_chunks,
    )
