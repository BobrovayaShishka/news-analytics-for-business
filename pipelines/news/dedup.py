"""Три уровня дедупа: точный хеш, SimHash, косинус эмбеддингов в окне времени."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone

from alfa.core.logging import get_logger
from alfa.core.timeutil import SECONDS_PER_HOUR
from alfa.kb.index import BgeM3Embedder, HashingEmbedder
from pipelines.news.textutil import hamming_hex

logger = get_logger("news")

TRUST_RANK = {"official": 3, "major_media": 2, "industry": 1, "unverified": 0}


def build_embedder(settings: dict):
    dedup = settings.get("dedup") or {}
    name = str(dedup.get("embedder") or "hashing")
    if name == "bge-m3":
        try:
            from alfa.core.config import resolve_path

            cache = resolve_path(settings.get("embedding_cache") or "data/cache/embeddings")
            return BgeM3Embedder(
                cache_folder=str(cache),
                max_seq_length=int(dedup["bge_max_seq_length"]),
                batch_size=int(dedup["bge_batch_size"]),
            )
        except Exception as exc:
            logger.info("bge-m3 недоступен, смысловой дедуп на hashing: %s", exc)
            return HashingEmbedder()
    return HashingEmbedder()


def _stamp(row) -> datetime:
    raw = row["published_at"] or row["fetched_at"]
    return datetime.fromisoformat(raw)


def _within(left, right, hours: int) -> bool:
    return abs((_stamp(left) - _stamp(right)).total_seconds()) <= hours * SECONDS_PER_HOUR


def choose_primary(members: list) -> dict:
    def key(row):
        published = row["published_at"] or "9999-12-31T00:00:00+00:00"
        return (
            -TRUST_RANK.get(row["trust_tier"], 0),
            -int(row["text_len"]),
            published,
            row["id"],
        )

    return min(members, key=key)


def recluster(store, embedder, settings: dict, focus_ids: list[str]) -> int:
    """Пересобирает кластеры вокруг новых документов. Возвращает число кластеров размера > 1."""
    if not focus_ids:
        return 0
    dedup = settings.get("dedup") or {}
    window = int(dedup.get("semantic_window_hours") or 72)
    max_distance = int(dedup.get("simhash_max_distance") if dedup.get("simhash_max_distance") is not None else 3)
    threshold = float(dedup.get("semantic_threshold") or 0.92)
    pool = store.rows_for_dedup(focus_ids, window)
    if not pool:
        return 0
    parent = {row["id"]: row["id"] for row in pool}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_hash: dict[str, list] = defaultdict(list)
    by_url: dict[str, list] = defaultdict(list)
    for row in pool:
        by_hash[row["text_hash"]].append(row)
        by_url[row["canonical_url"]].append(row)
    for group in list(by_hash.values()) + list(by_url.values()):
        head = group[0]["id"]
        for other in group[1:]:
            union(head, other["id"])

    if max_distance >= 0:
        for index, left in enumerate(pool):
            for right in pool[index + 1 :]:
                if hamming_hex(left["simhash"], right["simhash"]) <= max_distance:
                    union(left["id"], right["id"])

    vectors = embedder.encode([f"{row['title']}\n{row['text']}" for row in pool])
    for index, left in enumerate(pool):
        for offset, right in enumerate(pool[index + 1 :], start=index + 1):
            if not _within(left, right, window):
                continue
            score = float(vectors[index] @ vectors[offset])
            if score >= threshold:
                union(left["id"], right["id"])

    groups: dict[str, list] = defaultdict(list)
    for row in pool:
        groups[find(row["id"])].append(row)
    multi = 0
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for members in groups.values():
        existing = sorted(row["cluster_id"] for row in members if row["cluster_id"])
        if existing:
            cluster_id = existing[0]
        else:
            key = "|".join(sorted(row["canonical_url"] for row in members))
            cluster_id = "cl:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        primary = choose_primary(members)
        echoes = [row for row in members if row["id"] != primary["id"]]
        echo_sources = sorted({row["source"] for row in echoes})
        if len(members) > 1:
            multi += 1
        for row in members:
            store.update_cluster(
                row["id"],
                cluster_id=cluster_id,
                is_primary=row["id"] == primary["id"],
                echo_count=len(echoes),
                echo_sources=echo_sources,
            )
        store.save_cluster(cluster_id, primary["id"], len(echoes), echo_sources, now)
    store.refresh_queue([row["id"] for row in pool])
    return multi
