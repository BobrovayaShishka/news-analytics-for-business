"""Индексы чанков и карточек: BM25 и плотные вектора.

Лемматизация: pymorphy3, если пакет импортируется. Иначе токен остаётся
в нижнем регистре (стеммер Snowball/nltk в окружении не обязателен).
Плотный путь — BAAI/bge-m3 через sentence-transformers, с дисковым кэшем.
Тесты подставляют Embedder с детерминированными векторами и не качают веса.

score_components.bm25 / dense равны None, если канал в этом запросе не считался.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from alfa.core.config import load_runtime_config, resolve_path
from alfa.core.timeutil import attach_utc
from alfa.core.logging import get_logger
from alfa.core.schemas import AnalyticsCard, AnalyticsChunk, ScoreComponents, SearchHit
from alfa.kb.cards import card_search_text

logger = get_logger("alfa.kb.index")

_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_MORPH = None
_MORPH_READY = False


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _morph():
    global _MORPH, _MORPH_READY
    if _MORPH_READY:
        return _MORPH
    _MORPH_READY = True
    try:
        import pymorphy3

        _MORPH = pymorphy3.MorphAnalyzer()
        logger.info("tokenizer: pymorphy3")
    except Exception as exc:
        _MORPH = None
        logger.info("tokenizer: lowercase fallback (%s)", exc)
    return _MORPH


def tokenize(text: str) -> list[str]:
    morph = _morph()
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower().replace("ё", "е")):
        if any(ch.isdigit() for ch in raw) or morph is None:
            tokens.append(raw)
            continue
        tokens.append(morph.parse(raw)[0].normal_form)
    return tokens or ["_empty"]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


DEFAULT_HASH_DIM = 256


class HashingEmbedder:
    """Мешок хешей токенов. Нужен, когда bge-m3 нельзя поднять, и в unit-тестах."""

    name = "hashing"

    def __init__(self, dim: int = DEFAULT_HASH_DIM):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                matrix[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        return _normalize(matrix)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)


class BgeM3Embedder:
    """Локальный BAAI/bge-m3. Сначала ищет уже скачанный снимок, без сети.

    Новостной дедуп передаёт cache_folder и длину последовательности.
    Поиск по аналитике ищет снимок в hf_cache и не ограничивает длину.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        hf_cache: str | Path | None = None,
        *,
        cache_folder: str | Path | None = None,
        max_seq_length: int | None = None,
        batch_size: int | None = None,
    ):
        self.name = model_name
        self.batch_size = batch_size
        from sentence_transformers import SentenceTransformer

        if cache_folder is not None:
            self.model = SentenceTransformer(model_name, cache_folder=str(cache_folder))
        else:
            snapshot = _local_snapshot(hf_cache)
            target = snapshot or model_name
            self.model = SentenceTransformer(target, local_files_only=snapshot is not None)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def embed(self, texts: list[str]) -> np.ndarray:
        options: dict = {"normalize_embeddings": True, "show_progress_bar": False}
        if self.batch_size is not None:
            options["batch_size"] = self.batch_size
        vectors = self.model.encode(list(texts), **options)
        return np.asarray(vectors, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = int(self.model.get_sentence_embedding_dimension() or 1024)
            return np.zeros((0, dim), dtype=np.float32)
        return self.embed(texts)


def _local_snapshot(hf_cache: str | Path | None) -> str | None:
    if hf_cache is None:
        return None
    root = resolve_path(hf_cache) / "models--BAAI--bge-m3" / "snapshots"
    if not root.exists():
        return None
    for snapshot in root.iterdir():
        if (snapshot / "pytorch_model.bin").exists() or (snapshot / "model.safetensors").exists():
            if (snapshot / "modules.json").exists():
                return str(snapshot)
    return None


class CachingEmbedder:
    def __init__(self, inner: Embedder, cache_dir: str | Path):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = resolve_path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray | None] = []
        missing_text: list[str] = []
        missing_at: list[int] = []
        for index, text in enumerate(texts):
            path = self._path(text)
            if path.exists():
                rows.append(np.load(path))
            else:
                rows.append(None)
                missing_text.append(text)
                missing_at.append(index)
        if missing_text:
            computed = self.inner.embed(missing_text)
            for offset, index in enumerate(missing_at):
                vector = np.asarray(computed[offset], dtype=np.float32)
                np.save(self._path(texts[index]), vector)
                rows[index] = vector
        if not rows:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalize(np.vstack(rows))

    def _path(self, text: str) -> Path:
        digest = hashlib.sha1(f"{self.name}\n{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"


def load_embedder(config: dict | None = None, *, prefer_real: bool = True) -> tuple[Embedder, str]:
    cfg = config or load_runtime_config()
    cache_dir = cfg["paths"]["embedding_cache"]
    model_name = cfg["search"]["embedder"]
    if prefer_real:
        try:
            real = BgeM3Embedder(model_name, cfg["paths"].get("hf_cache"))
            note = model_name
            return CachingEmbedder(real, cache_dir), note
        except Exception as exc:
            logger.warning("bge-m3 недоступен, плотный канал на хешах: %s", exc)
            note = f"hashing (bge-m3 unavailable: {exc.__class__.__name__}: {exc})"
            return CachingEmbedder(HashingEmbedder(), cache_dir), note
    return CachingEmbedder(HashingEmbedder(), cache_dir), "hashing"


@dataclass
class SearchRecord:
    kind: str
    id: str
    doc_id: str
    page: int
    text: str
    published_at: datetime
    topics: list[str] = field(default_factory=list)
    bbox_rel: list[float] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class CollectionIndex:
    records: list[SearchRecord]
    embedder: Embedder
    matrix: np.ndarray

    @classmethod
    def build(cls, records: list[SearchRecord], embedder: Embedder) -> CollectionIndex:
        if not records:
            return cls(records=[], embedder=embedder, matrix=np.zeros((0, 1), dtype=np.float32))
        matrix = embedder.embed([record.text for record in records])
        return cls(records=records, embedder=embedder, matrix=_normalize(matrix))


@dataclass
class AnalyticsIndex:
    chunks: CollectionIndex
    cards: CollectionIndex
    embedder_name: str
    rrf_k: int = 60

    def search(
        self,
        query: str,
        *,
        topics: list[str] | None = None,
        as_of: datetime | None = None,
        k: int = 5,
        mode: str = "hybrid",
        include_cards: bool = True,
        include_chunks: bool = True,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        if include_chunks:
            hits.extend(self._search_collection(self.chunks, query, topics=None, as_of=as_of, k=k, mode=mode))
        if include_cards:
            hits.extend(
                self._search_collection(self.cards, query, topics=topics, as_of=as_of, k=k, mode=mode)
            )
        hits.sort(key=lambda item: (-item.score, item.id))
        return hits[:k]

    def _search_collection(
        self,
        collection: CollectionIndex,
        query: str,
        *,
        topics: list[str] | None,
        as_of: datetime | None,
        k: int,
        mode: str,
    ) -> list[SearchHit]:
        chosen = [
            (index, record)
            for index, record in enumerate(collection.records)
            if _keep(record, topics, as_of)
        ]
        if not chosen:
            return []
        pool = max(k, 20)
        bm25_rank: dict[str, tuple[int, float]] = {}
        dense_rank: dict[str, tuple[int, float]] = {}
        if mode in {"bm25", "hybrid"}:
            bm25_rank = _bm25_ranks(query, [record for _, record in chosen])
        if mode in {"dense", "hybrid"}:
            dense_rank = _dense_ranks(
                query,
                collection.embedder,
                [record for _, record in chosen],
                collection.matrix[[index for index, _ in chosen]],
            )
        fused: list[SearchHit] = []
        ids = {record.id for _, record in chosen}
        for record_id in ids:
            bm25 = bm25_rank.get(record_id)
            dense = dense_rank.get(record_id)
            score = 0.0
            if mode == "bm25":
                score = 0.0 if bm25 is None else 1.0 / (self.rrf_k + bm25[0])
            elif mode == "dense":
                score = 0.0 if dense is None else 1.0 / (self.rrf_k + dense[0])
            else:
                if bm25 is not None:
                    score += 1.0 / (self.rrf_k + bm25[0])
                if dense is not None:
                    score += 1.0 / (self.rrf_k + dense[0])
            record = next(record for _, record in chosen if record.id == record_id)
            fused.append(_hit(record, score, bm25[1] if bm25 else None, dense[1] if dense else None, mode))
        fused.sort(key=lambda item: (-item.score, item.id))
        return fused[:pool]


def _keep(record: SearchRecord, topics: list[str] | None, as_of: datetime | None) -> bool:
    if as_of is not None and attach_utc(record.published_at) > attach_utc(as_of):
        return False
    if topics and record.kind == "card":
        if not set(record.topics) & set(topics):
            return False
    return True


def _bm25_ranks(query: str, records: list[SearchRecord]) -> dict[str, tuple[int, float]]:
    tokenized = [tokenize(record.text) for record in records]
    model = BM25Okapi(tokenized)
    scores = model.get_scores(tokenize(query))
    order = sorted(range(len(records)), key=lambda index: (-float(scores[index]), records[index].id))
    return {records[index].id: (rank + 1, float(scores[index])) for rank, index in enumerate(order)}


def _dense_ranks(
    query: str,
    embedder: Embedder,
    records: list[SearchRecord],
    matrix: np.ndarray,
) -> dict[str, tuple[int, float]]:
    query_vec = _normalize(embedder.embed([query]))[0]
    scores = matrix @ query_vec
    order = sorted(range(len(records)), key=lambda index: (-float(scores[index]), records[index].id))
    return {records[index].id: (rank + 1, float(scores[index])) for rank, index in enumerate(order)}


def _hit(
    record: SearchRecord,
    score: float,
    bm25: float | None,
    dense: float | None,
    mode: str,
) -> SearchHit:
    if mode == "bm25":
        dense = None
    elif mode == "dense":
        bm25 = None
    return SearchHit(
        kind=record.kind,  # type: ignore[arg-type]
        id=record.id,
        doc_id=record.doc_id,
        page=record.page,
        text=record.text,
        score=float(score),
        score_components=ScoreComponents(bm25=bm25, dense=dense),
        metadata={
            "bbox_rel": record.bbox_rel,
            "topics": record.topics,
            **record.extra,
        },
    )


def records_from_chunks(chunks: list[AnalyticsChunk]) -> list[SearchRecord]:
    return [
        SearchRecord(
            kind="chunk",
            id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            page=chunk.page,
            text=chunk.text,
            published_at=chunk.published_at,
            bbox_rel=list(chunk.bbox_rel),
            extra={"section": str(chunk.section.value), "verified": str(chunk.verified.value)},
        )
        for chunk in chunks
    ]


def records_from_cards(cards: list[AnalyticsCard]) -> list[SearchRecord]:
    return [
        SearchRecord(
            kind="card",
            id=card.card_id,
            doc_id=card.doc_id,
            page=card.page,
            text=card_search_text(card),
            published_at=card.published_at,
            topics=[str(topic.value) for topic in card.topic],
            bbox_rel=list(card.bbox_rel),
            extra={
                "claim_type": str(card.claim_type.value),
                "stance": str(card.stance.value),
                "verified": str(card.verified.value),
            },
        )
        for card in cards
    ]


def build_index(
    chunks: list[AnalyticsChunk],
    cards: list[AnalyticsCard],
    embedder: Embedder,
    *,
    rrf_k: int = 60,
) -> AnalyticsIndex:
    return AnalyticsIndex(
        chunks=CollectionIndex.build(records_from_chunks(chunks), embedder),
        cards=CollectionIndex.build(records_from_cards(cards), embedder),
        embedder_name=getattr(embedder, "name", embedder.__class__.__name__),
        rrf_k=rrf_k,
    )


def main() -> None:
    from alfa.data.factory import build_sources

    cfg = load_runtime_config()
    _, store = build_sources(cfg)
    embedder, note = load_embedder(cfg, prefer_real=True)
    index = build_index(
        store.get_chunks(),
        store.get_cards(),
        embedder,
        rrf_k=int(cfg["search"]["rrf_k"]),
    )
    print(
        f"index chunks={len(index.chunks.records)} cards={len(index.cards.records)} embedder={note}"
    )


if __name__ == "__main__":
    main()
