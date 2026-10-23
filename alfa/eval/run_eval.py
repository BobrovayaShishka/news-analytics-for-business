"""Прогон eval. По умолчанию извлечение идёт в LLMClient.

Без ключа клиент вернёт ok=False, отчёт всё равно пишется.
`--mock` подставляет фиктивные предсказания и нужен для локальной проверки отчёта.
Поиск считается офлайн. Если bge-m3 не поднимается, в отчёте будет hashing и причина.

Запуск:
  python -m alfa.eval.run_eval --task extraction --split dev --mock
  python -m alfa.eval.run_eval --task search --split dev --noise none
Переменные: ALFA_LLM_API_KEY, ALFA_LOG_LEVEL.
"""

from __future__ import annotations

import argparse
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alfa.core.config import data_label, load_runtime_config, resolve_path, runtime_config_hash
from alfa.core.schemas import EventExtraction
from alfa.data.factory import build_sources
from alfa.data.stubs.news_stub import read_jsonl
from alfa.data.stubs.noise import add_ocr_noise
from alfa.extraction.event_extractor import extract_many, prompt_version
from alfa.kb.index import HashingEmbedder, build_index, load_embedder
from alfa.eval.metrics import extraction_report, search_metrics


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def dataset_path(config: dict[str, Any], name: str) -> Path:
    return resolve_path(Path("alfa/eval/datasets") / name)


def _mock_prediction(gold: dict[str, Any], model: str, version: str) -> dict[str, Any]:
    facts = []
    for fact in gold.get("facts") or []:
        facts.append(
            {
                "value": fact.get("value"),
                "unit": fact.get("unit"),
                "date": fact.get("date"),
                "subject": fact.get("subject"),
                "unverified": False,
            }
        )
    return {
        "doc_id": gold["doc_id"],
        "event_type": gold["event_type"],
        "is_business_relevant": gold["is_business_relevant"],
        "facts": facts,
        "pred_event_type": gold["event_type"],
        "pred_relevant": gold["is_business_relevant"],
        "pred_facts": facts,
        "pred_status": "ok",
        "model": model,
    }


def _from_extraction(gold: dict[str, Any], item: EventExtraction) -> dict[str, Any]:
    return {
        "doc_id": gold["doc_id"],
        "event_type": gold["event_type"],
        "is_business_relevant": gold["is_business_relevant"],
        "facts": gold.get("facts") or [],
        "pred_event_type": None if item.event_type is None else str(item.event_type.value),
        "pred_relevant": item.is_business_relevant,
        "pred_facts": [fact.model_dump() for fact in item.facts],
        "pred_status": str(item.extraction_status.value),
        "model": item.model,
    }


def run_extraction(split: str, *, mock: bool, model: str | None) -> dict[str, Any]:
    cfg = load_runtime_config()
    version = prompt_version(cfg)
    resolved = model or cfg["llm"]["roles"]["extract"]
    gold_rows = _read_rows(dataset_path(cfg, f"{split}.jsonl"))
    news, _store = build_sources(cfg)
    by_id = {doc.id: doc for doc in news.list_news(None, None)}
    started = datetime.now(timezone.utc)
    if mock:
        paired = [_mock_prediction(row, resolved, version) for row in gold_rows]
        llm_calls = 0
    else:
        docs = [by_id[row["doc_id"]] for row in gold_rows]
        from alfa.core.llm import LLMClient

        client = LLMClient(cfg)
        if model:
            cfg = json.loads(json.dumps(cfg))
            cfg["llm"]["roles"]["extract"] = model
            client = LLMClient(cfg)
        batch = extract_many(docs, max_workers=4, client=client, config=cfg)
        paired = [_from_extraction(gold, item) for gold, item in zip(gold_rows, batch.items, strict=True)]
        llm_calls = client.call_count
        resolved = cfg["llm"]["roles"]["extract"]
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    metrics = extraction_report(paired)
    return {
        "task": "extraction",
        "split": split,
        "mock": mock,
        "model": resolved,
        "prompt_version": version,
        "cache_enabled": bool(cfg["llm"]["cache_enabled"]),
        "config_hash": runtime_config_hash(),
        "data_label": data_label(cfg),
        "llm_calls": llm_calls,
        "latency_ms": elapsed_ms,
        "metrics": metrics,
    }


def _noise_rate(cfg: dict[str, Any], level: str) -> float:
    rates = cfg["noise"]
    if level not in rates:
        raise SystemExit(f"unknown noise level: {level}")
    return float(rates[level])


def _collection_flags(name: str) -> tuple[bool, bool]:
    if name == "chunks":
        return False, True
    if name == "cards":
        return True, False
    if name == "cards+chunks":
        return True, True
    raise ValueError(name)


def run_search(split: str, *, noise: str, prefer_real: bool) -> dict[str, Any]:
    cfg = load_runtime_config()
    queries = _read_rows(dataset_path(cfg, "analytics_queries.jsonl"))
    _news, store = build_sources(cfg)
    chunks = store.get_chunks()
    cards = store.get_cards()
    if prefer_real:
        embedder, embedder_note = load_embedder(cfg, prefer_real=True)
    else:
        embedder, embedder_note = HashingEmbedder(), "hashing"
    levels = ["none", "low", "medium"]
    if noise not in levels:
        raise SystemExit(f"unknown noise level: {noise}")
    rows: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    for level in levels:
        rate = _noise_rate(cfg, level)
        noised = []
        for chunk in chunks:
            text = add_ocr_noise(
                chunk.text,
                rate=rate,
                seed=zlib.crc32(chunk.chunk_id.encode("utf-8")) % (2**31),
            )
            noised.append(chunk.model_copy(update={"text": text}))
        index = build_index(noised, cards, embedder, rrf_k=int(cfg["search"]["rrf_k"]))
        for mode in ("bm25", "dense", "hybrid"):
            for collection in ("chunks", "cards", "cards+chunks"):
                include_cards, include_chunks = _collection_flags(collection)
                for query in queries:
                    hits = index.search(
                        query["query"],
                        topics=query.get("topics") or None,
                        k=5,
                        mode=mode,
                        include_cards=include_cards,
                        include_chunks=include_chunks,
                    )
                    if collection == "chunks":
                        relevant = list(query.get("relevant_chunk_ids") or [])
                        ranked = [hit.id for hit in hits if str(hit.kind.value) == "chunk"]
                    elif collection == "cards":
                        relevant = list(query.get("relevant_card_ids") or [])
                        ranked = [hit.id for hit in hits if str(hit.kind.value) == "card"]
                    else:
                        relevant = list(query.get("relevant_card_ids") or []) + list(
                            query.get("relevant_chunk_ids") or []
                        )
                        ranked = [hit.id for hit in hits]
                    rows.append(
                        {
                            "mode": mode,
                            "collection": collection,
                            "noise": level,
                            "relevant": relevant,
                            "ranked": ranked,
                        }
                    )
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    return {
        "task": "search",
        "split": split,
        "noise_requested": noise,
        "model": None,
        "embedder": embedder_note,
        "prompt_version": prompt_version(cfg),
        "cache_enabled": bool(cfg["llm"]["cache_enabled"]),
        "config_hash": runtime_config_hash(),
        "data_label": data_label(cfg),
        "llm_calls": 0,
        "latency_ms": elapsed_ms,
        "metrics": search_metrics(rows),
    }


def write_report(payload: dict[str, Any], out_dir: Path | None = None) -> Path:
    cfg = load_runtime_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = out_dir or (resolve_path(cfg["paths"]["reports"]) / stamp)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"task: {payload.get('task')}",
        f"data_label: {payload.get('data_label')}",
        f"config_hash: {payload.get('config_hash')}",
        f"prompt_version: {payload.get('prompt_version')}",
        f"model: {payload.get('model')}",
        f"cache_enabled: {payload.get('cache_enabled')}",
        f"embedder: {payload.get('embedder')}",
        json.dumps(payload.get("metrics"), ensure_ascii=False, indent=2),
    ]
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def resolve_model_arg(name: str | None) -> str | None:
    if not name:
        return None
    roles = load_runtime_config()["llm"]["roles"]
    if name in roles:
        return str(roles[name])
    for model in roles.values():
        if str(model).startswith(name):
            return str(model)
    return name


def main(argv: list[str] | None = None) -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Day-2 eval")
    parser.add_argument("--task", required=True, choices=["extraction", "search"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--noise", default="none", choices=["none", "low", "medium"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--embedder", default="auto", choices=["auto", "hashing", "bge"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    model = resolve_model_arg(args.model)
    if args.task == "extraction":
        payload = run_extraction(args.split, mock=args.mock, model=model)
    else:
        prefer_real = args.embedder != "hashing"
        if args.embedder == "auto":
            prefer_real = True
        payload = run_search(args.split, noise=args.noise, prefer_real=prefer_real)
    out = write_report(payload, Path(args.out_dir) if args.out_dir else None)
    print(out)
    print(payload["data_label"], payload.get("embedder") or payload.get("model"))


if __name__ == "__main__":
    main()
