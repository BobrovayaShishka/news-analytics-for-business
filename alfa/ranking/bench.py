"""Масштабный прогон. Полная база пропускается, если profiles.parquet нет.

python -m alfa.ranking.bench
python -m alfa.ranking.bench --smoke
"""

from __future__ import annotations

import argparse
import json
import tracemalloc
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from alfa.ranking.config import load_ranking_config, repo_path
from alfa.ranking.engine import score_arrays
from alfa.ranking.matrix import assemble_matrix
from alfa.ranking.models import ClientRow, RankEvent

INDUSTRIES = (
    "retail",
    "manufacturing_machinery",
    "gold_mining",
    "land_transport",
    "mining_oil_gas",
    "real_estate",
    "wholesale_metals",
    "construction_buildings",
)
INDUSTRY_P = np.array([0.28, 0.18, 0.12, 0.12, 0.1, 0.08, 0.07, 0.05], dtype=np.float64)
SIZES = ("S1", "S2", "S3", "S4", "S5")


def archetype(row: ClientRow) -> tuple:
    return (
        bool(row.has_credit),
        bool(row.has_leasing),
        bool(row.has_deposit),
        bool(row.has_payroll),
        row.fx_exposure == "known",
        row.industry_group or "unknown",
        row.size_class or "unknown",
        bool(row.group_id),
    )


def synthetic_rows(n: int, seed: int) -> list[ClientRow]:
    rng = np.random.default_rng(seed)
    industries = list(INDUSTRIES)
    weights = INDUSTRY_P / INDUSTRY_P.sum()
    rows: list[ClientRow] = []
    groups = max(n // 8, 1)
    for index in range(n):
        industry = industries[int(rng.choice(len(industries), p=weights))]
        size = SIZES[int(rng.integers(0, len(SIZES)))]
        has_credit = bool(rng.random() < 0.18)
        has_leasing = bool(rng.random() < 0.12)
        has_deposit = bool(rng.random() < 0.08)
        has_payroll = bool(rng.random() < 0.2)
        fx_known = bool(rng.random() < 0.15)
        grouped = bool(rng.random() < 0.22)
        has_pay = bool(rng.random() < 0.4)
        mix = None
        reliability = "none"
        edges = None
        known = None
        if has_pay:
            other = industries[int(rng.integers(0, len(industries)))]
            mix = {other: float(rng.uniform(0.25, 0.9))}
            reliability = str(rng.choice(["low", "medium", "high"], p=[0.25, 0.4, 0.35]))
            edges = int(rng.integers(1, 48))
            known = float(rng.uniform(0.05, 0.95))
        rows.append(
            ClientRow(
                client_id=f"S{index:05d}",
                client_type="CLIENT",
                profile_version="synth-v1",
                has_credit=has_credit,
                has_leasing=has_leasing,
                has_deposit=has_deposit,
                has_payroll=has_payroll,
                has_rko=bool(rng.random() < 0.7),
                has_guarantees=False,
                has_factoring=False,
                industry_group=industry,
                themes=[industry],
                size_class=size,
                workers_class=int(rng.integers(0, 4)),
                fx_exposure="known" if fx_known else None,
                fts_countries=["CN"] if fx_known and rng.random() < 0.5 else (["AZ"] if fx_known else None),
                group_id=f"SG{int(rng.integers(0, groups))}" if grouped else None,
                group_size_reported=4 if grouped else None,
                has_payment_data=has_pay,
                reliability=reliability if has_pay else "none",
                n_edges=edges,
                known_counterparty_volume_share=known,
                industry_mix_out=mix,
                region="москва",
                revenue=float(10_000_000 * (1 + index % 7)),
            )
        )
    return rows


def synthetic_events(count: int, as_of: datetime) -> list[RankEvent]:
    published = as_of - timedelta(days=1)
    specs = (
        ("key_rate", "official", []),
        ("fx", "major_media", [{"text": "Китай", "type": "country"}]),
        ("commodities_gold", "major_media", [{"text": "золотодобыча", "type": "industry"}]),
        ("labor", "official", []),
        ("sanctions_trade", "industry", [{"text": "Китай", "type": "country"}]),
    )
    events = []
    for index in range(count):
        event_type, trust, entities = specs[index % len(specs)]
        events.append(
            RankEvent(
                event_id=f"B{index}",
                event_type=event_type,
                published_at=published,
                trust_tier=trust,
                entities=list(entities),
                cluster_id=f"B{index}",
                title=f"bench {event_type} {index}",
                text=f"Синтетическое событие {event_type} номер {index}",
                magnitude=0.4,
                n_sources=2,
                fact_signature=f"b{index}",
            )
        )
    return events


def _coverage(rows: list[ClientRow]) -> dict:
    counts = Counter(archetype(row) for row in rows)
    ordered = counts.most_common()
    total = max(len(rows), 1)
    curve = []
    for k in (1, 5, 10, 20, 50):
        share = sum(count for _, count in ordered[:k]) / total
        curve.append({"k": k, "share": share})
    return {"n_clients": len(rows), "n_archetypes": len(counts), "coverage_curve": curve}


def _cost(config, client_pairs: int, archetype_pairs: int) -> dict:
    tariffs = config.scoring["tariffs"]
    tokens = float(tariffs["interpretation_tokens_per_pair"])
    price = float(tariffs["price_per_1k_tokens_rub"])
    return {
        "tariff_status": tariffs["status"],
        "note": tariffs["note"],
        "interpretation_tokens_per_pair": tokens,
        "price_per_1k_tokens_rub": price,
        "event_client_pairs": client_pairs,
        "event_archetype_pairs": archetype_pairs,
        "per_client_rub": client_pairs * tokens / 1000.0 * price,
        "per_archetype_rub": archetype_pairs * tokens / 1000.0 * price,
    }


def full_base_available() -> bool:
    path = repo_path(load_ranking_config().scoring["paths"]["profiles"])
    return path.exists()


def run_bench(
    sizes: list[int],
    event_counts: list[int],
    seed: int,
    out: Path | None = None,
    *,
    include_full: bool = True,
) -> dict:
    config = load_ranking_config()
    as_of = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    runs = []
    largest_meta = None
    for size in sizes:
        rows = synthetic_rows(size, seed)
        events = synthetic_events(max(event_counts), as_of)
        cards = []
        tracemalloc.start()
        matrix = assemble_matrix(rows, config, as_of, cache_dir=None)
        embed_ms = float(matrix.embed_seconds) * 1000.0
        arch = {row.client_id: archetype(row) for row in matrix.rows}
        for count in event_counts:
            timings = {key: 0.0 for key in ("rules", "group", "graph", "embeddings", "aggregation")}
            client_pairs = 0
            arch_pairs: set[tuple] = set()
            started = tracemalloc.get_traced_memory()[0]
            import time

            t0 = time.perf_counter()
            for event in events[:count]:
                arrays = score_arrays(matrix, event, as_of, config, cards, cache_dir=None)
                for key in timings:
                    timings[key] += arrays.timings.get(key, 0.0)
                chosen = np.flatnonzero(arrays.above & arrays.evidence)
                client_pairs += int(chosen.size)
                for index in chosen.tolist():
                    arch_pairs.add((event.event_id, arch[str(matrix.client_ids[index])]))
            elapsed = time.perf_counter() - t0
            _current, peak = tracemalloc.get_traced_memory()
            runs.append(
                {
                    "source": "synthetic",
                    "n_clients": size,
                    "n_events": count,
                    "seed": seed,
                    "stages_ms": {key: (embed_ms if key == "embeddings" else 0.0) + timings[key] * 1000.0 for key in timings},
                    "elapsed_s": elapsed,
                    "memory_peak_mb": peak / (1024 * 1024),
                    "matrix_mb": float(matrix.embeddings.nbytes) / (1024 * 1024),
                    "pairs_above_threshold": client_pairs,
                    "event_client_pairs": client_pairs,
                    "event_archetype_pairs": len(arch_pairs),
                    "embedder_backend": matrix.embedder_backend,
                    "_alloc_start": started,
                }
            )
        tracemalloc.stop()
        largest_meta = {
            "archetypes": _coverage(rows),
            "cost": _cost(config, runs[-1]["event_client_pairs"], runs[-1]["event_archetype_pairs"]),
            "n_clients": size,
        }
    if include_full:
        if full_base_available():
            from alfa.ranking.matrix import load_profile_rows

            profiles = repo_path(config.scoring["paths"]["profiles"])
            graphs = repo_path(config.scoring["paths"]["graph_features"])
            rows = load_profile_rows(profiles, graphs, config)
            event = synthetic_events(1, as_of)[0]
            import time

            t0 = time.perf_counter()
            matrix = assemble_matrix(rows, config, as_of, cache_dir=None)
            arrays = score_arrays(matrix, event, as_of, config, [], cache_dir=None)
            elapsed = time.perf_counter() - t0
            runs.append(
                {
                    "source": "profiles.parquet",
                    "n_clients": len(matrix),
                    "n_events": 1,
                    "elapsed_s": elapsed,
                    "pairs_above_threshold": int((arrays.above & arrays.evidence).sum()),
                    "skipped": False,
                    "timeout_s": float(config.scoring["bench"]["full_base_timeout_s"]),
                    "within_timeout": elapsed < float(config.scoring["bench"]["full_base_timeout_s"]),
                }
            )
        else:
            runs.append(
                {
                    "source": "profiles.parquet",
                    "skipped": True,
                    "reason": "profiles.parquet absent",
                    "n_events": 1,
                }
            )
    for run in runs:
        run.pop("_alloc_start", None)
    payload = {
        "seed": seed,
        "tariff_status": "placeholder",
        "tariff_note": config.scoring["tariffs"]["note"],
        "embedder_backend": "hashing-non-bge",
        "runs": runs,
        "largest": largest_meta,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    config = load_ranking_config()
    bench = config.scoring["bench"]
    parser = argparse.ArgumentParser(description="Прогон масштаба ранжирования")
    parser.add_argument("--sizes", default=",".join(str(item) for item in bench["sizes"]))
    parser.add_argument("--events", default=",".join(str(item) for item in bench["event_counts"]))
    parser.add_argument("--seed", type=int, default=int(bench["seed"]))
    parser.add_argument("--out", default="reports/scale-bench/results.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-full", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        sizes = [200]
        events = [1]
        include_full = False
    else:
        sizes = [int(item) for item in str(args.sizes).split(",") if item]
        events = [int(item) for item in str(args.events).split(",") if item]
        include_full = not args.no_full
    payload = run_bench(sizes, events, args.seed, Path(args.out), include_full=include_full)
    print(args.out)
    print(json.dumps({"runs": len(payload["runs"]), "tariff_status": payload["tariff_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
