"""Метрики ленты на фикстурной разметке. annotator_a согласован с правилами, annotator_b — вторая колонка."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from alfa.access import session_scope
from alfa.ranking.catalog import AS_OF, catalog_cards, catalog_events, catalog_rows
from alfa.ranking.config import load_ranking_config
from alfa.ranking.service import RankingService

LABELS_PATH = Path(__file__).resolve().parent / "fixtures" / "labels.jsonl"
MUST_PATH = Path(__file__).resolve().parent / "fixtures" / "must_show.jsonl"

STAGES = (
    ("embeddings", {"sem"}),
    ("sensitivity", {"sem", "sens"}),
    ("entities", {"sem", "sens", "entity"}),
    ("group", {"sem", "sens", "entity", "group"}),
    ("graph", {"sem", "sens", "entity", "group", "graph"}),
    ("analytics", {"sem", "sens", "entity", "group", "graph", "analytics"}),
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _dcg(grades: list[int]) -> float:
    total = 0.0
    for index, grade in enumerate(grades):
        total += (2 ** grade - 1) / math.log2(index + 2)
    return total


def _ndcg(ranked_grades: list[int], k: int) -> float | None:
    ideal = sorted(ranked_grades, reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return None
    return _dcg(ranked_grades[:k]) / idcg


def _scores(service: RankingService, enabled: set[str] | None = None) -> dict[tuple[str, str], float]:
    table: dict[tuple[str, str], float] = {}
    for event in service.events:
        for pair in service.score_event(event.event_id, AS_OF, enabled=enabled):
            table[(pair.client_id, pair.event_id)] = pair.score
    return table


def _ndcg_macro(labels: list[dict], scores: dict[tuple[str, str], float], clients: set[str] | None = None) -> float | None:
    by_client: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        if clients is not None and row["client_id"] not in clients:
            continue
        by_client[row["client_id"]].append(row)
    values = []
    for client_id, rows in by_client.items():
        ordered = sorted(rows, key=lambda item: (-scores.get((client_id, item["event_id"]), 0.0), item["event_id"]))
        grades = [int(item["annotator_a"]) for item in ordered]
        score = _ndcg(grades, 5)
        if score is not None:
            values.append(score)
    if not values:
        return None
    return sum(values) / len(values)


def run_metrics(output: Path | None = None, service: RankingService | None = None) -> dict:
    service = service or RankingService(catalog_rows(), catalog_events(), catalog_cards(), load_ranking_config())
    labels = _read_jsonl(LABELS_PATH)
    must = _read_jsonl(MUST_PATH)
    scores = _scores(service)
    agreement = sum(int(row["annotator_a"]) == int(row["annotator_b"]) for row in labels) / len(labels)
    disagreements = [
        {
            "client_id": row["client_id"],
            "event_id": row["event_id"],
            "annotator_a": row["annotator_a"],
            "annotator_b": row["annotator_b"],
            "note": row.get("note", ""),
        }
        for row in labels
        if int(row["annotator_a"]) != int(row["annotator_b"])
    ]
    labeled = {(row["client_id"], row["event_id"]): int(row["annotator_a"]) for row in labels}
    shown = [(key, grade) for key, grade in labeled.items() if scores.get(key, 0.0) > 0 and _above(service, key)]
    shown_zero = sum(grade == 0 for _, grade in shown)
    precision = None
    if shown:
        precision = sum(grade > 0 for _, grade in shown) / len(shown)
    per_client_precision = []
    by_client_shown: dict[str, list[int]] = defaultdict(list)
    for (client_id, event_id), grade in labeled.items():
        if _above(service, (client_id, event_id)):
            by_client_shown[client_id].append(grade)
    for grades in by_client_shown.values():
        head = grades[:5]
        if head:
            per_client_precision.append(sum(item > 0 for item in head) / len(head))
    must_hits = 0
    for row in must:
        with session_scope(row["client_id"]):
            feed = service.build_feed(row["client_id"], AS_OF, record_impressions=False)
        if any(item.event_id == row["event_id"] for item in feed.items):
            must_hits += 1
    clients = sorted({row["client_id"] for row in labels})
    empty = 0
    for client_id in clients:
        with session_scope(client_id):
            feed = service.build_feed(client_id, AS_OF, record_impressions=False)
        if feed.empty_state:
            empty += 1
    no_payments = {
        row.client_id
        for row in service.rows
        if row.has_payment_data is False
    }
    slices = _slices(service, labels, scores)
    ablation = []
    for name, blocks in STAGES:
        ablation.append(
            {
                "stage": name,
                "blocks": sorted(blocks),
                "ndcg_at_5": _ndcg_macro(labels, _scores(service, blocks)),
                "rerank": False,
            }
        )
    payload = {
        "n_labels": len(labels),
        "agreement_match_rate": agreement,
        "disagreements": disagreements,
        "ndcg_at_5": _ndcg_macro(labels, scores),
        "precision_at_threshold": precision,
        "precision_at_5": (sum(per_client_precision) / len(per_client_precision)) if per_client_precision else None,
        "shown_label0_share": (shown_zero / len(shown)) if shown else 0.0,
        "shown_labeled": len(shown),
        "must_show_recall": must_hits / len(must) if must else None,
        "must_show_total": len(must),
        "empty_feed_share": empty / len(clients) if clients else None,
        "slices": slices,
        "without_payments": {
            "n_clients": len(no_payments),
            "ndcg_at_5": _ndcg_macro(labels, scores, no_payments),
        },
        "ablation": ablation,
        "rerank_enabled": False,
        "label_source": "rule-consistent fixture, not jury double annotation",
        "config_version": service.config.config_version,
        "weight_version": service.config.weight_version,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _above(service: RankingService, key: tuple[str, str]) -> bool:
    client_id, event_id = key
    for pair in service.score_event(event_id, AS_OF):
        if pair.client_id == client_id:
            return pair.above_threshold
    return False


def _slices(service: RankingService, labels: list[dict], scores: dict[tuple[str, str], float]) -> dict:
    rows = {row.client_id: row for row in service.rows}
    buckets: dict[str, list[int]] = defaultdict(list)

    def add(name: str, grade: int) -> None:
        buckets[name].append(grade)

    for (client_id, event_id), grade in (( (row["client_id"], row["event_id"]), int(row["annotator_a"])) for row in labels):
        if not _above(service, (client_id, event_id)):
            continue
        client = rows[client_id]
        event = service.event_map()[event_id]
        add(f"event:{event.event_type}", grade)
        add("group" if client.group_id else "no_group", grade)
        add("payments" if client.has_payment_data else "no_payments", grade)
        pair = next(item for item in service.score_event(event_id, AS_OF) if item.client_id == client_id)
        add(f"reliability:{pair.reliability}", grade)
    return {
        name: {"n": len(grades), "precision": sum(item > 0 for item in grades) / len(grades)}
        for name, grades in sorted(buckets.items())
    }


def main() -> None:
    path = Path("reports/ranking-eval/metrics.json")
    payload = run_metrics(path)
    print(json.dumps({key: payload[key] for key in ("ndcg_at_5", "must_show_recall", "shown_label0_share", "agreement_match_rate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
