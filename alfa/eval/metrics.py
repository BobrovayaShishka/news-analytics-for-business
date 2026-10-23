"""Метрики извлечения и поиска. Цифры только из переданных предсказаний и разметки."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from alfa.core.textcheck import _canonical_number, normalize_ws


def _canon_value(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    pieces = []
    for match in re.finditer(r"-?\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?", normalize_ws(str(value))):
        canon = _canonical_number(match.group(0))
        if canon is not None:
            pieces.append(canon)
    if not pieces:
        return normalize_ws(str(value)).lower()
    return "|".join(pieces)


def _canon_unit(unit: str | None) -> str:
    if not unit:
        return ""
    return normalize_ws(unit).lower().replace(" ", "")


def _canon_date(value: str | None) -> str:
    if not value:
        return ""
    text = normalize_ws(str(value))
    iso = re.search(r"(20\d{2})-(\d{2})-(\d{2})", text)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
    dotted = re.search(r"(\d{1,2})[./](\d{1,2})[./](20\d{2})", text)
    if dotted:
        return f"{dotted.group(3)}-{int(dotted.group(2)):02d}-{int(dotted.group(1)):02d}"
    if re.fullmatch(r"20\d{2}", text):
        return text
    return text


def fact_key(fact: dict) -> tuple[str | None, str, str]:
    return (_canon_value(fact.get("value")), _canon_unit(fact.get("unit")), _canon_date(fact.get("date")))


def _prf(gold: list[str], pred: list[str]) -> dict[str, float | None]:
    labels = sorted(set(gold) | set(pred))
    f1s: list[float] = []
    matrix: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    for truth, guess in zip(gold, pred, strict=True):
        matrix.setdefault(truth, defaultdict(int))
        matrix[truth][guess] += 1
    accuracy = (sum(left == right for left, right in zip(gold, pred, strict=True)) / len(gold)) if gold else None
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(row.get(label, 0) for name, row in matrix.items() if name != label)
        fn = sum(count for guess, count in matrix.get(label, {}).items() if guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {
        "accuracy": accuracy,
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else None,
        "confusion": {truth: dict(row) for truth, row in matrix.items()},
    }


def relevance_scores(gold: list[bool], pred: list[bool]) -> dict[str, float | None]:
    tp = sum(left and right for left, right in zip(gold, pred, strict=True))
    fp = sum((not left) and right for left, right in zip(gold, pred, strict=True))
    fn = sum(left and (not right) for left, right in zip(gold, pred, strict=True))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def fact_scores(pairs: list[tuple[list[dict], list[dict]]]) -> dict[str, float | None]:
    matched = 0
    gold_n = 0
    pred_n = 0
    checked = 0
    checked_ok = 0
    for gold_facts, pred_facts in pairs:
        gold_keys = [fact_key(item) for item in gold_facts]
        pred_keys = [fact_key(item) for item in pred_facts]
        gold_n += len(gold_keys)
        pred_n += len(pred_keys)
        remaining = list(gold_keys)
        for key in pred_keys:
            if key in remaining:
                remaining.remove(key)
                matched += 1
        for item in pred_facts:
            checked += 1
            if not item.get("unverified", False):
                checked_ok += 1
    return {
        "precision": (matched / pred_n) if pred_n else None,
        "recall": (matched / gold_n) if gold_n else None,
        "textcheck_pass_share": (checked_ok / checked) if checked else None,
    }


def extraction_report(rows: list[dict]) -> dict:
    """rows: gold event_type, is_business_relevant, facts; pred those plus extraction_status, unverified."""
    scored = [row for row in rows if row.get("pred_status") in {"ok", "partial"} and row.get("pred_event_type")]
    event = _prf(
        [row["event_type"] for row in scored],
        [row["pred_event_type"] for row in scored],
    ) if scored else {"accuracy": None, "macro_f1": None, "confusion": {}}
    relevance = relevance_scores(
        [bool(row["is_business_relevant"]) for row in scored],
        [bool(row.get("pred_relevant")) for row in scored],
    ) if scored else {"precision": None, "recall": None}
    facts = fact_scores([(row.get("facts") or [], row.get("pred_facts") or []) for row in scored])
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("pred_status") or "missing")
        status_counts[status] = status_counts.get(status, 0) + 1
    valid = sum(1 for row in rows if row.get("pred_status") in {"ok", "partial"})
    return {
        "n": len(rows),
        "valid_json_share": (valid / len(rows)) if rows else None,
        "status_counts": status_counts,
        "event_type": event,
        "relevance": relevance,
        "facts": facts,
    }


def recall_at_k(relevant: list[str], ranked: list[str], k: int) -> float | None:
    if not relevant:
        return None
    found = set(ranked[:k]) & set(relevant)
    return len(found) / len(relevant)


def mrr(relevant: list[str], ranked: list[str]) -> float:
    wanted = set(relevant)
    for index, item in enumerate(ranked, start=1):
        if item in wanted:
            return 1.0 / index
    return 0.0


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def search_metrics(query_rows: list[dict]) -> dict:
    """query_rows item: mode, collection, noise, relevant ids, ranked ids."""
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in query_rows:
        key = (row["mode"], row["collection"], row["noise"])
        grouped.setdefault(key, []).append(row)
    report: dict[str, dict] = {}
    for (mode, collection, noise), rows in grouped.items():
        bucket = report.setdefault(mode, {}).setdefault(collection, {})
        bucket[noise] = {
            "recall@1": mean([recall_at_k(row["relevant"], row["ranked"], 1) for row in rows]),
            "recall@3": mean([recall_at_k(row["relevant"], row["ranked"], 3) for row in rows]),
            "recall@5": mean([recall_at_k(row["relevant"], row["ranked"], 5) for row in rows]),
            "mrr": mean([mrr(row["relevant"], row["ranked"]) for row in rows]),
            "n": len(rows),
        }
    return report


def utc_now() -> datetime:
    return datetime.now().astimezone()
