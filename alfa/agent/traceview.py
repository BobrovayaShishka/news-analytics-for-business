"""Трасса, по которой восстанавливаются источники, признаки и проверка."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def digest(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_trace_view(journal: list[dict[str, Any]], trace_id: str | None = None) -> dict[str, Any]:
    sources: list[str] = []
    features: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    plan = None
    repairs: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for event in journal:
        kind = event.get("type")
        if kind == "tool_call":
            for source_id in event.get("source_ids") or []:
                if source_id not in sources:
                    sources.append(source_id)
            if event.get("tool") == "get_profile_view" and isinstance(event.get("data_summary"), dict):
                features = dict(event["data_summary"])
        elif kind == "decision" and event.get("kind") == "plan_selected":
            plan = event.get("plan")
        elif kind == "decision" and event.get("kind") == "plan_rejected":
            rejected.append(event)
        elif kind == "decision" and event.get("kind") == "verification":
            verification = {
                "decision": event.get("decision"),
                "iterations": event.get("iterations"),
                "stage1": event.get("stage1"),
                "stage2": event.get("stage2"),
            }
        elif kind == "decision" and event.get("kind") == "repair":
            repairs.append(event)
        elif kind == "user_question":
            questions.append(event)
    resolved = trace_id or next((event.get("trace_id") for event in journal if event.get("trace_id")), "")
    return {
        "trace_id": resolved,
        "events": list(journal),
        "sources": sources,
        "client_features": features,
        "verification": verification,
        "plan": plan,
        "repairs": repairs,
        "questions": questions,
        "rejected_plans": rejected,
    }
