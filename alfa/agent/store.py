"""Checkpoint инсайта: состояние, ответы пользователя, обратная связь."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class InsightStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict[str, Any]] = {}
        self.by_trace: dict[str, str] = {}
        self.answers: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.question_count: dict[str, int] = {}
        self.session_usage: dict[str, dict[str, int]] = {}
        self.feedback: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        path = self.root / "store.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.records = payload.get("records") or {}
        self.by_trace = payload.get("by_trace") or {}
        self.question_count = payload.get("question_count") or {}
        self.session_usage = payload.get("session_usage") or {}
        self.feedback = payload.get("feedback") or []
        for item in payload.get("answers") or []:
            self.answers[(item["user_id"], item["client_id"], item["field"])] = item

    def save(self) -> None:
        payload = {
            "records": self.records,
            "by_trace": self.by_trace,
            "question_count": self.question_count,
            "session_usage": self.session_usage,
            "feedback": self.feedback,
            "answers": list(self.answers.values()),
        }
        (self.root / "store.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")

    def put(self, state: dict[str, Any], insight: dict[str, Any]) -> None:
        stored_state = json.loads(json.dumps(state, ensure_ascii=False, default=str))
        stored_state.pop("journal", None)
        record = self.records.get(insight["insight_id"]) or {"turns": 0}
        record.update(
            {
                "insight": insight,
                "state": stored_state,
                "journal": list(state.get("journal") or []),
                "draft_snapshot": stored_state.get("draft_snapshot"),
                "trace_id": insight["trace_id"],
                "client_id": insight["client_id"],
                "user_id": state.get("user_id"),
            }
        )
        self.records[insight["insight_id"]] = record
        self.by_trace[insight["trace_id"]] = insight["insight_id"]
        self.save()

    def get(self, insight_id: str) -> dict[str, Any] | None:
        return self.records.get(insight_id)

    def get_by_trace(self, trace_id: str) -> dict[str, Any] | None:
        insight_id = self.by_trace.get(trace_id)
        if insight_id is None:
            return None
        return self.records.get(insight_id)

    def answers_for(self, user_id: str, client_id: str) -> dict[str, dict[str, Any]]:
        return {
            field: item
            for (user, client, field), item in self.answers.items()
            if user == user_id and client == client_id
        }

    def remember_answer(self, user_id: str, client_id: str, field: str, value: str, answered_at: str) -> dict[str, Any]:
        item = {
            "user_id": user_id,
            "client_id": client_id,
            "field": field,
            "value": value,
            "answered_at": answered_at,
            "verified_by_bank": False,
        }
        self.answers[(user_id, client_id, field)] = item
        self.save()
        return item

    def add_question(self, user_id: str) -> None:
        self.question_count[user_id] = int(self.question_count.get(user_id) or 0) + 1
        self.save()

    def session_bucket(self, user_id: str) -> dict[str, int]:
        raw = self.session_usage.get(user_id) or {}
        return {"tokens": int(raw.get("tokens") or 0), "insights": int(raw.get("insights") or 0)}

    def add_session_usage(self, user_id: str, tokens: int) -> None:
        bucket = self.session_usage.setdefault(user_id, {"tokens": 0, "insights": 0})
        bucket["tokens"] = int(bucket.get("tokens") or 0) + max(0, int(tokens or 0))
        bucket["insights"] = int(bucket.get("insights") or 0) + 1
        self.save()

    def add_feedback(self, item: dict[str, Any]) -> None:
        self.feedback.append(item)
        self.save()
