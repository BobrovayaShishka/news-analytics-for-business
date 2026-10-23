"""Журнал шагов. Тот же jsonl сможет читать агент дня 4.

trace_id живёт в contextvar. Тела промптов пишутся только если log_prompts=true,
и в отдельный файл, а не в строку журнала.
"""

from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alfa.core.config import load_runtime_config, resolve_path

trace_id_var: ContextVar[str | None] = ContextVar("alfa_trace_id", default=None)
_LOCK = threading.Lock()


def set_trace_id(trace_id: str | None) -> None:
    trace_id_var.set(trace_id)


def current_trace_id() -> str | None:
    return trace_id_var.get()


def journal_path() -> Path:
    return resolve_path(load_runtime_config()["paths"]["journal"])


def prompt_log_path() -> Path:
    return resolve_path(load_runtime_config()["paths"]["prompt_log"])


def log_step(step: str, /, **fields: Any) -> None:
    journal = fields.pop("journal_path", None)
    prompt_log = fields.pop("prompt_log_path", None)
    log_prompts = bool(fields.pop("log_prompts", False))
    prompt = fields.pop("prompt", None)
    response = fields.pop("response", None)
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": fields.pop("trace_id", None) or current_trace_id(),
        "step": step,
    }
    payload.update(fields)
    path = Path(journal) if journal else journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if log_prompts and (prompt is not None or response is not None):
            body_path = Path(prompt_log) if prompt_log else prompt_log_path()
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body = {
                "ts": payload["ts"],
                "trace_id": payload["trace_id"],
                "step": step,
                "input_hash": payload.get("input_hash"),
                "prompt": prompt,
                "response": response,
            }
            with body_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(body, ensure_ascii=False, default=str) + "\n")
