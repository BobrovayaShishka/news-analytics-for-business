"""Доступ к профилю и графу только в рамках сессионного client_id."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alfa.schemas import GraphFeatures, Profile
from alfa.settings import load_profile_config, output_path

_session: ContextVar[str | None] = ContextVar("alfa_session_client", default=None)
_profiles: dict[str, dict] | None = None
_graphs: dict[str, dict] | None = None
_trace_path: Path | None = None


class AccessError(PermissionError):
    """client_id не совпадает с сессией."""


@contextmanager
def session_scope(client_id: str):
    token = _session.set(client_id)
    try:
        yield
    finally:
        _session.reset(token)


def reload() -> None:
    global _profiles, _graphs, _trace_path
    _profiles = None
    _graphs = None
    _trace_path = None


def _access_trace_path() -> Path:
    global _trace_path
    if _trace_path is None:
        _trace_path = output_path(load_profile_config(), "trace")
    return _trace_path


def _load() -> tuple[dict[str, dict], dict[str, dict]]:
    global _profiles, _graphs
    if _profiles is None or _graphs is None:
        config = load_profile_config()
        profiles = pd.read_parquet(output_path(config, "profiles"))
        graphs = pd.read_parquet(output_path(config, "graph_features"))
        _profiles = {row.client_id: json.loads(row.payload) for row in profiles.itertuples(index=False)}
        _graphs = {row.client_id: json.loads(row.payload) for row in graphs.itertuples(index=False)}
    return _profiles, _graphs


def _log(kind: str, client_id: str, started: float, version: str | None, extra: dict | None, error: str | None) -> None:
    path = _access_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trace_id": uuid.uuid4().hex,
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "arguments": {"client_id": client_id},
        "session_client_id": _session.get(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "profile_version": version,
        "error": error,
        "extra": extra or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _guard(client_id: str) -> None:
    current = _session.get()
    if not current or current != client_id:
        raise AccessError("client_id берётся из сессии и не совпадает с аргументом")


def _empty_graph(client_id: str, version: str) -> dict:
    return {
        "client_id": client_id,
        "profile_version": version,
        "has_payment_data": False,
        "missing_data": ["no_payment_data"],
    }


def traced(kind: str):
    def decorator(fn):
        def wrapper(client_id: str):
            started = time.perf_counter()
            state: dict = {"version": None}
            try:
                result, extra = fn(client_id, state)
                _log(kind, client_id, started, state["version"], extra, None)
                return result
            except Exception as exc:
                _log(kind, client_id, started, state["version"], None, type(exc).__name__)
                raise

        wrapper.__name__ = kind
        return wrapper

    return decorator


@traced("get_profile")
def get_profile(client_id: str, state: dict) -> tuple[Profile, dict]:
    _guard(client_id)
    profiles, _ = _load()
    payload = profiles[client_id]
    state["version"] = payload["profile_version"]
    return Profile.model_validate(payload), {}


@traced("get_graph_features")
def get_graph_features(client_id: str, state: dict) -> tuple[GraphFeatures, dict]:
    _guard(client_id)
    profiles, graphs = _load()
    if client_id not in profiles:
        raise KeyError(client_id)
    payload = graphs.get(client_id)
    if payload is None:
        payload = _empty_graph(client_id, profiles[client_id]["profile_version"])
        extra: dict = {}
    else:
        payload = dict(payload)
        extra = {
            "top_in": payload.pop("trace_top_in", []),
            "top_out": payload.pop("trace_top_out", []),
        }
    state["version"] = payload["profile_version"]
    return GraphFeatures.model_validate(payload), extra
