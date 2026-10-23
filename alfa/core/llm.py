"""OpenAI-совместимый клиент. Ошибки схемы и сети не бросают исключение наружу.

Повтор: сеть, HTTP 5xx и 429, с экспоненциальной паузой. 4xx кроме 429 не повторяются.
Если задана response_schema и JSON невалиден, одна корректирующая попытка
(число берётся из llm.schema_retries) с текстом ошибки. После этого LLMResult(ok=False).
Кэш sqlite только для успешных ответов, ключ — хэш (model, messages, temperature, prompt_version).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from alfa.core.config import load_runtime_config, resolve_path
from alfa.core.logging import get_logger
from alfa.core.schemas import LLMResult
from alfa.core.trace import current_trace_id, log_step

logger = get_logger("alfa.llm")


class TransientHTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


def _cache_key(model: str, messages: list[dict[str, Any]], temperature: float, prompt_version: str | None) -> str:
    blob = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "prompt_version": prompt_version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ):
        self.config = config or load_runtime_config()
        self._sleeper = sleeper
        self._transport = transport
        self._lock = threading.Lock()
        self.call_count = 0
        llm = self.config["llm"]
        self.cache_enabled = bool(llm.get("cache_enabled", False))
        self.log_prompts = bool(llm.get("log_prompts", False))
        self._cache_path = resolve_path(self.config["paths"]["cache_db"])
        if self.cache_enabled:
            self._prepare_cache()

    def model_for_role(self, role: str) -> str:
        roles = self.config["llm"]["roles"]
        if role not in roles:
            raise KeyError(f"unknown llm role: {role}")
        return str(roles[role])

    def complete(
        self,
        *,
        role: str,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        trace_id: str | None = None,
        prompt_version: str | None = None,
    ) -> LLMResult:
        resolved = model or self.model_for_role(role)
        trace = trace_id or current_trace_id()
        key = _cache_key(resolved, messages, temperature, prompt_version)
        started = time.perf_counter()

        if self.cache_enabled:
            cached = self._cache_get(key)
            if cached is not None:
                parsed, error = self._parse(cached["content"], response_schema)
                if error is None:
                    result = LLMResult(
                        ok=True,
                        content=cached["content"],
                        parsed=parsed,
                        cache_hit=True,
                        tokens_in=cached["tokens_in"],
                        tokens_out=cached["tokens_out"],
                        latency_ms=(time.perf_counter() - started) * 1000,
                        model=resolved,
                        raw=cached["content"],
                    )
                    self._journal(
                        trace, role, resolved, prompt_version, key, result, messages, cached["content"]
                    )
                    return result

        working_messages = list(messages)
        schema_tries = int(self.config["llm"].get("schema_retries", 1)) if response_schema else 0
        last_error = "empty"
        content: str | None = None
        tokens_in = tokens_out = None
        for schema_attempt in range(schema_tries + 1):
            call = self._call_with_retry(resolved, working_messages, temperature)
            tokens_in = call.tokens_in if call.tokens_in is not None else tokens_in
            tokens_out = call.tokens_out if call.tokens_out is not None else tokens_out
            if not call.ok:
                result = call.model_copy(
                    update={"latency_ms": (time.perf_counter() - started) * 1000, "model": resolved}
                )
                self._journal(trace, role, resolved, prompt_version, key, result, working_messages, call.content)
                return result
            content = call.content or ""
            parsed, last_error = self._parse(content, response_schema)
            if last_error is None:
                if self.cache_enabled:
                    self._cache_put(key, content, call.tokens_in, call.tokens_out, resolved)
                result = LLMResult(
                    ok=True,
                    content=content,
                    parsed=parsed,
                    cache_hit=False,
                    tokens_in=call.tokens_in,
                    tokens_out=call.tokens_out,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    model=resolved,
                    raw=content,
                )
                self._journal(trace, role, resolved, prompt_version, key, result, working_messages, content)
                return result
            if schema_attempt >= schema_tries:
                break
            working_messages = working_messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Ответ не прошёл проверку JSON-схемы. Верни только исправленный JSON. "
                        f"Ошибка: {last_error}"
                    ),
                },
            ]

        result = LLMResult(
            ok=False,
            content=content,
            error=last_error,
            cache_hit=False,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=resolved,
            raw=content,
        )
        self._journal(trace, role, resolved, prompt_version, key, result, working_messages, content)
        return result

    def _endpoint(self) -> str:
        base = str(self.config["llm"]["base_url"]).rstrip("/")
        return f"{base}/chat/completions"

    def _api_key(self) -> str | None:
        import os

        env_name = str(self.config["llm"].get("api_key_env") or "ALFA_LLM_API_KEY")
        value = os.environ.get(env_name)
        if value:
            return value
        configured = self.config["llm"].get("api_key")
        return str(configured) if configured else None

    def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
    ) -> LLMResult:
        key_name = str(self.config["llm"].get("api_key_env") or "ALFA_LLM_API_KEY")
        api_key = self._api_key()
        if not api_key:
            return LLMResult(ok=False, error=f"missing env {key_name}", model=model)
        max_attempts = int(self.config["llm"].get("max_attempts", 3))
        timeout = float(self.config["llm"].get("timeout_s", 60))
        backoff = float(self.config["llm"].get("backoff_s", 0.4))
        payload = {"model": model, "messages": messages, "temperature": temperature}
        options = self.config["llm"].get("options")
        if options:
            payload["options"] = options
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        last_error = "request failed"
        for attempt in range(max_attempts):
            try:
                with httpx.Client(timeout=timeout, transport=self._transport) as client:
                    self.call_count += 1
                    response = client.post(self._endpoint(), json=payload, headers=headers)
                if response.status_code == 429 or response.status_code >= 500:
                    raise TransientHTTPError(response.status_code, response.text[:500])
                if response.status_code >= 400:
                    return LLMResult(
                        ok=False,
                        error=f"HTTP {response.status_code}: {response.text[:500]}",
                        model=model,
                        raw=response.text,
                    )
                body = response.json()
                message = body["choices"][0]["message"]["content"]
                if isinstance(message, list):
                    message = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part) for part in message
                    )
                usage = body.get("usage") or {}
                return LLMResult(
                    ok=True,
                    content=str(message),
                    raw=str(message),
                    model=model,
                    tokens_in=usage.get("prompt_tokens"),
                    tokens_out=usage.get("completion_tokens"),
                )
            except TransientHTTPError as exc:
                last_error = str(exc)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < max_attempts:
                self._sleeper(backoff * (2**attempt))
        return LLMResult(ok=False, error=last_error, model=model)

    @staticmethod
    def _parse(content: str, response_schema: type[BaseModel] | None) -> tuple[Any, str | None]:
        if response_schema is None:
            return content, None
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"JSONDecodeError: {exc}"
        try:
            model = response_schema.model_validate(data)
        except ValidationError as exc:
            return None, exc.errors().__str__()
        return model, None

    def _prepare_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    model TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._cache_path, check_same_thread=False)
        return conn

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT content, tokens_in, tokens_out FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {"content": row[0], "tokens_in": row[1], "tokens_out": row[2]}

    def _cache_put(
        self,
        key: str,
        content: str,
        tokens_in: int | None,
        tokens_out: int | None,
        model: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_cache (key, content, tokens_in, tokens_out, model)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    content = excluded.content,
                    tokens_in = excluded.tokens_in,
                    tokens_out = excluded.tokens_out,
                    model = excluded.model
                """,
                (key, content, tokens_in, tokens_out, model),
            )

    def _journal(
        self,
        trace_id: str | None,
        role: str,
        model: str,
        prompt_version: str | None,
        input_hash: str,
        result: LLMResult,
        messages: list[dict[str, Any]],
        response: str | None,
    ) -> None:
        log_step(
            "llm.complete",
            trace_id=trace_id,
            role=role,
            model=model,
            prompt_version=prompt_version,
            input_hash=input_hash,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            ok=result.ok,
            error=result.error,
            cache_hit=result.cache_hit,
            log_prompts=self.log_prompts,
            prompt=messages if self.log_prompts else None,
            response=response if self.log_prompts else None,
            journal_path=resolve_path(self.config["paths"]["journal"]),
            prompt_log_path=resolve_path(self.config["paths"]["prompt_log"]),
        )
        if not result.ok:
            logger.info("llm %s failed: %s", role, result.error)
