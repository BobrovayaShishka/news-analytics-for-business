"""Инструменты только на чтение. client_id берётся из сессии, не из аргумента модели."""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Any

from alfa.access import session_scope
from alfa.agent.config import AgentConfig
from alfa.agent.models import ToolResult
from alfa.agent.security import detect_injection, mask_args, redact_tree, wrap_data
from alfa.core.trace import log_step
from alfa.ranking.engine import age_days, as_utc
from alfa.settings import load_profile_config

_CARD_TEXT = {
    "card-key": "11 сентября 2026 года ключевая ставка сохранена на уровне 14%.",
    "card-fx": "19 октября 2026 года официальный курс составил 82 рубля за доллар.",
    "card-gold": "10 октября 2026 года котировка золота составила 4400 долларов за унцию.",
    "card-labor": "15 октября 2026 года ориентир по зарплатам обновлён.",
    "card-infl": "1 октября 2026 года обзор фиксирует инфляционный фон.",
    "card-tax": "12 октября 2026 года налоговый режим остаётся в фокусе обзора.",
    "card-demand": "11 октября 2026 года оборот розницы остаётся повышенным.",
    "card-sanc": "16 октября 2026 года ограничения внешней торговли сохраняются.",
}
_CARD_META = {
    "card-key": ("policy-rate-20260911", 1, [0.08, 0.1, 0.92, 0.42]),
    "card-fx": ("fx-20261019", 1, [0.08, 0.1, 0.92, 0.42]),
    "card-gold": ("gold-aug-2026", 1, [0.08, 0.1, 0.92, 0.42]),
    "card-labor": ("labor-20261015", 1, [0.08, 0.1, 0.92, 0.42]),
    "card-infl": ("infl-weekly-20260831", 1, [0.08, 0.1, 0.92, 0.42]),
    "card-tax": ("tax-20261012", 1, [0.08, 0.1, 0.92, 0.42]),
    "card-demand": ("consumption-20260805", 1, [0.08, 0.1, 0.92, 0.42]),
    "card-sanc": ("sanctions-20261016", 1, [0.08, 0.1, 0.92, 0.42]),
}
_FEATURE_KEYS = (
    "has_credit",
    "has_leasing",
    "has_deposit",
    "has_guarantees",
    "has_factoring",
    "has_payroll",
    "has_rko",
    "industry_group",
    "themes",
    "size_class",
    "workers_class",
    "revenue",
    "fx_exposure",
    "fx_confidence",
    "fts_countries",
    "reliability",
    "has_payment_data",
    "known_counterparty_volume_share",
    "key_owner_known",
    "group_size",
    "region",
)
_DECIMAL = re.compile(r"\d+[,.]\d+")


def _row(ranking: Any, client_id: str):
    for row in ranking.rows:
        if row.client_id == client_id:
            return row
    return None


def _split_sentences(raw: str) -> list[str]:
    protected: list[str] = []

    def hide(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\u0000{len(protected) - 1}\u0000"

    shielded = _DECIMAL.sub(hide, raw or "")
    parts = re.split(r"[.!?]+", shielded)
    output: list[str] = []
    for part in parts:
        text = part
        for index, token in enumerate(protected):
            text = text.replace(f"\u0000{index}\u0000", token)
        text = text.strip()
        if text:
            output.append(text)
    return output


def _as_texts(value: Any) -> list[str]:
    if not value:
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("claim") or "").strip()
        else:
            text = str(getattr(item, "text", "") or "").strip()
        if text:
            output.append(text)
    return output


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


class Toolkit:
    def __init__(self, ranking: Any, config: AgentConfig, journal_path, clock: Callable[[], datetime] | None = None):
        self.ranking = ranking
        self.config = config
        self.journal_path = journal_path
        self.clock = clock or (lambda: datetime.now())
        self.calls: Counter[str] = Counter()

    def reset(self) -> None:
        self.calls = Counter()

    def call(self, name: str, args: dict[str, Any] | None, state: dict[str, Any]) -> tuple[ToolResult, dict[str, Any]]:
        started = time.perf_counter()
        payload_args = dict(args or {})
        event = self._event(name, payload_args, state, ok=False, error="error", source_ids=[], summary=None, redactions=[])
        try:
            result = self._dispatch(name, payload_args, state)
        except Exception as exc:  # инструмент не роняет граф
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        result.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        cleaned, hits = redact_tree(result.data)
        result = result.model_copy(update={"data": cleaned, "redactions": list(result.redactions) + hits})
        event = self._event(
            name,
            payload_args,
            state,
            ok=result.ok,
            error=result.error,
            source_ids=result.source_ids,
            summary=self._summary(name, result),
            redactions=result.redactions,
            latency_ms=result.latency_ms,
        )
        log_step(
            "agent.tool",
            trace_id=state.get("trace_id"),
            tool=name,
            status="ok" if result.ok else result.error,
            journal_path=self.journal_path,
        )
        return result, event

    def _event(self, name, args, state, *, ok, error, source_ids, summary, redactions, latency_ms=0.0) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "tool": name,
            "args": mask_args(args),
            "status": "ok" if ok else (error or "error"),
            "source_ids": list(source_ids or []),
            "result_bytes": len(str(summary)) if summary is not None else 0,
            "data_summary": summary if name == "get_profile_view" and ok else None,
            "sources_detail": summary if name != "get_profile_view" and isinstance(summary, list) else [],
            "redactions": list(redactions or []),
            "latency_ms": latency_ms,
            "trace_id": state.get("trace_id"),
        }

    def _summary(self, name: str, result: ToolResult):
        if not result.ok:
            return None
        if name == "get_profile_view":
            features = result.data.get("features") or {}
            from alfa.agent.interpret import _closed_ok, _public_features

            if not _closed_ok(self.config):
                return _public_features(features, self.config)
            return features
        details = []
        for source_id in result.source_ids:
            details.append({"id": source_id})
        return details

    def _deny(self, state: dict[str, Any], args: dict[str, Any]) -> ToolResult | None:
        supplied = args.pop("client_id", None)
        client_id = str(state.get("client_id") or "")
        allowed = set(state.get("allowed_company_ids") or [])
        if not client_id or client_id not in allowed:
            return ToolResult(ok=False, error="access_denied")
        if supplied is not None:
            return ToolResult(ok=False, error="access_denied")
        return None

    def _dispatch(self, name: str, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        if name not in {
            "get_profile_view",
            "get_group_context",
            "get_counterparties",
            "get_event",
            "search_alfa_analytics",
            "search_news",
            "external_stats",
            "calc",
            "explain_relevance",
        }:
            return ToolResult(ok=False, error="not_whitelisted")
        denied = self._deny(state, args)
        if denied is not None:
            return denied
        limit = int(self.config.max_calls.get(name) or 1)
        if self.calls[name] >= limit:
            return ToolResult(ok=False, error="call_limit")
        self.calls[name] += 1
        handler = getattr(self, name)
        return self._invoke_handler(handler, args, state)

    def _invoke_handler(self, handler, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        timeout = float(self.config.tool_timeout_s or 0)
        if timeout <= 0:
            return handler(args, state)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(handler, args, state)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            return ToolResult(ok=False, error="timeout")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def get_profile_view(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        row = _row(self.ranking, state["client_id"])
        if row is None:
            return ToolResult(ok=False, error="client_not_found")
        features = {key: _jsonable(getattr(row, key, None)) for key in _FEATURE_KEYS}
        missing = [key for key, value in features.items() if value is None]
        structural = list(load_profile_config()["structurally_missing"])
        data = {
            "features": features,
            "missing_fields": missing,
            "structurally_missing": structural,
            "fx": {
                "exposure": features.get("fx_exposure") or "unknown",
                "confidence": features.get("fx_confidence") or "none",
                "countries": list(features.get("fts_countries") or []),
            },
            "key_owner_known": features.get("key_owner_known"),
            "reliability": str(features.get("reliability") or "none"),
        }
        return ToolResult(ok=True, data=data, source_ids=["profile:features"])

    def get_group_context(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        row = _row(self.ranking, state["client_id"])
        if row is None or not row.group_id:
            return ToolResult(
                ok=True,
                data={
                    "in_group": False,
                    "industries": [],
                    "flags": {},
                    "size_classes": [],
                    "reliability": str((row.reliability if row else None) or "none"),
                },
                source_ids=["group:none"],
            )
        members = [item for item in self.ranking.rows if item.group_id == row.group_id]
        industries = sorted({item.industry_group for item in members if item.industry_group})
        sizes = sorted({item.size_class for item in members if item.size_class})
        flags = {}
        for name in ("has_credit", "has_leasing", "has_deposit", "has_payroll", "has_rko"):
            flags[name] = sum(1 for item in members if getattr(item, name, None) is True)
        data = {
            "in_group": True,
            "group_size": len(members),
            "industries": industries,
            "flags": flags,
            "size_classes": sizes,
            "reliability": str(row.reliability or "none"),
        }
        return ToolResult(ok=True, data=data, source_ids=["group:aggregate"])

    def get_counterparties(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        row = _row(self.ranking, state["client_id"])
        if row is None or not row.has_payment_data:
            return ToolResult(
                ok=True,
                data={"counterparties": [], "known_counterparty_volume_share": None, "reliability": "none"},
                source_ids=["graph:none"],
            )
        mix = dict(row.industry_mix_out or {})
        ordered = sorted(mix.items(), key=lambda item: item[1], reverse=True)[:5]
        slots = []
        for index, (industry, share) in enumerate(ordered, start=1):
            slots.append(
                {
                    "slot": f"counterparty_{index}",
                    "volume_share": round(float(share), 4),
                    "industry_group": industry,
                    "region": None,
                    "fts_flag": None,
                    "reliability": row.reliability or "none",
                }
            )
        data = {
            "counterparties": slots,
            "known_counterparty_volume_share": row.known_counterparty_volume_share,
            "reliability": row.reliability or "none",
        }
        return ToolResult(ok=True, data=data, source_ids=["graph:counterparties"])

    def get_event(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        event = self.ranking.event_map().get(state["event_id"])
        if event is None:
            return ToolResult(ok=False, error="event_not_found")
        raw = event.text or event.title or ""
        suspicious = detect_injection(raw)
        kept = [part for part in _split_sentences(raw) if not detect_injection(part)]
        claim = ". ".join(kept).strip() or (event.title or "Событие получено")
        if claim and not claim.endswith("."):
            claim += "."
        claim, _hits = redact_tree(claim)
        entities = [_jsonable(item) for item in (event.entities or [])]
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "title": event.title,
            "text": wrap_data(raw),
            "facts": [{"id": "f1", "claim": claim, "evidence_span": claim}],
            "entities": entities,
            "risks": _as_texts(getattr(event, "risks", None)),
            "opportunities": _as_texts(getattr(event, "opportunities", None)),
            "cluster_id": event.cluster_id,
            "echo_sources": self._echoes(event),
            "suspicious_instructions": suspicious,
            "published_at": as_utc(event.published_at).isoformat(),
        }
        return ToolResult(ok=True, data=data, source_ids=["fact:f1"])

    def _echoes(self, event) -> list[dict[str, Any]]:
        cluster = event.cluster_id or ""
        if not cluster:
            return []
        found = []
        for other in self.ranking.events:
            if other.event_id == event.event_id or (other.cluster_id or "") != cluster:
                continue
            found.append({"event_id": other.event_id, "title": other.title, "primary": bool(other.is_primary)})
        return found

    def _card_claim(self, card) -> str | None:
        text = _CARD_TEXT.get(card.card_id)
        if text:
            return text
        for attr in ("claim", "text"):
            value = getattr(card, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        doc_id = getattr(card, "doc_id", None) or card.card_id
        chunks = getattr(card, "chunks", None)
        if chunks is None:
            chunks = getattr(self.ranking, "chunks", None)
        if isinstance(chunks, dict):
            chunk_text = chunks.get(card.card_id) or chunks.get(doc_id)
            if isinstance(chunk_text, str) and chunk_text.strip():
                return chunk_text.strip()
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    cid = chunk.get("doc_id") or chunk.get("card_id")
                    ctext = chunk.get("text") or chunk.get("claim")
                else:
                    cid = getattr(chunk, "doc_id", None) or getattr(chunk, "card_id", None)
                    ctext = getattr(chunk, "text", None) or getattr(chunk, "claim", None)
                if cid in {card.card_id, doc_id} and isinstance(ctext, str) and ctext.strip():
                    return ctext.strip()
        return None

    def search_alfa_analytics(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        topic = str(args.get("topic") or state.get("event_type") or "")
        as_of = as_utc(datetime.fromisoformat(state["as_of"]))
        limit = int(args.get("k") or 3)
        cards = []
        for card in self.ranking.cards:
            if topic and topic not in card.topics:
                continue
            published = as_utc(card.published_at)
            if published > as_of:
                continue
            text = self._card_claim(card)
            if not text:
                continue
            doc_id, page, bbox = _CARD_META.get(card.card_id, (card.card_id, 1, [0.08, 0.1, 0.92, 0.42]))
            stale = age_days(published, as_of) > self.config.stale_after_days
            cards.append(
                {
                    "card_id": card.card_id,
                    "doc_id": doc_id,
                    "page": page,
                    "bbox_rel": bbox,
                    "published_at": published.isoformat(),
                    "verified": card.verified,
                    "stale": stale,
                    "topic": list(card.topics),
                    "claim": text,
                    "source": "alfa",
                    "chunk_id": f"an:{doc_id}:p{page}:c0",
                }
            )
            if len(cards) >= limit:
                break
        source_ids = [f"card:{item['card_id']}" for item in cards]
        return ToolResult(ok=True, data={"cards": cards, "external": False}, source_ids=source_ids)

    def search_news(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        topic = str(args.get("topic") or state.get("event_type") or "")
        found = []
        for event in self.ranking.events:
            if event.event_id == state.get("event_id"):
                continue
            if not event.is_primary:
                continue
            if topic and event.event_type != topic:
                continue
            found.append(
                {
                    "event_id": event.event_id,
                    "title": event.title,
                    "text": wrap_data(event.text or ""),
                    "published_at": as_utc(event.published_at).isoformat(),
                    "primary": True,
                }
            )
            if len(found) >= 3:
                break
        return ToolResult(
            ok=True,
            data={"news": found},
            source_ids=[f"event:{item['event_id']}" for item in found],
        )

    def external_stats(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        indicator = str(args.get("indicator") or "")
        if indicator not in {"cpi", "unemployment", "industry_price"}:
            return ToolResult(ok=False, error="unknown_indicator")
        return ToolResult(
            ok=True,
            data={
                "indicator": indicator,
                "value": None,
                "unit": None,
                "source": "Росстат",
                "external": True,
                "note": "публичный ряд не подключён, число не подставляется",
            },
            source_ids=[f"external:{indicator}"],
        )

    def calc(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from alfa.agent.calc import build_calc
        from alfa.agent.interpret import _closed_ok

        fn = str(args.get("fn") or "")
        user = {item["field"]: item["value"] for item in state.get("user_provided") or []}
        features = ((state.get("views") or {}).get("profile") or {}).get("features") or {}
        fact_text = " ".join(
            str(item.get("claim") or "")
            for item in ((state.get("evidence") or {}).get("event") or {}).get("facts") or []
        )
        try:
            data = build_calc(
                fn,
                user=user,
                features=features,
                revenue_allowed=_closed_ok(self.config),
                fact_text=fact_text,
            )
        except KeyError:
            return ToolResult(ok=False, error="unknown_calc")
        return ToolResult(ok=True, data=data, source_ids=[data["calc_id"]])

    def explain_relevance(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        with session_scope(state["client_id"]):
            explanation = self.ranking.explain(state["event_id"], state["client_id"])
        hypotheses = []
        for item in explanation.hypotheses:
            hypotheses.append(
                {
                    "text": item.text,
                    "confidence": item.confidence,
                    "evidence": _jsonable(item.evidence),
                }
            )
        data = {
            "score": explanation.score,
            "relevance": explanation.relevance,
            "importance": explanation.importance,
            "components": dict(explanation.components),
            "reliability": explanation.reliability,
            "priority": explanation.priority,
            "mechanism_ids": list(explanation.fired_rules and [rule.mechanism_id for rule in explanation.fired_rules] or []),
            "rules": [rule.rule_id for rule in explanation.fired_rules],
            "rule_mechanisms": [rule.mechanism_id for rule in explanation.fired_rules],
            "above_threshold": explanation.above_threshold,
            "hypotheses": hypotheses,
            "missing_features": list(explanation.missing_features),
            "group_signal": explanation.group_signal.model_dump() if explanation.group_signal else None,
            "graph_signal": explanation.graph_signal.model_dump() if explanation.graph_signal else None,
        }
        # mechanism_ids без дублей, в порядке вклада
        ordered: list[str] = []
        for rule in explanation.fired_rules:
            if rule.mechanism_id not in ordered:
                ordered.append(rule.mechanism_id)
        data["mechanism_ids"] = ordered
        return ToolResult(ok=True, data=data, source_ids=["ranking:explanation"])
