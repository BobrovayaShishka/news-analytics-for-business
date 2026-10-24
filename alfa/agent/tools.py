"""Инструменты только на чтение. client_id берётся из сессии, не из аргумента модели.

Инструменты вызывает код (gather по плану), модель их не выбирает.
search_alfa_analytics: гибридный поиск alfa/kb (BM25 + эмбеддинги) по фикстурам аналитики.
search_news: отбор primary-событий каталога по теме с url и сниппетами.
external_stats: ряд из фикстур alfa/data/fixtures/external; без ряда — value=None.
get_event: атомарные факты с evidence_span ⊆ текста, url и char-позициями; эхо сливаются.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from functools import lru_cache
from typing import Any

from alfa.access import session_scope
from alfa.agent.config import BANK_VIEW_TYPES, AgentConfig
from alfa.agent.models import ToolResult
from alfa.agent.security import detect_injection, mask_args, redact_tree, wrap_data
from alfa.core.config import load_runtime_config, resolve_path
from alfa.core.grounding import (
    atomic_facts_from_text,
    facts_from_extraction,
    merge_echo_facts,
    source_text_map,
    summarize_facts,
)
from alfa.core.trace import log_step
from alfa.kb.index import AnalyticsIndex
from alfa.kb.search import build_default_index, search_analytics
from alfa.ranking.engine import age_days, as_utc
from alfa.settings import load_profile_config

_DEFAULT_BBOX = [0.08, 0.1, 0.92, 0.42]
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
    "market_rank",
    "market_peers",
    "market_okved2",
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


def _row(ranking: Any, client_id: str):
    for row in ranking.rows:
        if row.client_id == client_id:
            return row
    return None


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


@lru_cache(maxsize=1)
def _external_stats_rows() -> list[dict[str, Any]]:
    cfg = load_runtime_config()
    directory = resolve_path((cfg.get("paths") or {}).get("external_fixtures") or "alfa/data/fixtures/external")
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


class Toolkit:
    def __init__(
        self,
        ranking: Any,
        config: AgentConfig,
        journal_path,
        clock: Callable[[], datetime] | None = None,
        kb_index: AnalyticsIndex | None = None,
    ):
        self.ranking = ranking
        self.config = config
        self.journal_path = journal_path
        self.clock = clock or (lambda: datetime.now())
        self.calls: Counter[str] = Counter()
        self._kb_index = kb_index

    def reset(self) -> None:
        self.calls = Counter()

    @property
    def kb_index(self) -> AnalyticsIndex:
        if self._kb_index is None:
            # HashingEmbedder: без загрузки bge-m3 на старте агента.
            self._kb_index = build_default_index(prefer_real=False)
        return self._kb_index

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
        url = str(getattr(event, "url", "") or "") or None
        publisher = str(getattr(event, "source", "") or "") or None
        title = str(event.title or "")
        stored = list(getattr(event, "facts", None) or [])
        if stored and any(item.get("evidence_span") or item.get("claim") for item in stored if isinstance(item, dict)):
            facts = facts_from_extraction(
                stored,
                text=raw,
                title=title,
                url=url,
                publisher=publisher,
            )
            for index, row in enumerate(facts):
                origin = stored[index] if index < len(stored) else {}
                if isinstance(origin, dict) and origin.get("id"):
                    row["id"] = str(origin["id"])
                if isinstance(origin, dict) and origin.get("source_ids"):
                    row["source_ids"] = list(origin["source_ids"])
        else:
            facts = atomic_facts_from_text(
                raw,
                title=title,
                url=url,
                publisher=publisher,
                skip_injection=detect_injection,
            )
        for fact in facts:
            cleaned, _hits = redact_tree(fact.get("claim") or "")
            fact["claim"] = cleaned
            span, _ = redact_tree(fact.get("evidence_span") or "")
            fact["evidence_span"] = span
        echoes = self._echoes(event)
        facts, disagreements = merge_echo_facts(facts, echoes)
        entities = [_jsonable(item) for item in (event.entities or [])]
        source_ids = [f"fact:{fact['id']}" for fact in facts]
        for echo in echoes:
            source_ids.append(f"event:{echo['event_id']}")
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "title": title,
            "url": url,
            "source": publisher,
            "text": wrap_data(raw),
            "facts": facts,
            "summary": summarize_facts(facts),
            "entities": entities,
            "risks": _as_texts(getattr(event, "risks", None)),
            "opportunities": _as_texts(getattr(event, "opportunities", None)),
            "cluster_id": event.cluster_id,
            "echo_sources": echoes,
            "echo_disagreements": disagreements,
            "source_text": source_text_map(facts, echoes),
            "suspicious_instructions": suspicious,
            "published_at": as_utc(event.published_at).isoformat(),
        }
        return ToolResult(ok=True, data=data, source_ids=source_ids)

    def _echoes(self, event) -> list[dict[str, Any]]:
        cluster = event.cluster_id or ""
        if not cluster:
            return []
        found = []
        for other in self.ranking.events:
            if other.event_id == event.event_id or (other.cluster_id or "") != cluster:
                continue
            other_text = other.text or other.title or ""
            other_url = str(getattr(other, "url", "") or "") or None
            other_publisher = str(getattr(other, "source", "") or "") or None
            other_facts = list(getattr(other, "facts", None) or [])
            if not other_facts:
                other_facts = atomic_facts_from_text(
                    other_text,
                    title=str(other.title or ""),
                    url=other_url,
                    publisher=other_publisher,
                    skip_injection=detect_injection,
                )
            found.append(
                {
                    "event_id": other.event_id,
                    "title": other.title,
                    "primary": bool(other.is_primary),
                    "url": other_url,
                    "publisher": other_publisher,
                    "source": other_publisher,
                    "text": other_text,
                    "facts": other_facts,
                }
            )
        return found

    def _kb_record(self, hit_id: str):
        for collection in (self.kb_index.cards, self.kb_index.chunks):
            for record in collection.records:
                if record.id == hit_id:
                    return record
        return None

    @staticmethod
    def _claim_from_text(text: str, *, kind: str = "card") -> str:
        # У карточек в индексе text = claim + metric/topics; claim — первая строка.
        value = (text or "").strip()
        if kind == "card" and "\n" in value:
            return value.split("\n", 1)[0].strip()
        return value

    def _card_payload(self, record, *, as_of: datetime, score: float = 0.0) -> dict[str, Any] | None:
        claim = self._claim_from_text(record.text, kind=record.kind)
        if not claim:
            return None
        published = as_utc(record.published_at)
        claim_type = str(record.extra.get("claim_type") or "")
        return {
            "card_id": record.id,
            "doc_id": record.doc_id,
            "page": int(record.page or 1),
            "bbox_rel": list(record.bbox_rel or _DEFAULT_BBOX),
            "published_at": published.isoformat(),
            "verified": str(record.extra.get("verified") or "manual"),
            "stale": age_days(published, as_of) > self.config.stale_after_days,
            "topic": list(record.topics or []),
            "claim": claim,
            "claim_type": claim_type,
            "source": "alfa",
            "chunk_id": f"an:{record.doc_id}:p{int(record.page or 1)}:c0",
            "score": score,
        }

    def _freshest_bank_view(self, topic: str, as_of: datetime):
        best = None
        for record in self.kb_index.cards.records:
            if topic and topic not in (record.topics or []):
                continue
            if as_utc(record.published_at) > as_of:
                continue
            if str(record.extra.get("claim_type") or "") not in BANK_VIEW_TYPES:
                continue
            if best is None or as_utc(record.published_at) > as_utc(best.published_at):
                best = record
        return best

    def _event_query(self, state: dict[str, Any]) -> str:
        event = self.ranking.event_map().get(state.get("event_id"))
        if event is None:
            return ""
        return " ".join(part for part in (event.title, event.text) if part)[:400]

    def search_alfa_analytics(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from alfa.ranking.open_world import infer_event_types, is_open_topic

        topic = str(args.get("topic") or state.get("event_type") or "")
        event = self.ranking.event_map().get(state.get("event_id"))
        query = str(args.get("q") or "").strip() or self._event_query(state) or topic or "аналитика"
        as_of = as_utc(datetime.fromisoformat(state["as_of"]))
        limit = int(args.get("k") or 3)
        if is_open_topic(topic):
            inferred = infer_event_types(event) if event is not None else []
            topic_filter = inferred or None
        else:
            topic_filter = [topic] if topic else None
        hits = search_analytics(
            query,
            topics=topic_filter,
            as_of=as_of,
            k=max(limit * 3, 9),
            mode="hybrid",
            include_cards=True,
            include_chunks=False,
            index=self.kb_index,
        )
        if is_open_topic(topic) and not hits:
            hits = search_analytics(
                query,
                topics=None,
                as_of=as_of,
                k=max(limit * 3, 9),
                mode="hybrid",
                include_cards=True,
                include_chunks=False,
                index=self.kb_index,
            )
        by_id: dict[str, dict[str, Any]] = {}
        for hit in hits:
            record = self._kb_record(hit.id)
            if record is None:
                continue
            payload = self._card_payload(record, as_of=as_of, score=float(hit.score or 0))
            if payload is not None:
                by_id[payload["card_id"]] = payload
        # Гибрид может утопить свежий взгляд банка — гарантируем свежайшую bank-view по теме.
        pin_topics = list(topic_filter or ([topic] if topic and not is_open_topic(topic) else []))
        if pin_topics:
            fresh = self._freshest_bank_view(pin_topics[0], as_of)
            if fresh is not None and fresh.id not in by_id:
                payload = self._card_payload(fresh, as_of=as_of, score=1.0)
                if payload is not None:
                    by_id[payload["card_id"]] = payload
            elif fresh is not None:
                by_id[fresh.id]["score"] = max(float(by_id[fresh.id].get("score") or 0), 1.0)
        cards = list(by_id.values())
        bank = [item for item in cards if item.get("claim_type") in BANK_VIEW_TYPES]
        other = [item for item in cards if item.get("claim_type") not in BANK_VIEW_TYPES]
        bank.sort(key=lambda item: (item.get("published_at") or "", float(item.get("score") or 0)), reverse=True)
        other.sort(key=lambda item: (item.get("published_at") or "", float(item.get("score") or 0)), reverse=True)
        cards = (bank + other)[:limit]
        # card_id уже вида card:doc:n — без второго префикса.
        source_ids = [str(item["card_id"]) for item in cards]
        return ToolResult(ok=True, data={"cards": cards, "external": False}, source_ids=source_ids)

    def search_news(self, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from alfa.ranking.open_world import infer_event_types, is_open_topic

        # Отбор primary-событий: по теме или, для открытого мира, по выведенным темам текста.
        topic = str(args.get("topic") or state.get("event_type") or "")
        current = self.ranking.event_map().get(state.get("event_id"))
        inferred = infer_event_types(current) if current is not None and is_open_topic(topic) else []
        found = []
        for event in self.ranking.events:
            if event.event_id == state.get("event_id"):
                continue
            if not event.is_primary:
                continue
            if is_open_topic(topic):
                if inferred and event.event_type not in inferred:
                    continue
            elif topic and event.event_type != topic:
                continue
            raw = event.text or ""
            url = str(getattr(event, "url", "") or "") or None
            publisher = str(getattr(event, "source", "") or "") or None
            facts = list(getattr(event, "facts", None) or [])
            if not facts:
                facts = atomic_facts_from_text(
                    raw,
                    title=str(event.title or ""),
                    url=url,
                    publisher=publisher,
                    skip_injection=detect_injection,
                    max_facts=3,
                )
            found.append(
                {
                    "event_id": event.event_id,
                    "title": event.title,
                    "url": url,
                    "source": publisher,
                    "text": wrap_data(raw),
                    "facts": facts,
                    "summary": summarize_facts(facts, max_items=3),
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
        # Фикстурный внешний ряд; если индикатора нет — value=None, число не выдумывается.
        indicator = str(args.get("indicator") or "")
        if indicator not in {"cpi", "unemployment", "industry_price"}:
            return ToolResult(ok=False, error="unknown_indicator")
        as_of = as_utc(datetime.fromisoformat(state["as_of"]))
        match = None
        for row in _external_stats_rows():
            if str(row.get("indicator") or "") != indicator:
                continue
            published = row.get("as_of") or row.get("published_at")
            if published:
                try:
                    stamp = as_utc(datetime.fromisoformat(str(published).replace("Z", "+00:00")))
                except ValueError:
                    stamp = None
                if stamp is not None and stamp > as_of:
                    continue
            if match is None:
                match = row
                continue
            prev = match.get("as_of") or match.get("published_at") or ""
            cur = row.get("as_of") or row.get("published_at") or ""
            if str(cur) > str(prev):
                match = row
        if match is None:
            return ToolResult(
                ok=True,
                data={
                    "indicator": indicator,
                    "value": None,
                    "unit": None,
                    "source": "Росстат",
                    "external": True,
                    "url": None,
                    "note": "публичный ряд не найден в фикстурах, число не подставляется",
                },
                source_ids=[f"external:{indicator}"],
            )
        value = match.get("value")
        note = str(match.get("note") or match.get("claim") or "")
        return ToolResult(
            ok=True,
            data={
                "indicator": indicator,
                "value": value,
                "unit": match.get("unit"),
                "source": str(match.get("source") or "Росстат"),
                "external": True,
                "url": match.get("url"),
                "as_of": match.get("as_of") or match.get("published_at"),
                "claim": note,
                "evidence_span": str(match.get("evidence_span") or note),
                "note": note,
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
            "summary": explanation.summary,
            "signals": [item.model_dump() for item in explanation.signals],
        }
        # mechanism_ids без дублей, в порядке вклада
        ordered: list[str] = []
        for rule in explanation.fired_rules:
            if rule.mechanism_id not in ordered:
                ordered.append(rule.mechanism_id)
        data["mechanism_ids"] = ordered
        return ToolResult(ok=True, data=data, source_ids=["ranking:explanation"])
