"""API инсайта для интерфейса. Сессия проверяется на каждом методе."""

from __future__ import annotations

import copy
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from alfa.agent.chat import answer_followup
from alfa.agent.config import AgentConfig, load_agent_config
from alfa.agent.graph import build_graph
from alfa.agent.models import (
    AgentError,
    AnswerRequest,
    ChatRequest,
    CheckNewsRequest,
    FeedbackRequest,
    GenerateRequest,
    Session,
)
from alfa.agent.store import InsightStore
from alfa.agent.tools import Toolkit
from alfa.agent.traceview import build_trace_view
from alfa.core.config import resolve_path
from alfa.ranking.engine import as_utc
from alfa.ranking.models import RankEvent


class InsightService:
    """Методы совпадают с HTTP-контрактом: generate, get, answer, chat, trace, feedback, check_news."""

    def __init__(
        self,
        ranking=None,
        config: AgentConfig | None = None,
        store_dir: Path | None = None,
        llm=None,
        llm_enabled: bool = False,
        journal_path: Path | None = None,
    ):
        self.config = config or load_agent_config()
        self.ranking = ranking if ranking is not None else _catalog_ranking()
        root = Path(store_dir) if store_dir else resolve_path("data/agent")
        self.store = InsightStore(root)
        self.llm = llm
        self.llm_enabled = llm_enabled
        self.interpret_override = None
        self.repair_override = None
        self.force_critic = False
        self.journal_path = Path(journal_path) if journal_path else root / "journal.jsonl"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.toolkit = Toolkit(self.ranking, self.config, self.journal_path)
        self.graph = build_graph(self)
        self.cache: dict[tuple, dict] = {}

    def generate(self, session: Session, event_id: str, client_id: str, role: str | None = None) -> dict:
        self._check(session, client_id)
        resolved_role = role or session.role
        cache_key = self._cache_key(session, event_id, client_id, resolved_role)
        if self.interpret_override is None and self.repair_override is None and cache_key in self.cache:
            return copy.deepcopy(self.cache[cache_key])
        state = self._initial(session, event_id, client_id, resolved_role)
        before_tokens = int(state.get("tokens") or 0)
        final = self._invoke(state)
        insight = final["insight"]
        self.store.put(final, insight)
        self._account(session.user_id, before_tokens, final)
        if self.interpret_override is None and self.repair_override is None:
            self.cache[cache_key] = copy.deepcopy(insight)
        return copy.deepcopy(insight)

    def get(self, session: Session, insight_id: str) -> dict:
        record = self._record(insight_id)
        self._check(session, record["client_id"])
        return record["insight"]

    def answer(self, session: Session, insight_id: str, value: str, field: str | None = None) -> dict:
        record = self._record(insight_id)
        self._check(session, record["client_id"])
        pending = (record.get("insight") or {}).get("pending_question")
        if not pending:
            raise AgentError("no_question", "По этому инсайту нет открытого вопроса", 409)
        target = field or pending["field"]
        if target != pending["field"]:
            raise AgentError("wrong_field", "Поле не совпадает с вопросом")
        mapped = self.map_option(target, value)
        if mapped is None:
            raise AgentError("invalid_answer", "Ответ не входит в варианты")
        answered_at = datetime.now(timezone.utc).isoformat()
        self.store.remember_answer(session.user_id, record["client_id"], target, mapped, answered_at)
        state = _copy(record["state"])
        state["journal"] = list(record.get("journal") or [])
        state["user_provided"] = list(self.store.answers_for(session.user_id, record["client_id"]).values())
        state["phase"] = "resume"
        state["iteration"] = 0
        state["pending_question"] = None
        state["questions_asked"] = 1
        state["verification"] = {}
        state["started"] = time.perf_counter()
        state["journal"].append({"type": "user_question", "field": target, "answer": mapped, "ts": answered_at, "trace_id": state.get("trace_id")})
        before_tokens = int(state.get("tokens") or 0)
        final = self._invoke(state)
        final["insight"]["insight_id"] = insight_id
        final["insight"]["trace_id"] = record["trace_id"]
        self.store.put(final, final["insight"])
        self._account(session.user_id, before_tokens, final)
        self.cache = {key: item for key, item in self.cache.items() if key[1] != record["client_id"]}
        return copy.deepcopy(final["insight"])

    def chat(self, session: Session, insight_id: str, question: str, cut: str | None = None) -> dict:
        record = self._record(insight_id)
        self._check(session, record["client_id"])
        return answer_followup(self, record, question, cut)

    def trace(self, session: Session, trace_id: str) -> dict:
        record = self.store.get_by_trace(trace_id)
        if record is None:
            raise AgentError("not_found", "Трасса не найдена", 404)
        self._check(session, record["client_id"])
        return build_trace_view(record.get("journal") or [], trace_id)

    def feedback(self, session: Session, insight_id: str, label: str, reason: str = "") -> dict:
        record = self._record(insight_id)
        self._check(session, record["client_id"])
        if label not in {"useful", "irrelevant", "inaccurate"}:
            raise AgentError("bad_label", "Неизвестная оценка")
        item = {
            "insight_id": insight_id,
            "user_id": session.user_id,
            "label": label,
            "reason": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.store.add_feedback(item)
        return {"ok": True}

    def check_news(self, session: Session, client_id: str, text: str | None = None, url: str | None = None, title: str | None = None) -> dict:
        self._check(session, client_id)
        if not text:
            raise AgentError("needs_text", "Загрузка ссылки в этом контуре не подключена, передайте текст.", 400)
        event_id = "adhoc_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        event = RankEvent(
            event_id=event_id,
            event_type=_guess_type(text),
            published_at=as_utc(session.as_of or self.ranking.as_of),
            title=title or "Проверка новости",
            text=text,
            trust_tier="unverified",
            cluster_id=event_id,
            fact_signature=event_id,
        )
        original = self.ranking.events
        self.ranking.events = [*original, event]
        try:
            return self.generate(session, event_id, client_id, session.role)
        finally:
            self.ranking.events = original

    def select_question(self, state: dict) -> dict | None:
        user_id = str(state.get("user_id") or "")
        if self.store.question_count.get(user_id, 0) >= self.config.max_questions_per_session:
            return None
        if int(state.get("questions_asked") or 0) >= self.config.max_questions_per_insight:
            return None
        if _priority_rank(str(state.get("priority") or "regular")) < _priority_rank(self.config.min_priority):
            return None
        profile = ((state.get("views") or {}).get("profile") or {})
        if not profile:
            return None
        structural = set(profile.get("structurally_missing") or [])
        provided = {item["field"] for item in state.get("user_provided") or []}
        provided.update(self.store.answers_for(user_id, str(state.get("client_id") or "")))
        order = ["debt_rate_type", "fx_expense_share", "fx_revenue_share", "debt_amount", "debt_maturity"]
        pending: list[str] = []
        for mechanism_id in state.get("mechanism_ids") or []:
            spec = self.config.mechanisms.get(mechanism_id)
            if spec is None:
                continue
            for field_name in spec.critical_fields:
                if field_name not in spec.changes_conclusion or field_name in provided:
                    continue
                if structural and field_name not in structural:
                    continue
                if field_name not in pending:
                    pending.append(field_name)
        if not pending:
            return None
        pending.sort(key=lambda name: order.index(name) if name in order else len(order))
        field_name = pending[0]
        question = self.config.questions[field_name]
        return {
            "field": field_name,
            "text": question.text,
            "options": [item["label"] for item in question.options],
            "why_asked": question.why_asked,
        }

    def map_option(self, field: str, value: str) -> str | None:
        spec = self.config.questions.get(field)
        if spec is None:
            return None
        raw = str(value).strip()
        for option in spec.options:
            if raw.casefold() == option["label"].casefold() or raw == option["value"]:
                return option["value"]
        return None

    def _invoke(self, state: dict) -> dict:
        self.toolkit.reset()
        thread = f"{state['trace_id']}:{uuid.uuid4().hex[:8]}"
        return self.graph.invoke(state, {"recursion_limit": 40, "configurable": {"thread_id": thread}})

    def _initial(self, session: Session, event_id: str, client_id: str, role: str) -> dict:
        trace_id = uuid.uuid4().hex
        as_of = as_utc(session.as_of or self.ranking.as_of)
        return {
            "trace_id": trace_id,
            "insight_id": "ins_" + trace_id[:16],
            "client_id": client_id,
            "event_id": event_id,
            "as_of": as_of.isoformat(),
            "role": role,
            "user_id": session.user_id,
            "allowed_company_ids": list(session.allowed_company_ids),
            "priority": "regular",
            "reliability": "none",
            "importance": 0.0,
            "event_type": "",
            "mechanism_ids": [],
            "reasons": {},
            "explanation": {},
            "plan": {},
            "views": {},
            "evidence": {"event": {}, "cards": [], "news": [], "stats": [], "sources": {}},
            "calc_results": [],
            "user_provided": list(self.store.answers_for(session.user_id, client_id).values()),
            "draft": {},
            "verification": {},
            "iteration": 0,
            "flags": [],
            "pending_question": None,
            "phase": "run",
            "suspicious": False,
            "started": time.perf_counter(),
            "tokens": 0,
            "questions_asked": 0,
            "journal": [],
        }

    def _check(self, session: Session, client_id: str) -> None:
        if client_id not in set(session.allowed_company_ids):
            raise AgentError("access_denied", "Компания не входит в сессию", 403)

    def _record(self, insight_id: str) -> dict:
        record = self.store.get(insight_id)
        if record is None:
            raise AgentError("not_found", "Инсайт не найден", 404)
        return record

    def _cache_key(self, session: Session, event_id: str, client_id: str, role: str) -> tuple:
        answers = self.store.answers_for(session.user_id, client_id)
        signature = tuple(sorted((field, item["value"]) for field, item in answers.items()))
        as_of = as_utc(session.as_of or self.ranking.as_of).isoformat()
        return (session.user_id, client_id, event_id, role, signature, self.config.version, as_of)

    def _account(self, user_id: str, before_tokens: int, final: dict) -> None:
        delta = int(final.get("tokens") or 0) - int(before_tokens or 0)
        self.store.add_session_usage(user_id, max(0, delta))


def _priority_rank(value: str) -> int:
    return {"low": 0, "regular": 1, "urgent": 2}.get(value, -1)


def _guess_type(text: str) -> str:
    lowered = text.lower()
    checks = (
        ("золот", "commodities_gold"),
        ("курс", "fx"),
        ("доллар", "fx"),
        ("зарплат", "labor"),
        ("мрот", "labor"),
        ("инфляц", "inflation"),
        ("ставк", "key_rate"),
    )
    for needle, event_type in checks:
        if needle in lowered:
            return event_type
    return "other_econ"


def _catalog_ranking():
    from alfa.ranking.catalog import AS_OF, catalog_cards, catalog_events, catalog_rows
    from alfa.ranking.config import load_ranking_config
    from alfa.ranking.service import RankingService

    return RankingService(catalog_rows(), catalog_events(), catalog_cards(), load_ranking_config(), as_of=AS_OF)


def _copy(payload: dict) -> dict:
    import json

    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def session_from_generate(body: GenerateRequest) -> Session:
    as_of = datetime.fromisoformat(body.as_of) if body.as_of else None
    return Session(
        user_id=body.user_id,
        allowed_company_ids=list(body.allowed_company_ids),
        active_client_id=body.active_client_id or body.client_id,
        role=body.role,
        as_of=as_of,
    )


def create_app(service: InsightService | None = None):
    """HTTP-обёртка. Нужен пакет fastapi; сами сценарии ходят через InsightService."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise AgentError("missing_http", "fastapi не установлен") from exc

    app = FastAPI(title="alfa-agent")
    bound = service or InsightService()

    def _session(user_id: str, allowed: list[str], role: str = "default", client_id: str | None = None, as_of: str | None = None) -> Session:
        parsed = datetime.fromisoformat(as_of) if as_of else None
        return Session(
            user_id=user_id,
            allowed_company_ids=list(allowed),
            active_client_id=client_id or (allowed[0] if allowed else ""),
            role=role,  # type: ignore[arg-type]
            as_of=parsed,
        )

    @app.post("/insights/generate")
    def generate(body: GenerateRequest):
        try:
            return bound.generate(session_from_generate(body), body.event_id, body.client_id, body.role)
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    @app.get("/insights/{insight_id}")
    def get_insight(insight_id: str, user_id: str, allowed_company_ids: str):
        try:
            session = _session(user_id, allowed_company_ids.split(","))
            return bound.get(session, insight_id)
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    @app.post("/insights/{insight_id}/answer")
    def answer(insight_id: str, body: AnswerRequest):
        try:
            session = _session(body.user_id, body.allowed_company_ids)
            return bound.answer(session, insight_id, body.value, body.field)
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    @app.post("/chat")
    def chat(body: ChatRequest, insight_id: str):
        try:
            session = _session(body.user_id, body.allowed_company_ids)
            return bound.chat(session, insight_id, body.question, body.cut)
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    @app.get("/trace/{trace_id}")
    def trace(trace_id: str, user_id: str, allowed_company_ids: str):
        try:
            session = _session(user_id, allowed_company_ids.split(","))
            return bound.trace(session, trace_id)
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    @app.post("/feedback")
    def feedback(body: FeedbackRequest):
        try:
            session = _session(body.user_id, body.allowed_company_ids)
            return bound.feedback(session, body.insight_id, body.label, body.reason)
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    @app.post("/check-news")
    def check_news(body: CheckNewsRequest):
        try:
            session = _session(body.user_id, body.allowed_company_ids, client_id=body.client_id)
            return bound.check_news(session, body.client_id, body.text, body.url, body.title)
        except AgentError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message}) from exc

    return app
