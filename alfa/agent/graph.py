"""Оркестратор LangGraph. План исполняет код, модель не вызывает инструменты сама.

План: YAML baseline (+ LLM optional) → gather → replan (адаптивно по observations) →
при новых optional-шагах снова gather. Обязательные шаги не удаляются; replan только
дописывает whitelist optional с явным why. При verify=repair — повторный поиск по
плану + ремонт claims.

При llm_enabled=False: baseline + heuristic_replan (те же правила, без модели).
gather вызывает Toolkit; модель инструменты не выбирает сама.
Ограничения инструментов — в docstring alfa.agent.tools.

Память между generate / answer / chat — InsightStore, не LangGraph checkpointer.

init_context → plan → gather ⇄ replan → ask_user → interpret → verify ─┬→ finalize
                                                         ↑              │
                                         research+claims └──── repair ──┘
"""

from __future__ import annotations

import copy
import json
import time
import warnings
from datetime import datetime, timedelta, timezone
from operator import add
from typing import Annotated, Any, TypedDict

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible",
)

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from alfa.agent.config import OPTIONAL_TOOL_NAMES
from alfa.agent.contract import assemble_insight
from alfa.agent.impact import assess_impact
from alfa.agent.consequences import apply_company_consequences
from alfa.agent.interpret import build_context_pack, clip_text, llm_failure_kind, template_draft
from alfa.agent.mechanism_text import role_hint
from alfa.agent.models import DraftRegistry, PlanDraft, RepairDraft
from alfa.agent.planner import (
    apply_replan_additions,
    default_plan,
    gather_observations,
    heuristic_replan_additions,
    merge_llm_plan,
    merge_llm_replan_additions,
    optional_catalog,
    step_key,
    step_key_str,
    validate_plan,
)
from alfa.agent.traceview import digest
from alfa.agent.verify import (
    apply_critic_status,
    apply_repair,
    bank_divergences,
    contradictions,
    decide,
    merge_replacements,
    run_critic,
    settle_blocking,
    should_call_critic,
    verify_stage1,
)
from alfa.core.trace import set_trace_id


class AgentState(TypedDict, total=False):
    trace_id: str
    insight_id: str
    client_id: str
    event_id: str
    as_of: str
    role: str
    user_id: str
    allowed_company_ids: list[str]
    priority: str
    reliability: str
    importance: float
    event_type: str
    mechanism_ids: list[str]
    reasons: dict
    explanation: dict
    plan: dict
    views: dict
    evidence: dict
    calc_results: list
    user_provided: list
    draft: dict
    draft_snapshot: dict
    verification: dict
    iteration: int
    flags: list
    status: str
    pending_question: dict | None
    insight: dict
    phase: str
    suspicious: bool
    started: float
    tokens: int
    questions_asked: int
    cost_rub: float
    repair_same_text: bool
    retried_steps: list
    gathered_keys: list
    replan_round: int
    journal: Annotated[list[dict], add]


def _flags(state: AgentState, *names: str) -> list[str]:
    flags = list(state.get("flags") or [])
    for name in names:
        if name and name not in flags:
            flags.append(name)
    return flags


def _node(
    name: str,
    started: float,
    extra: list[dict] | None = None,
    *,
    received: Any = None,
    output: Any = None,
    config=None,
) -> list[dict]:
    event = {
        "type": "node",
        "name": name,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "input_hash": digest(received if received is not None else {"name": name}),
        "output_ref": digest(_public_output(output) if output is not None else {"name": name}),
    }
    if config is not None:
        event["retain_until"] = (
            datetime.now(timezone.utc) + timedelta(days=int(config.prompt_retention_days))
        ).date().isoformat()
    return [event, *(extra or [])]


def _public_output(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    hidden = {"journal", "prompt", "messages", "response"}
    return {key: value for key, value in payload.items() if key not in hidden}


class BudgetGuard:
    """Потолок токенов и числа инсайтов на пользователя. Отдельный граф не нужен."""

    def __init__(self, config):
        self.max_tokens = int(config.session_max_tokens)
        self.max_insights = int(config.session_max_insights)

    def exceeded(self, store, user_id: str) -> bool:
        bucket = store.session_bucket(str(user_id or ""))
        if int(bucket.get("insights") or 0) >= self.max_insights:
            return True
        return int(bucket.get("tokens") or 0) > self.max_tokens


def _block_reason(state: AgentState, config, store) -> str | None:
    flags = set(state.get("flags") or [])
    if _over_budget(state, config) or "budget_exhausted" in flags:
        return "budget_exhausted"
    if BudgetGuard(config).exceeded(store, str(state.get("user_id") or "")):
        return "session_budget"
    return None


def _absorb(evidence: dict, views: dict, calcs: list, tool: str, result) -> tuple[list, bool | None]:
    from alfa.core.grounding import build_locator, locator_map, source_text_map

    if not result.ok:
        return calcs, None
    data = result.data or {}
    suspicious = None
    locators = dict(evidence.get("locators") or {})
    if tool == "get_event":
        evidence["event"] = data
        facts = list(data.get("facts") or [])
        echoes = list(data.get("echo_sources") or [])
        evidence["sources"].update(source_text_map(facts, echoes))
        # Явный текст из payload, если source_text уже собран.
        for key, value in (data.get("source_text") or {}).items():
            evidence["sources"][str(key)] = value
        locators.update(
            locator_map(
                facts,
                event_title=str(data.get("title") or ""),
                event_url=data.get("url"),
                event_text=str(data.get("text") or ""),
                publisher=data.get("source"),
                echoes=echoes,
            )
        )
        suspicious = bool(data.get("suspicious_instructions"))
    elif tool == "search_alfa_analytics":
        existing = {card["card_id"]: card for card in evidence["cards"]}
        for card in data.get("cards") or []:
            existing[card["card_id"]] = card
            evidence["sources"][str(card["card_id"])] = card["claim"]
        evidence["cards"] = list(existing.values())
        locators.update(locator_map([], cards=evidence["cards"]))
    elif tool == "search_news":
        news_items = list(data.get("news") or [])
        evidence["news"].extend(news_items)
        for item in news_items:
            event_id = item.get("event_id")
            if not event_id:
                continue
            key = f"event:{event_id}"
            body = str(item.get("text") or item.get("summary") or item.get("title") or "")
            span = str(item.get("summary") or "")[:240] or body[:240]
            evidence["sources"][key] = body or span
            locators[key] = build_locator(
                source_id=key,
                title=str(item.get("title") or event_id),
                url=item.get("url"),
                evidence_span=span,
                text=body or span,
                source_type="external_news",
                publisher=item.get("source"),
            )
    elif tool == "external_stats":
        evidence["stats"].append(data)
        text = str(data.get("evidence_span") or data.get("claim") or data.get("note") or "")
        if data.get("value") is not None:
            text = f"{text} {data.get('value')} {data.get('unit') or ''}".strip()
        evidence["sources"].update({source_id: text for source_id in result.source_ids})
        for source_id in result.source_ids:
            locators[str(source_id)] = build_locator(
                source_id=str(source_id),
                title=str(data.get("source") or source_id),
                url=data.get("url"),
                evidence_span=text,
                text=text,
                source_type="external_stats",
                publisher=str(data.get("as_of") or data.get("source") or "") or None,
            )
    elif tool == "calc":
        calcs = [item for item in calcs if item.get("calc_id") != data.get("calc_id")]
        calcs.append(data)
        calc_text = str(data.get("text") or "")
        evidence["sources"][data["calc_id"]] = calc_text
        locators[data["calc_id"]] = build_locator(
            source_id=data["calc_id"],
            title=str(data.get("fn") or data["calc_id"]),
            url=None,
            evidence_span=calc_text,
            text=calc_text,
            source_type="calc",
            publisher=str(data.get("formula") or "") or None,
        )
    elif tool == "get_profile_view":
        views["profile"] = data
    elif tool == "get_group_context":
        views["group"] = data
    elif tool == "get_counterparties":
        views["counterparties"] = data
    elif tool == "explain_relevance":
        views["explanation"] = data
    evidence["locators"] = locators
    return calcs, suspicious


def _empty_draft(config, headline: str) -> dict:
    definition = config.rubric["levels"]["low"]["definition"]
    return {
        "claims": [],
        "headline": headline,
        "what_happened": [],
        "why_it_matters": [],
        "what_to_watch": [],
        "impact_rationale": definition,
        "impact_level": "low",
        "impact_definition": definition,
        "scenarios": [],
        "assumptions": [],
        "risks": [],
        "opportunities": [],
        "bank_block_ids": [],
        "disclaimer": config.disclaimer,
        "diverges_from_bank_view": False,
        "confidence": 0.0,
    }


# Замечания, при которых имеет смысл повторить поиск (не только править текст).
_RESEARCH_ISSUES = frozenset(
    {
        "ungrounded",
        "incomplete",
        "missing_ref",
        "no_analytics",
        "stale",
        "unjustified_inference",
    }
)
# Какие инструменты плана трогаем под тип замечания. Только whitelist tools.
_RESEARCH_TOOLS: dict[str, frozenset[str]] = {
    "ungrounded": frozenset({"search_alfa_analytics", "get_event", "search_news"}),
    "incomplete": frozenset(
        {"search_alfa_analytics", "get_profile_view", "calc", "get_counterparties", "get_group_context"}
    ),
    "missing_ref": frozenset({"search_alfa_analytics", "get_event", "calc"}),
    "no_analytics": frozenset({"search_alfa_analytics", "search_news"}),
    "stale": frozenset({"search_alfa_analytics", "search_news"}),
    "unjustified_inference": frozenset({"search_alfa_analytics", "get_event", "calc"}),
}


def _claim_query(state: AgentState, issues: list[dict]) -> str:
    for issue in issues:
        if issue.get("type") not in _RESEARCH_ISSUES or not issue.get("claim_id"):
            continue
        for claim in (state.get("draft") or {}).get("claims") or []:
            if claim.get("id") == issue["claim_id"]:
                return str(claim.get("text") or "")[:180]
    return ""


def _research_steps(state: AgentState, issues: list[dict], config, retried: set[str]) -> list[dict]:
    """Шаги повторного поиска из принятого плана (+ optional whitelist). Без LLM re-plan."""
    needed: set[str] = set()
    for issue in issues:
        kind = str(issue.get("type") or "")
        if kind in _RESEARCH_TOOLS:
            needed |= _RESEARCH_TOOLS[kind]
    flags = set(state.get("flags") or [])
    for step in (state.get("plan") or {}).get("steps") or []:
        if step.get("required") and f"skipped:{step.get('tool')}" in flags:
            needed.add(str(step.get("tool") or ""))

    if not needed:
        return []

    event_type = str(state.get("event_type") or "other_econ")
    query = _claim_query(state, issues)
    selected: list[dict] = []
    seen: set[tuple] = set()

    def _add(tool: str, args: dict, step_id: str, *, required: bool) -> None:
        key = step_key(tool, args)
        if key in seen or step_id in retried:
            return
        seen.add(key)
        payload_args = dict(args)
        if tool == "search_alfa_analytics" and query and "q" not in payload_args:
            payload_args = {**payload_args, "q": query}
        selected.append(
            {"id": step_id, "tool": tool, "args": payload_args, "required": required, "why": "repair_research"}
        )

    for step in (state.get("plan") or {}).get("steps") or []:
        tool = str(step.get("tool") or "")
        if tool not in needed:
            continue
        _add(tool, dict(step.get("args") or {}), str(step.get("id") or tool), required=bool(step.get("required")))

    # Optional whitelist: дополнить поиск, если в плане шага ещё не было.
    for item in optional_catalog(list(state.get("mechanism_ids") or []), event_type, config):
        if item.tool not in needed or item.tool not in OPTIONAL_TOOL_NAMES:
            continue
        key = step_key(item.tool, item.args)
        suffix = "_".join(f"{name}-{value}" for name, value in key[1]) or "default"
        sid = f"research:{item.tool}:{suffix}"
        _add(item.tool, dict(item.args), sid, required=False)

    return selected[: max(1, int(config.max_steps))]


def _over_budget(state: AgentState, config) -> bool:
    started = float(state.get("started") or 0)
    if started and time.perf_counter() - started > config.max_seconds:
        return True
    return int(state.get("tokens") or 0) > config.max_tokens


def _blank_evidence() -> dict:
    return {"event": {}, "cards": [], "news": [], "stats": [], "sources": {}, "locators": {}}


def build_graph(runtime):
    toolkit = runtime.toolkit
    config = runtime.config

    def _finish(name: str, started: float, journal: list[dict], received: Any, payload: dict) -> dict:
        body = dict(payload)
        body["journal"] = _node(name, started, journal, received=received, output=body, config=config)
        return body

    def init_context(state: AgentState) -> dict:
        started = time.perf_counter()
        set_trace_id(state.get("trace_id"))
        journal: list[dict] = []
        flags = _flags(state)
        received = {"event_id": state.get("event_id"), "client_id": state.get("client_id")}
        if state["client_id"] not in set(state.get("allowed_company_ids") or []):
            flags = _flags({"flags": flags}, "access_denied")
            return _finish(
                "init_context",
                started,
                journal,
                received,
                {
                    "flags": flags,
                    "draft": _empty_draft(config, "Инсайт не опубликован"),
                    "verification": {"decision": "withheld", "stage1": {}, "stage2": {}, "issues": []},
                },
            )
        event = runtime.ranking.event_map().get(state["event_id"])
        if event is None:
            flags = _flags({"flags": flags}, "unknown_event")
            return _finish(
                "init_context",
                started,
                journal,
                received,
                {
                    "flags": flags,
                    "draft": _empty_draft(config, "Инсайт не опубликован"),
                    "verification": {"decision": "withheld", "stage1": {}, "stage2": {}, "issues": []},
                },
            )
        result, event_log = toolkit.call("explain_relevance", {}, state)
        journal.append(event_log)
        explanation = result.data if result.ok else {}
        if not result.ok:
            flags = _flags({"flags": flags}, "explain_failed")
        reasons = {
            mechanism_id: runtime.ranking.config.reason(mechanism_id)
            for mechanism_id in explanation.get("mechanism_ids") or []
        }
        return _finish(
            "init_context",
            started,
            journal,
            received,
            {
                "event_type": event.event_type,
                "mechanism_ids": list(explanation.get("mechanism_ids") or []),
                "priority": explanation.get("priority") or "regular",
                "reliability": explanation.get("reliability") or "none",
                "importance": float(explanation.get("importance") or 0.0),
                "explanation": explanation,
                "reasons": reasons,
                "flags": flags,
                "views": {"explanation": explanation},
                "evidence": _blank_evidence(),
            },
        )

    def plan_node(state: AgentState) -> dict:
        started = time.perf_counter()
        journal: list[dict] = []
        flags = _flags(state)
        baseline = default_plan(state.get("mechanism_ids") or [], state.get("event_type") or "", config)
        chosen = baseline
        tokens = int(state.get("tokens") or 0)
        block = _block_reason(state, config, runtime.store)
        if block == "session_budget":
            flags = _flags({"flags": flags}, "session_budget")
        if runtime.llm_enabled and runtime.llm is not None and block is None:
            plan_path = config.prompt_path("plan")
            plan_prompt = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""
            optional = optional_catalog(
                state.get("mechanism_ids") or [],
                state.get("event_type") or "",
                config,
            )
            result = runtime.llm.complete(
                role=config.roles.get("plan", "critic"),
                model="",
                messages=[
                    {"role": "system", "content": plan_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "event_type": state.get("event_type"),
                                "mechanisms": state.get("mechanism_ids"),
                                "default_steps": baseline["steps"],
                                "optional_steps": [
                                    {"tool": step.tool, "args": step.args, "required": False} for step in optional
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_schema=PlanDraft,
                temperature=0.0,
                trace_id=state.get("trace_id"),
                prompt_version=config.prompt_versions.get("plan"),
            )
            journal.append(_llm_event("plan", result, config.prompt_versions.get("plan")))
            tokens += int(result.tokens_in or 0) + int(result.tokens_out or 0)
            if not result.ok:
                journal.append({"type": "decision", "kind": "planner_unavailable", "reason": result.error or "planner_unavailable"})
            else:
                proposed = _parsed_dict(result)
                if proposed is None and result.parsed is not None:
                    journal.append({"type": "decision", "kind": "plan_rejected", "reason": "план не объект"})
                    flags = _flags({"flags": flags}, "plan_rejected")
                elif proposed is not None:
                    try:
                        merged, error = merge_llm_plan(
                            baseline,
                            proposed,
                            state.get("mechanism_ids") or [],
                            state.get("event_type") or "",
                            config,
                        )
                    except (TypeError, ValueError) as exc:
                        merged, error = baseline, str(exc)
                    if error:
                        journal.append({"type": "decision", "kind": "plan_rejected", "reason": error})
                        flags = _flags({"flags": flags}, "plan_rejected")
                    else:
                        chosen = merged
        error = validate_plan(chosen, config)
        if error:
            journal.append({"type": "decision", "kind": "plan_rejected", "reason": error})
            chosen = baseline
            flags = _flags({"flags": flags}, "plan_rejected")
        if chosen.get("truncated"):
            flags = _flags({"flags": flags}, "plan_truncated")
        journal.append({"type": "decision", "kind": "plan_selected", "plan": chosen})
        return _finish(
            "plan",
            started,
            journal,
            {"mechanism_ids": list(state.get("mechanism_ids") or [])},
            {"plan": chosen, "flags": flags, "tokens": tokens},
        )

    def gather_node(state: AgentState) -> dict:
        started = time.perf_counter()
        journal: list[dict] = []
        flags = _flags(state)
        prev = state.get("evidence") or {}
        evidence = {
            "event": dict(prev.get("event") or {}),
            "cards": list(prev.get("cards") or []),
            "news": list(prev.get("news") or []),
            "stats": list(prev.get("stats") or []),
            "sources": dict(prev.get("sources") or {}),
            "locators": dict(prev.get("locators") or {}),
        }
        views = dict(state.get("views") or {})
        calcs = list(state.get("calc_results") or [])
        gathered = set(state.get("gathered_keys") or [])
        working = dict(state)
        working["views"] = views
        working["evidence"] = evidence
        working["calc_results"] = calcs
        suspicious = bool(state.get("suspicious"))
        ran: list[str] = []
        for step in (state.get("plan") or {}).get("steps") or []:
            key = step_key_str(step["tool"], dict(step.get("args") or {}))
            if key in gathered:
                continue
            if _over_budget(working, config):
                flags = _flags({"flags": flags}, "budget_exhausted")
                break
            result, event = toolkit.call(step["tool"], step.get("args") or {}, working)
            journal.append(event)
            gathered.add(key)
            ran.append(str(step.get("id") or key))
            if not result.ok:
                flags = _flags({"flags": flags}, f"skipped:{step['tool']}")
                continue
            calcs, suspicious_bit = _absorb(evidence, views, calcs, step["tool"], result)
            if suspicious_bit:
                suspicious = True
            if step["tool"] == "explain_relevance":
                working["explanation"] = result.data
            working["views"] = views
            working["evidence"] = evidence
            working["calc_results"] = calcs
        if suspicious:
            flags = _flags({"flags": flags}, "suspicious_instructions")
        return _finish(
            "gather",
            started,
            journal,
            {"steps": ran},
            {
                "evidence": evidence,
                "views": views,
                "calc_results": calcs,
                "suspicious": suspicious,
                "flags": flags,
                "gathered_keys": sorted(gathered),
            },
        )

    def replan_node(state: AgentState) -> dict:
        """Адаптивное дописывание optional-шагов после gather; required не трогаем."""
        started = time.perf_counter()
        journal: list[dict] = []
        flags = _flags(state)
        tokens = int(state.get("tokens") or 0)
        round_no = int(state.get("replan_round") or 0)
        plan = dict(state.get("plan") or {})
        mechanisms = list(state.get("mechanism_ids") or [])
        event_type = str(state.get("event_type") or "")
        observations = gather_observations(state)
        additions: list[dict] = []
        reasons: list[str] = []
        block = _block_reason(state, config, runtime.store)
        if round_no >= int(config.max_replan) or block is not None or _over_budget(state, config):
            journal.append(
                {
                    "type": "decision",
                    "kind": "replan_skip",
                    "round": round_no,
                    "reason": block or ("max_replan" if round_no >= config.max_replan else "budget"),
                    "observations": observations,
                }
            )
            return _finish(
                "replan",
                started,
                journal,
                {"round": round_no},
                {"plan": plan, "flags": flags, "tokens": tokens, "replan_round": round_no},
            )

        if runtime.llm_enabled and runtime.llm is not None and block is None:
            optional = optional_catalog(mechanisms, event_type, config)
            plan_path = config.prompt_path("plan")
            plan_prompt = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""
            result = runtime.llm.complete(
                role=config.roles.get("plan", "critic"),
                model="",
                messages=[
                    {"role": "system", "content": plan_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "mode": "replan",
                                "event_type": event_type,
                                "mechanisms": mechanisms,
                                "observations": observations,
                                "default_steps": plan.get("steps") or [],
                                "optional_steps": [
                                    {"tool": step.tool, "args": step.args, "required": False} for step in optional
                                ],
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
                response_schema=PlanDraft,
                temperature=0.0,
                trace_id=state.get("trace_id"),
                prompt_version=config.prompt_versions.get("plan"),
            )
            journal.append(_llm_event("plan", result, config.prompt_versions.get("plan")))
            tokens += int(result.tokens_in or 0) + int(result.tokens_out or 0)
            if result.ok and result.parsed is not None:
                proposed = _parsed_dict(result)
                if isinstance(proposed, dict):
                    added, reasoning_text, error = merge_llm_replan_additions(
                        plan, proposed, mechanisms, event_type, config
                    )
                    if error:
                        journal.append({"type": "decision", "kind": "replan_rejected", "reason": error})
                        flags = _flags({"flags": flags}, "plan_rejected")
                    else:
                        additions.extend(added)
                        reasons.extend(str(step.get("why") or "") for step in added)
                        if reasoning_text:
                            reasons.insert(0, reasoning_text)

        if not additions:
            heur_add, heur_why = heuristic_replan_additions(plan, observations, mechanisms, event_type, config)
            additions = heur_add
            reasons = heur_why or reasons

        if not additions:
            journal.append(
                {
                    "type": "decision",
                    "kind": "replan_noop",
                    "round": round_no,
                    "observations": observations,
                    "reasoning": "пробелов для optional-шагов нет",
                }
            )
            return _finish(
                "replan",
                started,
                journal,
                {"round": round_no, "added": []},
                {"plan": plan, "flags": flags, "tokens": tokens, "replan_round": round_no},
            )

        reasoning = "; ".join(item for item in reasons if item)[:500]
        updated = apply_replan_additions(plan, additions, reasoning)
        error = validate_plan(updated, config)
        if error:
            journal.append({"type": "decision", "kind": "replan_rejected", "reason": error})
            return _finish(
                "replan",
                started,
                journal,
                {"round": round_no},
                {"plan": plan, "flags": _flags({"flags": flags}, "plan_rejected"), "tokens": tokens, "replan_round": round_no},
            )
        next_round = round_no + 1
        before_n = len(plan.get("steps") or [])
        added_ids = [step.get("id") for step in (updated.get("steps") or [])[before_n:]]
        journal.append(
            {
                "type": "decision",
                "kind": "replan",
                "round": next_round,
                "added": added_ids,
                "reasoning": reasoning,
                "observations": observations,
                "plan": updated,
            }
        )
        return _finish(
            "replan",
            started,
            journal,
            {"round": next_round, "added": added_ids},
            {"plan": updated, "flags": flags, "tokens": tokens, "replan_round": next_round},
        )

    def ask_user_node(state: AgentState) -> dict:
        started = time.perf_counter()
        question = runtime.select_question(state)
        journal = []
        flags = _flags(state)
        pending = None
        asked = int(state.get("questions_asked") or 0)
        if question is not None:
            pending = question
            asked += 1
            runtime.store.add_question(state["user_id"])
            flags = _flags({"flags": flags}, "conditional")
            journal.append({"type": "decision", "kind": "ask_user", "field": question["field"]})
        return _finish(
            "ask_user",
            started,
            journal,
            {"field": question["field"] if question else None},
            {"pending_question": pending, "questions_asked": asked, "flags": flags},
        )

    def interpret_node(state: AgentState) -> dict:
        started = time.perf_counter()
        journal: list[dict] = []
        flags = _flags(state)
        evidence = dict(state.get("evidence") or _blank_evidence())
        evidence["sources"] = dict(evidence.get("sources") or {})
        calcs = list(state.get("calc_results") or [])
        if state.get("phase") == "resume":
            for step in (state.get("plan") or {}).get("steps") or []:
                if step["tool"] != "calc":
                    continue
                result, event = toolkit.call("calc", step.get("args") or {}, state)
                journal.append(event)
                if result.ok:
                    calcs = [item for item in calcs if item.get("calc_id") != result.data.get("calc_id")]
                    calcs.append(result.data)
                    evidence["sources"][result.data["calc_id"]] = result.data["text"]
        local = dict(state)
        local["calc_results"] = calcs
        local["evidence"] = evidence
        iteration = int(state.get("iteration") or 0)
        state_tokens = int(state.get("tokens") or 0)
        repair_same = bool(state.get("repair_same_text"))
        if iteration > 0 and runtime.repair_override is not None:
            draft = runtime.repair_override(local)
        elif iteration > 0:
            draft, flags, repair_same, state_tokens = _repair_draft(local, flags, journal, state_tokens)
        elif runtime.interpret_override is not None:
            draft = runtime.interpret_override(local)
        else:
            block = _block_reason(local, config, runtime.store)
            if (not runtime.llm_enabled) or runtime.llm is None or block:
                draft = template_draft(local, config)
                flags = _flags({"flags": flags}, "template_fallback")
                if block == "session_budget":
                    flags = _flags({"flags": flags}, "session_budget")
                elif block == "budget_exhausted":
                    flags = _flags({"flags": flags}, "budget_exhausted")
                journal.append({"type": "decision", "kind": "llm_skipped", "reason": block or "template_fallback"})
            else:
                draft, flags, state_tokens = _interpret_model(local, flags, journal, state_tokens)
        extra = draft.pop("extra_sources", None) or {}
        evidence["sources"].update(extra)
        updates: dict[str, Any] = {
            "draft": draft,
            "evidence": evidence,
            "calc_results": calcs,
            "flags": flags,
            "tokens": state_tokens,
            "repair_same_text": repair_same,
        }
        if iteration == 0 and not state.get("draft_snapshot"):
            updates["draft_snapshot"] = json.loads(json.dumps(draft, ensure_ascii=False, default=str))
        return _finish(
            "interpret",
            started,
            journal,
            {"sources": sorted((evidence.get("sources") or {}).keys()), "iteration": iteration},
            updates,
        )

    def _interpret_model(local: dict, flags: list[str], journal: list[dict], state_tokens: int):
        pack = build_context_pack(local, config)
        path = config.prompt_path("interpret")
        prompt = path.read_text(encoding="utf-8") if path.exists() else ""
        result = runtime.llm.complete(
            role=config.roles.get("interpret", "interpret"),
            model="",
            messages=[
                {"role": "system", "content": prompt + "\n" + role_hint(local.get("role"), config)},
                {"role": "user", "content": pack},
            ],
            response_schema=DraftRegistry,
            temperature=0.0,
            trace_id=local.get("trace_id"),
            prompt_version=config.prompt_versions.get("interpret"),
        )
        journal.append(_llm_event("interpret", result, config.prompt_versions.get("interpret")))
        state_tokens += int(result.tokens_in or 0) + int(result.tokens_out or 0)
        parsed = result.parsed
        draft = None
        invalid = isinstance(parsed, str)
        if result.ok and parsed is not None and not isinstance(parsed, str):
            try:
                draft = _draft_from_parsed(parsed, local, config)
            except (TypeError, ValueError, ValidationError):
                invalid = True
        elif llm_failure_kind(result.error) == "invalid_json":
            invalid = True
        if draft is None and invalid:
            flags = _flags({"flags": flags}, "invalid_json")
            draft = _empty_draft(config, "Инсайт не опубликован")
        elif draft is None:
            draft = template_draft(local, config)
            flags = _flags({"flags": flags}, "template_fallback")
        return draft, flags, state_tokens

    def _repair_draft(local: dict, flags: list[str], journal: list[dict], state_tokens: int):
        issues = (local.get("verification") or {}).get("issues") or []
        draft = copy.deepcopy(local.get("draft") or {})
        if any(item.get("type") == "impact_inconsistent" for item in issues):
            draft["impact_rationale"] = assess_impact(local, config)["rationale"]
        blocking_ids = {
            item.get("claim_id")
            for item in issues
            if item.get("severity") == "blocking"
            and item.get("claim_id") not in {"", "impact"}
            and item.get("type") != "impact_inconsistent"
        }
        iteration = int(local.get("iteration") or 0)
        if iteration >= config.max_repair or local.get("repair_same_text"):
            draft = settle_blocking(draft, issues)
            journal.append(
                {
                    "type": "decision",
                    "kind": "repair",
                    "iteration": iteration,
                    "dropped": draft.pop("repair_dropped", []),
                    "reason": "удаление утверждений, которые остались после ремонта",
                }
            )
            return draft, flags, False, state_tokens
        if not blocking_ids:
            structural = any(
                item.get("severity") == "blocking" and item.get("type") != "impact_inconsistent" for item in issues
            )
            if structural:
                draft = apply_repair(draft, issues, local, config)
                journal.append(
                    {
                        "type": "decision",
                        "kind": "repair",
                        "iteration": iteration,
                        "dropped": draft.pop("repair_dropped", []),
                        "reason": "точечная замена утверждений с замечаниями",
                    }
                )
            return draft, flags, False, state_tokens
        block = _block_reason(local, config, runtime.store)
        if runtime.llm_enabled and runtime.llm is not None and block is None:
            return _repair_model(draft, issues, blocking_ids, local, flags, journal, state_tokens, iteration)
        if block == "session_budget":
            flags = _flags({"flags": flags}, "session_budget", "template_fallback")
        draft = apply_repair(draft, issues, local, config)
        journal.append(
            {
                "type": "decision",
                "kind": "repair",
                "iteration": iteration,
                "dropped": draft.pop("repair_dropped", []),
                "reason": "точечная замена утверждений с замечаниями",
            }
        )
        return draft, flags, False, state_tokens

    def _repair_model(draft, issues, blocking_ids, local, flags, journal, state_tokens, iteration):
        evidence = local.get("evidence") or {}
        sources = dict(evidence.get("sources") or {})
        hints: dict[str, list[str]] = {}
        for item in issues:
            if item.get("claim_id") in blocking_ids:
                hints.setdefault(str(item.get("claim_id")), []).append(str(item.get("hint") or ""))
        selected = []
        for claim in draft.get("claims") or []:
            if claim.get("id") not in blocking_ids:
                continue
            refs = list(claim.get("refs") or []) + list(claim.get("calc_refs") or []) + list(claim.get("evidence") or [])
            selected.append(
                {
                    "claim": {
                        "id": claim.get("id"),
                        "layer": claim.get("layer"),
                        "text": claim.get("text"),
                        "refs": list(claim.get("refs") or []),
                        "calc_refs": list(claim.get("calc_refs") or []),
                        "evidence": list(claim.get("evidence") or []),
                        "feature_refs": list(claim.get("feature_refs") or []),
                        "mechanism_id": claim.get("mechanism_id"),
                        "confidence": claim.get("confidence"),
                    },
                    "sources": {ref: sources.get(ref, "") for ref in refs},
                    "hint": " ".join(hints.get(str(claim.get("id")), [])),
                }
            )
        cards = [
            {"card_id": card.get("card_id"), "claim": card.get("claim"), "published_at": card.get("published_at")}
            for card in evidence.get("cards") or []
        ]
        path = config.prompt_path("repair")
        prompt = path.read_text(encoding="utf-8") if path.exists() else ""
        result = runtime.llm.complete(
            role=config.roles.get("repair", "interpret"),
            model="",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "claims": selected,
                            "cards": cards,
                            "sources": {key: clip_text(str(value), 800) for key, value in sources.items()},
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                },
            ],
            response_schema=RepairDraft,
            temperature=0.0,
            trace_id=local.get("trace_id"),
            prompt_version=config.prompt_versions.get("repair"),
        )
        journal.append(_llm_event("repair", result, config.prompt_versions.get("repair")))
        state_tokens += int(result.tokens_in or 0) + int(result.tokens_out or 0)
        accepted = False
        same = False
        merged = draft
        if result.ok and result.parsed is not None and not isinstance(result.parsed, str):
            replacements = result.parsed.replacements if hasattr(result.parsed, "replacements") else result.parsed.get("replacements") or []
            try:
                merged, accepted, same = merge_replacements(draft, replacements, sources)
            except (TypeError, ValueError):
                accepted = False
        if accepted:
            journal.append(
                {
                    "type": "decision",
                    "kind": "repair",
                    "iteration": iteration,
                    "dropped": [],
                    "reason": "точечная замена утверждений с замечаниями",
                }
            )
            return merged, flags, same, state_tokens
        flags = _flags({"flags": flags}, "repair_template")
        fallback = apply_repair(draft, issues, local, config)
        journal.append(
            {
                "type": "decision",
                "kind": "repair",
                "iteration": iteration,
                "dropped": fallback.pop("repair_dropped", []),
                "reason": "шаблон ремонта: модель не вернула замену",
            }
        )
        return fallback, flags, False, state_tokens

    def verify_node(state: AgentState) -> dict:
        started = time.perf_counter()
        journal: list[dict] = []
        flags = _flags(state)
        if "invalid_json" in flags:
            verification = {"decision": "withheld", "stage1": {"issues": []}, "stage2": {"issues": [], "backend": "skipped"}, "issues": []}
            journal.append({"type": "decision", "kind": "verification", "decision": "withheld", "iterations": state.get("iteration") or 0, "stage1": {}, "stage2": {}})
            return _finish(
                "verify",
                started,
                journal,
                {"claims": [], "decision": "withheld"},
                {"verification": verification, "flags": flags},
            )
        draft = dict(state.get("draft") or {})
        stage1 = verify_stage1(draft, state, config)
        divs = bank_divergences(draft, state)
        if divs:
            draft["diverges_from_bank_view"] = True
        stage2: dict[str, Any] = {"issues": [], "backend": "skipped"}
        if should_call_critic(state, draft, stage1, config, force=bool(runtime.force_critic)):
            stage2 = run_critic(draft, state, config, runtime.llm, llm_enabled=bool(runtime.llm_enabled))
            if stage2.get("backend") == "llm":
                journal.append({"type": "llm_call", "role": "critic", "prompt_version": config.prompt_versions.get("critic")})
        issues = list(stage1["issues"]) + list(stage2.get("issues") or [])
        decision = decide(issues)
        flags, decision = apply_critic_status(flags, draft, state, stage2, decision)
        if any(item["type"] == "stale" for item in issues):
            flags = _flags({"flags": flags}, "stale_analytics")
        if any(item["type"] == "no_analytics" for item in issues):
            flags = _flags({"flags": flags}, "no_bank_analytics")
        verification = {
            "decision": decision,
            "stage1": stage1,
            "stage2": stage2,
            "issues": issues,
            "contradictions": contradictions((state.get("evidence") or {}).get("cards") or []) + divs,
        }
        journal.append(
            {
                "type": "decision",
                "kind": "verification",
                "decision": decision,
                "iterations": int(state.get("iteration") or 0),
                "stage1": {"issues": stage1["issues"], "decision": stage1["decision"]},
                "stage2": {"issues": stage2.get("issues") or [], "backend": stage2.get("backend")},
            }
        )
        claim_ids = [str(item.get("id") or "") for item in draft.get("claims") or []]
        return _finish(
            "verify",
            started,
            journal,
            {"claims": claim_ids, "decision": decision},
            {"verification": verification, "flags": flags, "draft": draft},
        )

    def repair_node(state: AgentState) -> dict:
        """Повторный поиск по фиксированному плану, затем interpret правит claims."""
        started = time.perf_counter()
        iteration = int(state.get("iteration") or 0) + 1
        journal: list[dict] = []
        evidence = dict(state.get("evidence") or _blank_evidence())
        evidence["sources"] = dict(evidence.get("sources") or {})
        views = dict(state.get("views") or {})
        calcs = list(state.get("calc_results") or [])
        retried = set(state.get("retried_steps") or [])
        flags = _flags(state)
        suspicious = bool(state.get("suspicious"))
        issues = (state.get("verification") or {}).get("issues") or []
        researched: list[str] = []
        if not _over_budget(state, config):
            working = dict(state)
            working["evidence"] = evidence
            working["views"] = views
            working["calc_results"] = calcs
            for step in _research_steps(state, issues, config, retried):
                if _over_budget(working, config):
                    flags = _flags({"flags": flags}, "budget_exhausted")
                    break
                result, event = toolkit.call(step["tool"], step.get("args") or {}, working)
                journal.append(event)
                step_id = str(step.get("id") or step["tool"])
                retried.add(step_id)
                researched.append(step_id)
                if not result.ok:
                    continue
                calcs, suspicious_bit = _absorb(evidence, views, calcs, step["tool"], result)
                if suspicious_bit:
                    suspicious = True
                if step["tool"] == "explain_relevance":
                    working["explanation"] = result.data
                working["evidence"] = evidence
                working["views"] = views
                working["calc_results"] = calcs
        journal.append(
            {
                "type": "decision",
                "kind": "repair",
                "iteration": iteration,
                "reason": "повторный поиск по плану, затем ремонт утверждений",
                "researched": researched,
            }
        )
        return _finish(
            "repair",
            started,
            journal,
            {"iteration": iteration, "retried": sorted(retried), "researched": researched},
            {
                "iteration": iteration,
                "evidence": evidence,
                "views": views,
                "calc_results": calcs,
                "retried_steps": sorted(item for item in retried if item),
                "suspicious": suspicious,
                "flags": flags,
            },
        )

    def finalize_node(state: AgentState) -> dict:
        started = time.perf_counter()
        insight = assemble_insight(state, config)
        tokens = int(state.get("tokens") or 0)
        cost = round(tokens / 1000 * config.rub_per_1k, 6)
        latency = round((time.perf_counter() - float(state.get("started") or started)) * 1000, 3)
        journal = [
            {
                "type": "decision",
                "kind": "status",
                "status": insight["status"],
                "llm_tokens": tokens,
                "cost_rub": cost,
                "latency_ms": latency,
            }
        ]
        return _finish(
            "finalize",
            started,
            journal,
            {"status": insight["status"]},
            {"insight": insight, "status": insight["status"], "cost_rub": cost},
        )

    def route_entry(state: AgentState) -> str:
        return "interpret" if state.get("phase") == "resume" else "init_context"

    def route_after_init(state: AgentState) -> str:
        flags = set(state.get("flags") or [])
        if "access_denied" in flags or "unknown_event" in flags:
            return "finalize"
        return "plan"

    def route_after_replan(state: AgentState) -> str:
        gathered = set(state.get("gathered_keys") or [])
        for step in (state.get("plan") or {}).get("steps") or []:
            key = step_key_str(step["tool"], dict(step.get("args") or {}))
            if key not in gathered:
                return "gather"
        return "ask_user"

    def route_after_verify(state: AgentState) -> str:
        flags = set(state.get("flags") or [])
        if "invalid_json" in flags:
            return "finalize"
        decision = (state.get("verification") or {}).get("decision")
        if decision == "repair" and int(state.get("iteration") or 0) < config.max_repair:
            return "repair"
        return "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("init_context", init_context)
    graph.add_node("plan", plan_node)
    graph.add_node("gather", gather_node)
    graph.add_node("replan", replan_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("interpret", interpret_node)
    graph.add_node("verify", verify_node)
    graph.add_node("repair", repair_node)
    graph.add_node("finalize", finalize_node)
    graph.add_conditional_edges(START, route_entry, {"init_context": "init_context", "interpret": "interpret"})
    graph.add_conditional_edges("init_context", route_after_init, {"plan": "plan", "finalize": "finalize"})
    graph.add_edge("plan", "gather")
    graph.add_edge("gather", "replan")
    graph.add_conditional_edges("replan", route_after_replan, {"gather": "gather", "ask_user": "ask_user"})
    graph.add_edge("ask_user", "interpret")
    graph.add_edge("interpret", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"repair": "repair", "finalize": "finalize"})
    graph.add_edge("repair", "interpret")
    graph.add_edge("finalize", END)
    # Без checkpointer: диалог и resume живут в InsightStore, не в MemorySaver.
    return graph.compile()


def _llm_event(role: str, result, prompt_version: str | None) -> dict:
    return {
        "type": "llm_call",
        "role": role,
        "model": result.model,
        "prompt_version": prompt_version,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "cache_hit": result.cache_hit,
        "error": result.error,
        "latency_ms": result.latency_ms,
    }


def _parsed_dict(result) -> dict | None:
    """Достать объект из LLM complete (model_dump / JSON-строка)."""
    proposed = result.parsed.model_dump() if hasattr(result.parsed, "model_dump") else result.parsed
    if isinstance(proposed, str):
        try:
            proposed = json.loads(proposed)
        except json.JSONDecodeError:
            return None
    return proposed if isinstance(proposed, dict) else None


def _draft_from_parsed(parsed: Any, state: dict, config) -> dict:
    if isinstance(parsed, str):
        raise ValueError("textual parsed")
    if hasattr(parsed, "model_dump"):
        data = parsed.model_dump()
    elif isinstance(parsed, dict):
        data = dict(parsed)
    else:
        raise TypeError("parsed is not an object")
    data = DraftRegistry.model_validate(data).model_dump()
    impact = assess_impact(state, config)
    data["impact_level"] = impact["level"]
    data["impact_definition"] = impact["definition"]
    if not data.get("impact_rationale"):
        data["impact_rationale"] = impact["rationale"]
    data["disclaimer"] = config.disclaimer
    data["risks"] = list(data.get("risks") or [])
    data["opportunities"] = list(data.get("opportunities") or [])
    data["bank_block_ids"] = [item["id"] for item in data.get("claims") or [] if item.get("layer") == "alfa"]
    data["diverges_from_bank_view"] = any(item.get("diverges_from_bank_view") for item in data.get("claims") or [])
    data["confidence"] = float(data.get("confidence") or 0.5)
    data.setdefault("scenarios", [])
    data.setdefault("assumptions", [])
    # LLM мог оставить пустые risks/opportunities/scenarios — добираем клиентский слой кодом.
    return apply_company_consequences(data, state, config)
