"""Оркестратор LangGraph. План исполняет код, модель не вызывает инструменты сама.

init_context → plan → gather → ask_user → interpret → verify ─┬→ finalize
                                              ↑                 │
                                              └──── repair ─────┘
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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from alfa.agent.contract import assemble_insight
from alfa.agent.impact import assess_impact
from alfa.agent.interpret import build_context_pack, clip_text, llm_failure_kind, template_draft
from alfa.agent.models import DraftRegistry, PlanDraft, RepairDraft
from alfa.agent.planner import default_plan, merge_llm_plan, validate_plan
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
    if not result.ok:
        return calcs, None
    data = result.data or {}
    suspicious = None
    if tool == "get_event":
        evidence["event"] = data
        claim = ((data.get("facts") or [{}])[0]).get("claim") or ""
        evidence["sources"]["fact:f1"] = claim
        suspicious = bool(data.get("suspicious_instructions"))
    elif tool == "search_alfa_analytics":
        existing = {card["card_id"]: card for card in evidence["cards"]}
        for card in data.get("cards") or []:
            existing[card["card_id"]] = card
            evidence["sources"][f"card:{card['card_id']}"] = card["claim"]
        evidence["cards"] = list(existing.values())
    elif tool == "search_news":
        evidence["news"].extend(data.get("news") or [])
    elif tool == "external_stats":
        evidence["stats"].append(data)
        evidence["sources"].update({source_id: data.get("note") or "" for source_id in result.source_ids})
    elif tool == "calc":
        calcs = [item for item in calcs if item.get("calc_id") != data.get("calc_id")]
        calcs.append(data)
        evidence["sources"][data["calc_id"]] = data["text"]
    elif tool == "get_profile_view":
        views["profile"] = data
    elif tool == "get_group_context":
        views["group"] = data
    elif tool == "get_counterparties":
        views["counterparties"] = data
    elif tool == "explain_relevance":
        views["explanation"] = data
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


def _over_budget(state: AgentState, config) -> bool:
    started = float(state.get("started") or 0)
    if started and time.perf_counter() - started > config.max_seconds:
        return True
    return int(state.get("tokens") or 0) > config.max_tokens


def _blank_evidence() -> dict:
    return {"event": {}, "cards": [], "news": [], "stats": [], "sources": {}}


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
                                "default_steps": [step["tool"] for step in baseline["steps"]],
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
            journal.append(
                {
                    "type": "llm_call",
                    "role": "plan",
                    "model": result.model,
                    "prompt_version": config.prompt_versions.get("plan"),
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "cache_hit": result.cache_hit,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                }
            )
            tokens += int(result.tokens_in or 0) + int(result.tokens_out or 0)
            if not result.ok:
                journal.append({"type": "decision", "kind": "planner_unavailable", "reason": result.error or "planner_unavailable"})
            else:
                proposed = result.parsed.model_dump() if hasattr(result.parsed, "model_dump") else result.parsed
                if isinstance(proposed, str):
                    try:
                        proposed = json.loads(proposed)
                    except json.JSONDecodeError as exc:
                        journal.append({"type": "decision", "kind": "plan_rejected", "reason": str(exc)})
                        flags = _flags({"flags": flags}, "plan_rejected")
                        proposed = None
                if isinstance(proposed, dict):
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
                elif proposed is not None:
                    journal.append({"type": "decision", "kind": "plan_rejected", "reason": "план не объект"})
                    flags = _flags({"flags": flags}, "plan_rejected")
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
        evidence = {
            "event": {},
            "cards": [],
            "news": [],
            "stats": [],
            "sources": dict((state.get("evidence") or {}).get("sources") or {}),
        }
        views = dict(state.get("views") or {})
        calcs = list(state.get("calc_results") or [])
        working = dict(state)
        working["views"] = views
        suspicious = bool(state.get("suspicious"))
        for step in (state.get("plan") or {}).get("steps") or []:
            if _over_budget(working, config):
                flags = _flags({"flags": flags}, "budget_exhausted")
                break
            result, event = toolkit.call(step["tool"], step.get("args") or {}, working)
            journal.append(event)
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
            {"steps": [step.get("id") for step in (state.get("plan") or {}).get("steps") or []]},
            {
                "evidence": evidence,
                "views": views,
                "calc_results": calcs,
                "suspicious": suspicious,
                "flags": flags,
            },
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
                {"role": "system", "content": prompt + "\n" + _role_hint(local.get("role"))},
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
        started = time.perf_counter()
        iteration = int(state.get("iteration") or 0) + 1
        journal: list[dict] = []
        evidence = dict(state.get("evidence") or _blank_evidence())
        evidence["sources"] = dict(evidence.get("sources") or {})
        issues = (state.get("verification") or {}).get("issues") or []
        if any(item["type"] in {"ungrounded", "incomplete"} for item in issues) and not _over_budget(state, config):
            claim_text = ""
            for issue in issues:
                if issue["type"] in {"ungrounded", "incomplete"} and issue.get("claim_id"):
                    for claim in (state.get("draft") or {}).get("claims") or []:
                        if claim.get("id") == issue["claim_id"]:
                            claim_text = str(claim.get("text") or "")[:180]
            result, event = toolkit.call(
                "search_alfa_analytics",
                {"topic": state.get("event_type") or "other_econ", "q": claim_text},
                state,
            )
            journal.append(event)
            if result.ok:
                existing = {card["card_id"]: card for card in evidence.get("cards") or []}
                for card in result.data.get("cards") or []:
                    existing[card["card_id"]] = card
                    evidence["sources"][f"card:{card['card_id']}"] = card["claim"]
                evidence["cards"] = list(existing.values())
        views = dict(state.get("views") or {})
        calcs = list(state.get("calc_results") or [])
        retried = set(state.get("retried_steps") or [])
        flags = _flags(state)
        suspicious = bool(state.get("suspicious"))
        if any(item.get("type") == "incomplete" for item in issues):
            working = dict(state)
            working["evidence"] = evidence
            working["views"] = views
            working["calc_results"] = calcs
            for step in (state.get("plan") or {}).get("steps") or []:
                if not step.get("required") or f"skipped:{step['tool']}" not in flags or step.get("id") in retried:
                    continue
                result, event = toolkit.call(step["tool"], step.get("args") or {}, working)
                journal.append(event)
                retried.add(step.get("id"))
                if not result.ok:
                    continue
                calcs, suspicious_bit = _absorb(evidence, views, calcs, step["tool"], result)
                if suspicious_bit:
                    suspicious = True
                working["evidence"] = evidence
                working["views"] = views
                working["calc_results"] = calcs
        journal.append({"type": "decision", "kind": "repair", "iteration": iteration, "reason": "повторный проход проверки"})
        return _finish(
            "repair",
            started,
            journal,
            {"iteration": iteration, "retried": sorted(retried)},
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
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("interpret", interpret_node)
    graph.add_node("verify", verify_node)
    graph.add_node("repair", repair_node)
    graph.add_node("finalize", finalize_node)
    graph.add_conditional_edges(START, route_entry, {"init_context": "init_context", "interpret": "interpret"})
    graph.add_conditional_edges("init_context", route_after_init, {"plan": "plan", "finalize": "finalize"})
    graph.add_edge("plan", "gather")
    graph.add_edge("gather", "ask_user")
    graph.add_edge("ask_user", "interpret")
    graph.add_edge("interpret", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"repair": "repair", "finalize": "finalize"})
    graph.add_edge("repair", "interpret")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=MemorySaver())


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


def _role_hint(role: str | None) -> str:
    if role == "cfo":
        return "В «почему важно» сначала стоимость денег. Факты не меняй."
    if role == "ceo":
        return "В «на что смотреть» сначала устойчивость выручки. Факты не меняй."
    return "Порядок: что произошло, почему важно, на что смотреть."


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
    return data
