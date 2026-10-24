"""Уточнения по сохранённому инсайту (не агентный цикл generate).

By design без LLM на классификацию и черновик ответа — даже при llm_enabled=True:
intent и срезы (cut) идут через regex и шаблоны Toolkit, чтобы offline и
llm_enabled=False вели себя одинаково, а refuse/UID оставались детерминированными.
Critic на follow-up может вызвать LLM только если runtime.llm_enabled (как в verify).
Контекст берётся из InsightStore, не из MemorySaver.
"""

from __future__ import annotations

import re
from typing import Any

from alfa.agent.chat_followup import append_claim, apply_clarify, apply_source, model_anchor, model_claim
from alfa.agent.contract import assemble_insight
from alfa.agent.interpret import template_draft
from alfa.agent.verify import apply_critic_status, decide, naked_numbers, run_critic, should_call_critic, verify_stage1

_REFUSE = re.compile(
    r"что купить|что продать|инвестиционн|другая компания|чуж(ой|ая|ую) компан|внутренн(ий|ие|его)\s+рейтинг|покажи\s+uid",
    re.IGNORECASE,
)
_SOURCE = re.compile(r"откуда|источник|на ч[её]м основан|как получен", re.IGNORECASE)
_UID = re.compile(r"\bU_\d+\b")
_NO_SLICE = "В этом срезе нет данных."


def classify(question: str, cut: str | None) -> str:
    text = question or ""
    if _REFUSE.search(text) or (_UID.search(text) and not _SOURCE.search(text)):
        return "refuse"
    if cut == "source" or _SOURCE.search(text):
        return "source"
    if cut:
        return "cut"
    return "clarify"


def answer_followup(runtime, record: dict[str, Any], question: str, cut: str | None) -> dict[str, Any]:
    state = json_copy(record["state"])
    state["journal"] = list(record.get("journal") or [])
    repeated = _same_question(state["journal"], question)
    turns = int(record.get("turns") or 0)
    if not repeated:
        turns += 1
    record["turns"] = turns
    intent = "limit" if turns > runtime.config.max_chat_turns or len(question or "") > 1000 else classify(question, cut)
    state["journal"].append(
        {
            "type": "user_question",
            "field": cut or intent,
            "answer": (question or "")[:300],
            "ts": state.get("as_of"),
        }
    )
    original_ids = {claim.get("id") for claim in (state.get("draft") or {}).get("claims") or []}
    draft = _follow_draft(runtime, state, intent, cut, question)
    draft.pop("extra_sources", None)
    stage1 = verify_stage1(draft, state, runtime.config)
    stage2 = {"issues": [], "backend": "skipped"}
    if should_call_critic(state, draft, stage1, runtime.config, force=bool(runtime.force_critic)):
        stage2 = run_critic(draft, state, runtime.config, runtime.llm, llm_enabled=bool(runtime.llm_enabled))
    issues = list(stage1["issues"]) + list(stage2.get("issues") or [])
    decision = decide(issues)
    if decision == "repair":
        draft, stage1, decision, issues = _drop_new_blocking(draft, state, runtime, original_ids, issues, stage2)
    flags = list(state.get("flags") or [])
    flags, decision = apply_critic_status(flags, draft, state, stage2, decision)
    if intent == "refuse":
        flags.append("out_of_scope")
    elif intent == "limit":
        flags.append("chat_limit")
    elif intent == "source":
        flags.append("source_lookup")
    elif intent == "clarify":
        flags.append("clarify")
    else:
        flags.append(f"cut:{cut}")
    state["draft"] = draft
    state["verification"] = {
        "decision": decision,
        "stage1": stage1,
        "stage2": stage2,
        "issues": issues,
        "contradictions": list((state.get("verification") or {}).get("contradictions") or []),
    }
    state["flags"] = flags
    state["pending_question"] = None
    insight = assemble_insight(state, runtime.config)
    state["journal"].append({"type": "decision", "kind": "followup", "intent": intent, "decision": decision})
    follow_id = f"{record['insight']['insight_id']}:q{turns}"
    insight["insight_id"] = follow_id
    runtime.store.put(state, insight)
    runtime.store.records[record["insight"]["insight_id"]]["turns"] = turns
    runtime.store.records[record["insight"]["insight_id"]]["journal"] = state["journal"]
    runtime.store.save()
    return insight


def _same_question(journal: list[dict], question: str) -> bool:
    for event in reversed(journal):
        if event.get("type") == "user_question":
            return str(event.get("answer") or "") == (question or "")[:300]
    return False


def _follow_draft(runtime, state: dict, intent: str, cut: str | None, question: str = "") -> dict:
    base = json_copy(state.get("draft") or {}) or template_draft(state, runtime.config)
    if not base.get("claims"):
        base = template_draft(state, runtime.config)
    if intent in {"refuse", "limit"}:
        text = (
            "На такой запрос ответа нет: здесь только это событие и эта компания, без совета по сделке и без закрытых данных."
            if intent == "refuse"
            else "Лимит диалога по этому инсайту исчерпан."
        )
        append_claim(base, model_claim(state, "c_refuse" if intent == "refuse" else "c_limit", text), "why_it_matters")
        return base
    if intent == "source":
        apply_source(state, base)
        return base
    if intent == "clarify":
        apply_clarify(state, base, question)
        return base
    _apply_cut(runtime, state, base, cut or "")
    return base


def _apply_cut(runtime, state: dict, draft: dict, cut: str) -> None:
    runtime.toolkit.reset()
    mechanisms = set(state.get("mechanism_ids") or [])
    steps = _cut_steps(cut, state, mechanisms)
    produced = False
    for tool, args in steps:
        result, _event = runtime.toolkit.call(tool, args, state)
        if not result.ok:
            continue
        _absorb_cut(state, tool, result.data)
        claim = _claim_from_cut(cut, tool, result.data, state)
        if claim is None:
            continue
        append_claim(draft, claim, "why_it_matters" if claim["layer"] == "model" else "what_happened")
        produced = True
    if cut == "scenario":
        claim = _scenario_claim(state)
        if claim is not None:
            append_claim(draft, claim, "why_it_matters")
            produced = True
    if not produced:
        append_claim(draft, _empty_slice_claim(state), "what_happened")


def _cut_steps(cut: str, state: dict, mechanisms: set[str]) -> list[tuple[str, dict]]:
    if cut == "cost_side":
        steps = [("get_profile_view", {})]
        seen = set()
        for item in state.get("calc_results") or []:
            fn = item.get("fn")
            if fn and fn not in seen:
                seen.add(fn)
                steps.append(("calc", {"fn": fn}))
        return steps
    if cut == "credit_portfolio":
        if "debt_service_cost" not in mechanisms:
            return []
        return [("get_profile_view", {}), ("calc", {"fn": "rate_sensitivity"})]
    if cut == "counterparties":
        return [("get_counterparties", {})]
    if cut == "group":
        return [("get_group_context", {})]
    return []


def _absorb_cut(state: dict, tool: str, data: dict) -> None:
    views = dict(state.get("views") or {})
    if tool == "get_profile_view":
        views["profile"] = data
    elif tool == "get_group_context":
        views["group"] = data
    elif tool == "get_counterparties":
        views["counterparties"] = data
    elif tool == "calc" and data.get("calc_id"):
        evidence = dict(state.get("evidence") or {})
        sources = dict(evidence.get("sources") or {})
        sources[data["calc_id"]] = data.get("text") or ""
        evidence["sources"] = sources
        state["evidence"] = evidence
        calcs = [item for item in state.get("calc_results") or [] if item.get("calc_id") != data.get("calc_id")]
        calcs.append(data)
        state["calc_results"] = calcs
    state["views"] = views


def _claim_from_cut(cut: str, tool: str, data: dict, state: dict) -> dict | None:
    if tool == "calc" and data.get("calc_id"):
        anchor = model_anchor(state)
        return {
            "id": f"c_cut_{cut}_{data['fn']}",
            "layer": "model",
            "text": data.get("text") or "",
            "refs": [],
            "feature_refs": anchor["feature_refs"],
            "mechanism_id": anchor["mechanism_id"],
            "calc_refs": [data["calc_id"]],
        }
    if tool == "get_profile_view" and cut in {"cost_side", "credit_portfolio"}:
        return model_claim(state, f"c_cut_{cut}_profile", "Срез собран из профиля компании и уже посчитанного расчёта.")
    if tool == "get_counterparties":
        if not (data.get("counterparties") or []):
            return None
        return model_claim(state, "c_cut_counterparties", "Платёжное окружение в этом срезе показано обезличенно.")
    if tool == "get_group_context":
        if not data.get("in_group"):
            return None
        return model_claim(state, "c_cut_group", "Компания входит в группу, состав показан агрегатом без идентификаторов.")
    return None


def _scenario_claim(state: dict) -> dict | None:
    scenarios = (state.get("draft") or {}).get("scenarios") or []
    if not scenarios:
        return None
    sources = (state.get("evidence") or {}).get("sources") or {}
    corpus = "\n".join(str(value) for value in sources.values())
    text = " ".join(f"{item.get('name')}: {item.get('logic')}" for item in scenarios)
    if naked_numbers(text) - naked_numbers(corpus):
        text = "В черновике уже есть сценарии, новый поиск не выполнялся."
    return model_claim(state, "c_cut_scenario", text)


def _empty_slice_claim(state: dict) -> dict:
    return {"id": "c_cut_empty", "layer": "fact", "text": _NO_SLICE, "refs": [_known_source(state)]}


def _known_source(state: dict) -> str:
    sources = (state.get("evidence") or {}).get("sources") or {}
    facts = [key for key in sources if str(key).startswith("fact:") and "@" not in str(key)]
    if facts:
        # Стабильный порядок: f1, f2, …
        facts.sort(key=lambda key: (len(str(key)), str(key)))
        return str(facts[0])
    for key in sources:
        if str(key).startswith("fact:"):
            return str(key)
    return "fact:f1"


def _drop_new_blocking(draft, state, runtime, original_ids, issues, stage2):
    blocking_new = {
        item.get("claim_id")
        for item in issues
        if item.get("severity") == "blocking" and item.get("claim_id") and item.get("claim_id") not in original_ids
    }
    if blocking_new:
        draft["claims"] = [claim for claim in draft.get("claims") or [] if claim.get("id") not in blocking_new]
        for section in ("what_happened", "why_it_matters", "what_to_watch"):
            draft[section] = [item for item in draft.get(section) or [] if item not in blocking_new]
    stage1 = verify_stage1(draft, state, runtime.config)
    issues = list(stage1["issues"]) + list(stage2.get("issues") or [])
    return draft, stage1, decide(issues), issues


def json_copy(payload: dict) -> dict:
    import json

    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
