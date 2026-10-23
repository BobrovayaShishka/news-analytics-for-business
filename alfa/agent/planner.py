"""План исполняет код. Модель может только предложить необязательные шаги."""

from __future__ import annotations

from typing import Any

from alfa.agent.calc import FUNCTIONS
from alfa.agent.config import TOPICS, TOOL_NAMES, AgentConfig, ToolStep

_WHY = {
    "get_event": "нужны факты события",
    "explain_relevance": "нужны механизмы и компоненты отбора",
    "get_profile_view": "нужны признаки компании",
    "get_group_context": "нужен агрегат группы без идентификаторов",
    "get_counterparties": "нужно обезличенное платёжное окружение",
    "search_alfa_analytics": "нужна карточка Альфа-Банка по теме",
    "search_news": "нужны связанные первичные новости",
    "external_stats": "нужен внешний показатель с явным источником",
    "calc": "нужен детерминированный расчёт",
}


def _schema_error(tool: str, args: dict[str, Any]) -> str | None:
    if tool not in TOOL_NAMES:
        return f"инструмент вне белого списка: {tool}"
    if "client_id" in args:
        return "client_id нельзя передавать в план"
    unknown = set(args) - _ALLOWED.get(tool, set())
    if unknown:
        return f"аргументы вне схемы: {sorted(unknown)}"
    if tool == "search_alfa_analytics" or tool == "search_news":
        topic = args.get("topic")
        if topic is not None and str(topic) not in TOPICS:
            return f"неизвестная тема: {topic}"
        if "k" in args and (not isinstance(args["k"], int) or args["k"] < 1 or args["k"] > 5):
            return "k вне диапазона"
    if tool == "get_counterparties" and "direction" in args and args["direction"] not in {"in", "out", "both"}:
        return "direction"
    if tool == "calc" and args.get("fn") not in FUNCTIONS:
        return "неизвестная функция calc"
    if tool == "external_stats" and args.get("indicator") not in {"cpi", "unemployment", "industry_price"}:
        return "неизвестный показатель"
    return None


_ALLOWED = {
    "get_profile_view": set(),
    "get_group_context": set(),
    "get_counterparties": {"direction"},
    "get_event": set(),
    "search_alfa_analytics": {"topic", "k", "q"},
    "search_news": {"topic"},
    "external_stats": {"indicator"},
    "calc": {"fn"},
    "explain_relevance": set(),
}


def step_key(tool: str, args: dict[str, Any]) -> tuple:
    return (tool, tuple(sorted((str(key), str(value)) for key, value in args.items())))


def _public_step(step_id: str, tool: str, args: dict[str, Any], required: bool) -> dict[str, Any]:
    return {
        "id": step_id,
        "tool": tool,
        "args": dict(args),
        "why": _WHY.get(tool, tool),
        "required": required,
    }


def specialize(args: dict[str, Any], event_type: str) -> dict[str, Any]:
    output = dict(args)
    if output.get("topic") == "from_event":
        output["topic"] = event_type if event_type in TOPICS else "other_econ"
    return output


def default_plan(mechanism_ids: list[str], event_type: str, config: AgentConfig) -> dict[str, Any]:
    collected: list[tuple[str, dict[str, Any], bool]] = []
    for step in config.base_tools:
        collected.append((step.tool, specialize(step.args, event_type), True))
    for mechanism_id in mechanism_ids:
        spec = config.mechanisms.get(mechanism_id)
        if spec is None:
            continue
        for step in spec.tools:
            collected.append((step.tool, specialize(step.args, event_type), True))
    deduped: list[tuple[str, dict[str, Any], bool]] = []
    seen: set[tuple] = set()
    for tool, args, required in collected:
        key = step_key(tool, args)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((tool, args, required))
    truncated = False
    if len(deduped) > config.max_steps:
        head = [item for item in deduped if item[2]][: config.max_steps]
        deduped = head
        truncated = True
    steps = [
        _public_step(f"s{index}", tool, args, required)
        for index, (tool, args, required) in enumerate(deduped, start=1)
    ]
    return {
        "steps": steps,
        "stop_conditions": ["max_steps", "budget", "tool_error_skip"],
        "truncated": truncated,
    }


def optional_catalog(mechanism_ids: list[str], event_type: str, config: AgentConfig) -> list[ToolStep]:
    items = list(config.optional_tools)
    for mechanism_id in mechanism_ids:
        spec = config.mechanisms.get(mechanism_id)
        if spec is None:
            continue
        items.extend(spec.optional_tools)
    return [ToolStep(tool=item.tool, args=specialize(item.args, event_type), required=False) for item in items]


def validate_plan(plan: dict[str, Any], config: AgentConfig) -> str | None:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return "план без steps"
    if len(steps) > config.max_steps:
        return "слишком много шагов"
    seen: set[tuple] = set()
    for step in steps:
        tool = str(step.get("tool") or "")
        args = dict(step.get("args") or {})
        error = _schema_error(tool, args)
        if error:
            return error
        key = step_key(tool, args)
        if key in seen:
            return "повтор шага"
        seen.add(key)
    return None


def merge_llm_plan(
    baseline: dict[str, Any],
    proposed: dict[str, Any],
    mechanism_ids: list[str],
    event_type: str,
    config: AgentConfig,
) -> tuple[dict[str, Any], str | None]:
    error = validate_plan(proposed, config)
    if error:
        return baseline, error
    required = {step_key(step["tool"], step["args"]) for step in baseline["steps"] if step.get("required", True)}
    proposed_keys = {step_key(step["tool"], dict(step.get("args") or {})) for step in proposed["steps"]}
    if not required <= proposed_keys:
        return baseline, "удалён обязательный шаг"
    allowed = {step_key(step.tool, step.args) for step in optional_catalog(mechanism_ids, event_type, config)}
    allowed |= {step_key(step["tool"], step["args"]) for step in baseline["steps"]}
    for step in proposed["steps"]:
        key = step_key(step["tool"], dict(step.get("args") or {}))
        if key not in allowed:
            return baseline, "шаг не из обязательных и не из необязательных"
    steps = []
    for index, step in enumerate(proposed["steps"], start=1):
        tool = str(step["tool"])
        args = dict(step.get("args") or {})
        steps.append(
            {
                "id": f"s{index}",
                "tool": tool,
                "args": args,
                "why": str(step.get("why") or _WHY.get(tool, tool))[:240],
                "required": step_key(tool, args) in required,
            }
        )
    if len(steps) > config.max_steps:
        return baseline, "слишком много шагов"
    return {"steps": steps, "stop_conditions": baseline.get("stop_conditions") or [], "truncated": False}, None
