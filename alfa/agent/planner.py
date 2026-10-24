"""План: YAML baseline → опционально LLM → после gather адаптивный replan.

Обязательные шаги из YAML не удаляются. Перепланирование после gather может
только добавить шаги из optional whitelist (search_news, external_stats,
get_group_context) с явным why. Свободного выбора произвольных tools нет.
"""

from __future__ import annotations

from typing import Any

from alfa.agent.calc import FUNCTIONS
from alfa.agent.config import OPTIONAL_TOOL_NAMES, TOPICS, TOOL_NAMES, AgentConfig, ToolStep

_WHY = {
    "get_event": "нужны факты события",
    "explain_relevance": "нужно объяснение значимости события для клиента",
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


def step_key_str(tool: str, args: dict[str, Any]) -> str:
    key = step_key(tool, args)
    tail = "&".join(f"{name}={value}" for name, value in key[1])
    return f"{tool}|{tail}" if tail else tool


def reason_why(tool: str, args: dict[str, Any], mechanism_ids: list[str], event_type: str) -> str:
    """Контекстный why для шага плана (не общая заглушка)."""
    mechanisms = ", ".join(mechanism_ids) or "без механизма"
    topic = str(args.get("topic") or event_type or "")
    if tool == "get_event":
        return f"зафиксировать факты события типа {event_type or 'unknown'}"
    if tool == "explain_relevance":
        return f"получить механизмы отбора для события {event_type or 'unknown'}"
    if tool == "get_profile_view":
        return f"снять признаки компании под механизмы: {mechanisms}"
    if tool == "search_alfa_analytics":
        return f"найти карточку Альфы по теме {topic or 'from_event'} для {mechanisms}"
    if tool == "calc":
        return f"посчитать {args.get('fn') or 'calc'} по профилю для {mechanisms}"
    if tool == "get_counterparties":
        return f"посмотреть платёжное окружение ({args.get('direction') or 'both'}) при валютном/торговом контуре"
    if tool == "search_news":
        return f"добавить первичные новости по теме {topic or event_type} — банк-карточек может не хватить"
    if tool == "external_stats":
        return f"взять внешний ряд {args.get('indicator') or 'indicator'} как опору для макро-контекста"
    if tool == "get_group_context":
        return f"проверить агрегат группы: механизмы {mechanisms} могут затрагивать связанный контур"
    return _WHY.get(tool, tool)


def _public_step(
    step_id: str,
    tool: str,
    args: dict[str, Any],
    required: bool,
    *,
    mechanism_ids: list[str] | None = None,
    event_type: str = "",
    why: str | None = None,
) -> dict[str, Any]:
    return {
        "id": step_id,
        "tool": tool,
        "args": dict(args),
        "why": (why or reason_why(tool, args, mechanism_ids or [], event_type))[:240],
        "required": required,
    }


def topic_for_event(event_type: str) -> str:
    return event_type if event_type in TOPICS else "other_econ"


def specialize(args: dict[str, Any], event_type: str) -> dict[str, Any]:
    output = dict(args)
    if output.get("topic") == "from_event":
        output["topic"] = topic_for_event(event_type)
    return output


def macro_stats_indicator(event_type: str) -> str | None:
    if event_type in {"key_rate", "inflation"}:
        return "cpi"
    if event_type == "labor":
        return "unemployment"
    return None


def default_plan(mechanism_ids: list[str], event_type: str, config: AgentConfig) -> dict[str, Any]:
    """План из YAML: base_tools + tools механизмов. Без optional и без LLM."""
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
        _public_step(f"s{index}", tool, args, required, mechanism_ids=mechanism_ids, event_type=event_type)
        for index, (tool, args, required) in enumerate(deduped, start=1)
    ]
    return {
        "steps": steps,
        "stop_conditions": ["max_steps", "budget", "tool_error_skip", "max_replan"],
        "truncated": truncated,
        "reasoning": f"базовый план по механизмам [{', '.join(mechanism_ids) or '—'}] и событию {event_type or '—'}",
    }


def optional_catalog(mechanism_ids: list[str], event_type: str, config: AgentConfig) -> list[ToolStep]:
    items = list(config.optional_tools)
    for mechanism_id in mechanism_ids:
        spec = config.mechanisms.get(mechanism_id)
        if spec is None:
            continue
        items.extend(spec.optional_tools)
    catalog: list[ToolStep] = []
    seen: set[tuple] = set()
    for item in items:
        if item.tool not in OPTIONAL_TOOL_NAMES:
            continue
        args = specialize(item.args, event_type)
        key = step_key(item.tool, args)
        if key in seen:
            continue
        seen.add(key)
        catalog.append(ToolStep(tool=item.tool, args=args, required=False))
    return catalog


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
                "why": str(step.get("why") or reason_why(tool, args, mechanism_ids, event_type))[:240],
                "required": step_key(tool, args) in required,
            }
        )
    if len(steps) > config.max_steps:
        return baseline, "слишком много шагов"
    reasoning = str(proposed.get("reasoning") or baseline.get("reasoning") or "")
    return {
        "steps": steps,
        "stop_conditions": baseline.get("stop_conditions") or [],
        "truncated": False,
        "reasoning": reasoning[:500],
    }, None


def merge_llm_replan_additions(
    baseline: dict[str, Any],
    proposed: dict[str, Any],
    mechanism_ids: list[str],
    event_type: str,
    config: AgentConfig,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Replan: принять полный план или только optional-добавки.

    Промпт просит «добавь только optional» — модель часто возвращает лишь новые шаги.
    Полный план (со всеми required) тоже допустим: из него берутся только additions.
    """
    error = validate_plan(proposed, config)
    if error:
        return [], "", error
    required = {step_key(step["tool"], step["args"]) for step in baseline.get("steps") or [] if step.get("required", True)}
    before = {step_key(step["tool"], dict(step.get("args") or {})) for step in baseline.get("steps") or []}
    proposed_keys = {step_key(step["tool"], dict(step.get("args") or {})) for step in proposed.get("steps") or []}
    full_plan = bool(required) and required <= proposed_keys
    if full_plan:
        merged, merge_error = merge_llm_plan(baseline, proposed, mechanism_ids, event_type, config)
        if merge_error:
            return [], "", merge_error
        additions: list[dict[str, Any]] = []
        for step in merged.get("steps") or []:
            key = step_key(step["tool"], dict(step.get("args") or {}))
            if key in before or step.get("required"):
                continue
            additions.append(step)
        reasoning = str(merged.get("reasoning") or proposed.get("reasoning") or "")
        return additions, reasoning[:500], None

    # Режим additions-only: каждый шаг — optional из каталога, не из baseline.
    allowed = {step_key(step.tool, step.args) for step in optional_catalog(mechanism_ids, event_type, config)}
    additions = []
    for step in proposed.get("steps") or []:
        tool = str(step.get("tool") or "")
        args = dict(step.get("args") or {})
        key = step_key(tool, args)
        if key in before:
            continue
        if key not in allowed:
            return [], "", "шаг не из необязательных"
        if step.get("required") is True:
            return [], "", "в additions-only нельзя помечать шаг как required"
        additions.append(
            {
                "id": str(step.get("id") or f"r{len(additions) + 1}"),
                "tool": tool,
                "args": args,
                "why": str(step.get("why") or reason_why(tool, args, mechanism_ids, event_type))[:240],
                "required": False,
            }
        )
    if len(baseline.get("steps") or []) + len(additions) > config.max_steps:
        return [], "", "слишком много шагов"
    reasoning = str(proposed.get("reasoning") or "")
    return additions, reasoning[:500], None


def gather_observations(state: dict[str, Any]) -> dict[str, Any]:
    """Сжатый итог gather для перепланирования — факты, не промпт-инъекции."""
    evidence = state.get("evidence") or {}
    views = state.get("views") or {}
    profile = views.get("profile") or {}
    features = profile.get("features") or {}
    flags = [str(item) for item in state.get("flags") or []]
    cards = list(evidence.get("cards") or [])
    news = list(evidence.get("news") or [])
    stats = list(evidence.get("stats") or [])
    calcs = list(state.get("calc_results") or [])
    skipped = sorted(item.removeprefix("skipped:") for item in flags if str(item).startswith("skipped:"))
    return {
        "event_type": state.get("event_type"),
        "mechanisms": list(state.get("mechanism_ids") or []),
        "cards": len(cards),
        "news": len(news),
        "stats": len(stats),
        "calcs": len(calcs),
        "has_group_view": bool(views.get("group")),
        "has_counterparties": bool(views.get("counterparties")),
        "has_credit": features.get("has_credit") is True,
        "has_leasing": features.get("has_leasing") is True,
        "fx_exposure": features.get("fx_exposure"),
        "skipped_tools": skipped,
        "stale_cards": sum(1 for card in cards if card.get("stale")),
        "priority": state.get("priority"),
    }


def heuristic_replan_additions(
    plan: dict[str, Any],
    observations: dict[str, Any],
    mechanism_ids: list[str],
    event_type: str,
    config: AgentConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Детерминированные добавки optional-шагов с реальным why по итогам gather."""
    existing = {step_key(step["tool"], dict(step.get("args") or {})) for step in plan.get("steps") or []}
    catalog = {step_key(step.tool, step.args): step for step in optional_catalog(mechanism_ids, event_type, config)}
    additions: list[dict[str, Any]] = []
    reasons: list[str] = []

    def _try(tool: str, args: dict[str, Any], why: str) -> None:
        key = step_key(tool, args)
        if key in existing or key not in catalog:
            return
        if len(plan.get("steps") or []) + len(additions) >= config.max_steps:
            return
        additions.append(
            _public_step(
                f"r{len(additions) + 1}",
                tool,
                args,
                False,
                mechanism_ids=mechanism_ids,
                event_type=event_type,
                why=why,
            )
        )
        reasons.append(why)
        existing.add(key)

    topic = topic_for_event(event_type)
    if int(observations.get("cards") or 0) == 0:
        _try(
            "search_news",
            specialize({"topic": topic}, event_type),
            f"после gather карточек Альфы 0 — взять первичные новости по {topic}",
        )
    if int(observations.get("stale_cards") or 0) > 0 and int(observations.get("news") or 0) == 0:
        _try(
            "search_news",
            specialize({"topic": topic}, event_type),
            f"карточки устарели ({observations.get('stale_cards')}) — обновить контекст новостями",
        )
    indicator = macro_stats_indicator(event_type)
    if indicator and int(observations.get("stats") or 0) == 0:
        _try(
            "external_stats",
            {"indicator": indicator},
            f"макро-событие {event_type} без внешнего ряда — запросить {indicator}",
        )
    if (
        (observations.get("has_credit") or observations.get("has_leasing"))
        and not observations.get("has_group_view")
        and any(item in {"debt_service_cost", "demand_shift", "input_cost"} for item in mechanism_ids)
    ):
        _try(
            "get_group_context",
            {},
            "в профиле есть долг/лизинг и механизм чувствителен к группе — снять агрегат группы",
        )
    if "search_alfa_analytics" in set(observations.get("skipped_tools") or []) and int(observations.get("news") or 0) == 0:
        _try(
            "search_news",
            specialize({"topic": topic}, event_type),
            "search_alfa_analytics был пропущен — запасной поиск новостей",
        )
    return additions, reasons


def apply_replan_additions(plan: dict[str, Any], additions: list[dict[str, Any]], reasoning: str) -> dict[str, Any]:
    if not additions:
        return dict(plan)
    steps = list(plan.get("steps") or [])
    start = len(steps) + 1
    for offset, step in enumerate(additions):
        item = dict(step)
        item["id"] = f"s{start + offset}"
        item["required"] = False
        steps.append(item)
    output = dict(plan)
    output["steps"] = steps
    output["truncated"] = False
    output["reasoning"] = reasoning[:500]
    return output
