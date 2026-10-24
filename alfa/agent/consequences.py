"""Клиентские последствия: risks/opportunities/watches и apply к draft.

Шаблоны — consequence_templates; сценарии — consequence_scenarios.
Extraction risks — только evidence.
"""

from __future__ import annotations

from alfa.agent.config import AgentConfig
from alfa.agent.consequence_scenarios import build_scenarios, fact_refs
from alfa.agent.consequence_templates import NEUTRAL, rendered_for_mechanisms

# Re-export for callers that import scenarios/templates via consequences.
__all__ = [
    "NEUTRAL",
    "apply_company_consequences",
    "build_company_opportunities",
    "build_company_risks",
    "build_company_watches",
    "build_scenarios",
    "consequence_assumptions",
    "consequences_pack_section",
]
from alfa.agent.state_view import mechanism_ids as state_mechanism_ids
from alfa.ranking.config import RankingConfig, load_ranking_config


def _calc_scale_phrases(state: dict) -> list[str]:
    phrases: list[str] = []
    for item in state.get("calc_results") or []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if item.get("illustrative"):
            phrases.append(f"Масштаб (иллюстрация, не сумма компании): {text}.")
        else:
            phrases.append(f"Масштаб по расчёту: {text}.")
    return phrases


def _context_soften_or_amplify(state: dict) -> tuple[list[str], list[str]]:
    amplify: list[str] = []
    soften: list[str] = []
    evidence = state.get("evidence") or {}
    cards = list(evidence.get("cards") or [])
    if cards:
        amplify.append("Карточка Альфа-Банка по теме усиливает внимание к этому контуру риска.")
        if any(card.get("stale") for card in cards):
            soften.append("Есть устаревшая карточка: вывод по банку осторожнее, риск профиля сохраняется.")
    if evidence.get("news"):
        amplify.append("Смежные новости подтверждают фон события и не отменяют клиентский контур.")
    stats = list(evidence.get("stats") or [])
    if stats:
        indicator = stats[0].get("indicator") or "показатель"
        amplify.append(f"Внешняя статистика ({indicator}) усиливает макрофон для оценки последствий.")
    views = state.get("views") or {}
    if views.get("group") or views.get("counterparties"):
        soften.append(
            "Группа/платёжное окружение может смягчить или перераспределить эффект — "
            "смотреть связи, не только головную компанию."
        )
    return amplify, soften


def _extraction_evidence_notes(state: dict) -> list[str]:
    event = (state.get("evidence") or {}).get("event") or {}
    notes: list[str] = []
    risks = [str(item).strip() for item in event.get("risks") or [] if str(item).strip()]
    opps = [str(item).strip() for item in event.get("opportunities") or [] if str(item).strip()]
    if risks:
        notes.append("Evidence из текста события (риски, не ответ клиенту): " + "; ".join(risks[:3]))
    else:
        notes.append("В тексте события отдельные риски не названы (это не отменяет клиентский контур).")
    if opps:
        notes.append("Evidence из текста события (возможности, не ответ клиенту): " + "; ".join(opps[:3]))
    else:
        notes.append("В тексте события отдельные возможности не названы (это не отменяет клиентский контур).")
    return notes


def _dedupe(items: list[str], *, limit: int = 6) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = " ".join(str(item).split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _lists_from_blocks(
    state: dict,
    blocks: dict[str, dict[str, str | None]],
) -> tuple[list[str], list[str], list[str]]:
    risks: list[str] = []
    opps: list[str] = []
    watches: list[str] = []
    for mechanism_id in state_mechanism_ids(state):
        block = blocks.get(mechanism_id) or {}
        risk = block.get("risk_template")
        if risk:
            risks.append(risk)
        opp = block.get("opportunity_template")
        if opp:
            opps.append(opp)
        watch = block.get("watch_template")
        if watch:
            watches.append(watch)
    scale = _calc_scale_phrases(state)
    amplify, soften = _context_soften_or_amplify(state)
    risks.extend(scale)
    risks.extend(amplify)
    risks = _dedupe(risks)
    if state_mechanism_ids(state) and not risks:
        risks = [f"{NEUTRAL}: сработавший механизм не даёт отдельного негативного контура при текущих данных."]
    opps.extend(soften)
    if opps:
        for phrase in scale:
            if "иллюстрация" not in phrase.lower():
                opps.append(phrase)
    opps = _dedupe(opps)
    watches = _dedupe(watches, limit=4)
    return risks, opps, watches


def build_company_risks(
    state: dict,
    config: AgentConfig,
    *,
    ranking: RankingConfig | None = None,
    blocks: dict[str, dict[str, str | None]] | None = None,
) -> list[str]:
    if not state_mechanism_ids(state):
        return []
    ranking = ranking or load_ranking_config()
    rendered = blocks if blocks is not None else rendered_for_mechanisms(state, config, ranking=ranking)
    risks, _opps, _watches = _lists_from_blocks(state, rendered)
    return risks


def build_company_opportunities(
    state: dict,
    config: AgentConfig,
    *,
    ranking: RankingConfig | None = None,
    blocks: dict[str, dict[str, str | None]] | None = None,
) -> list[str]:
    if not state_mechanism_ids(state):
        return []
    ranking = ranking or load_ranking_config()
    rendered = blocks if blocks is not None else rendered_for_mechanisms(state, config, ranking=ranking)
    _risks, opps, _watches = _lists_from_blocks(state, rendered)
    return opps


def build_company_watches(
    state: dict,
    config: AgentConfig,
    *,
    ranking: RankingConfig | None = None,
    blocks: dict[str, dict[str, str | None]] | None = None,
) -> list[str]:
    if not state_mechanism_ids(state):
        return []
    ranking = ranking or load_ranking_config()
    rendered = blocks if blocks is not None else rendered_for_mechanisms(state, config, ranking=ranking)
    _risks, _opps, watches = _lists_from_blocks(state, rendered)
    return watches


def consequence_assumptions(
    state: dict,
    config: AgentConfig,
    *,
    ranking: RankingConfig | None = None,
    opportunities: list[str] | None = None,
) -> list[str]:
    notes = _extraction_evidence_notes(state)
    opps = (
        opportunities
        if opportunities is not None
        else build_company_opportunities(state, config, ranking=ranking)
    )
    if state_mechanism_ids(state) and not opps:
        notes.append(f"{NEUTRAL}: отдельных клиентских возможностей при текущих признаках нет.")
    return notes


def apply_company_consequences(draft: dict, state: dict, config: AgentConfig) -> dict:
    """Подставить клиентские risks/opportunities/scenarios (template и LLM fallback)."""
    output = dict(draft)
    mechanisms = state_mechanism_ids(state)
    ranking = load_ranking_config()
    blocks = rendered_for_mechanisms(state, config, ranking=ranking)
    company_risks, company_opps, company_watches = _lists_from_blocks(state, blocks)
    company_scenarios = build_scenarios(state, config, fact_refs(state), ranking=ranking)

    if mechanisms:
        if not output.get("risks"):
            output["risks"] = list(company_risks)
        if output.get("opportunities") is None or output.get("opportunities") == []:
            output["opportunities"] = list(company_opps)
        scenarios = list(output.get("scenarios") or [])
        if len(scenarios) < 2:
            output["scenarios"] = company_scenarios
        if company_watches:
            for claim in output.get("claims") or []:
                if claim.get("id") == "c_hyp" and claim.get("layer") == "hypothesis":
                    claim["text"] = company_watches[0]
                    break
    else:
        output.setdefault("risks", [])
        output.setdefault("opportunities", [])
        if not output.get("scenarios"):
            output["scenarios"] = company_scenarios

    assumptions = list(output.get("assumptions") or [])
    for note in consequence_assumptions(state, config, opportunities=company_opps):
        if note not in assumptions:
            assumptions.append(note)
    cleaned: list[str] = []
    drop_markers = (
        "отдельные риски в тексте события не названы",
        "отдельные возможности в тексте события не названы",
    )
    for item in assumptions:
        lowered = item.lower()
        if mechanisms and any(marker in lowered for marker in drop_markers):
            if "не отменяет клиентский" not in lowered and "evidence" not in lowered:
                continue
        cleaned.append(item)
    output["assumptions"] = cleaned
    return output


def consequences_pack_section(state: dict, config: AgentConfig) -> str:
    ranking = load_ranking_config()
    blocks = rendered_for_mechanisms(state, config, ranking=ranking)
    risks, opps, watches = _lists_from_blocks(state, blocks)
    scenarios = build_scenarios(state, config, ranking=ranking)
    lines = [
        "Клиентские последствия (черновик кода из mechanisms/consequences.yaml, не пересказ новости):",
        "риски: " + ("; ".join(risks) if risks else "—"),
        "возможности: " + ("; ".join(opps) if opps else "—"),
        "на что смотреть: " + ("; ".join(watches) if watches else "—"),
        "сценарии:",
    ]
    for item in scenarios:
        lines.append(f"- {item.get('name')}: {item.get('logic')}")
    lines.append(
        "Extraction risks/opportunities из новости — только evidence в assumptions, "
        "не единственный ответ в risks/opportunities. "
        f"Пустые массивы при сработавших механизмах недопустимы без явного «{NEUTRAL}»."
    )
    return "\n".join(lines)
