"""Проверка черновика: сначала код, затем Critic по спорным местам."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alfa.agent.config import AgentConfig
from alfa.agent.interpret import clip_text, template_draft
from alfa.core import textcheck

_BANK = re.compile(r"альфа-банк|по оценке банка|банк считает|позиция банка", re.IGNORECASE)
_PARTY = re.compile(r"\bпоставщик\w*\b|\bпокупател(?:ь|я|и|ем)\b", re.IGNORECASE)
_ADVICE = re.compile(
    r"рекомендуем\s+(купить|открыть|оформить|продукт)|стоит\s+(купить|продать)|покупайте|продавайте|"
    r"\bоткройте\b|вам\s+подойд[её]т|оформите\s+вклад",
    re.IGNORECASE,
)
_CONTRACT_PHRASE = re.compile(r"имеет\s+смысл\s+уточнить\s+уже\s+открытый\s+договор", re.IGNORECASE)
_BUY_VERB = re.compile(r"купить|продать|покупайте|продавайте|\bоткройте\b|\bоформите\b", re.IGNORECASE)
_PRODUCT_NAME = re.compile(
    r"вклад|депозит|облигац|акци[июя]|ипотек|накопительн|кредитн\w*\s+карт|страхов",
    re.IGNORECASE,
)
_BARE_TOCHNO = re.compile(r"\bточно\b", re.IGNORECASE)
_DIVERGENCE_FEATURES = ("ставк", "курс", "золот")
_CRITIC_TYPES = {
    "ungrounded",
    "unjustified_inference",
    "contradiction",
    "misleading",
    "overclaim",
    "layer_violation",
    "incomplete",
    "impact_inconsistent",
    "paraphrase_only",
    "single_factor",
    "no_consequence",
    "news_only_risks",
    "impact_rationale_boilerplate",
}
_CRITIC_TYPE_LIST = (
    "ungrounded, unjustified_inference, contradiction, misleading, overclaim, "
    "layer_violation, incomplete, impact_inconsistent, paraphrase_only, "
    "single_factor, no_consequence, news_only_risks, impact_rationale_boilerplate"
)
_RATIONALE_BOILERPLATE = re.compile(r"уровень\s+посчитан\s+по\s+рубрике", re.IGNORECASE)
_SCENARIO_STUB = re.compile(r"условный\s+масштаб|масштаб\s+эффекта\s+остаётся\s+неопределённ", re.IGNORECASE)
_ALLOWED_CONTRACT = {"deposit_yield", "debt_service_cost"}
_CATEGORICAL = re.compile(r"\b(точно|обязательно|гарантированно|неизбежно)\b", re.IGNORECASE)
_CUE = re.compile(r"если|при условии|когда|неизвест|не извест|нет данных|вы указали", re.IGNORECASE)
_OVERCLAIM = re.compile(r"однозначно|неизбежно|гарантированно|точно известно", re.IGNORECASE)
_MISLEAD = re.compile(r"можно не учитывать|не обращайте внимания|это неважно", re.IGNORECASE)
_LEAP = re.compile(r"обанкрот|закроется|потеряет рынок", re.IGNORECASE)
_FIELD_TERMS = {
    "debt_rate_type": ("плавающ", "фиксированн"),
    "debt_amount": ("сумма кредита", "размер долга"),
    "debt_maturity": ("срок кредита", "срок долга"),
    "fx_revenue_share": ("доля валютной выручки", "валютная выручка"),
    "fx_expense_share": ("доля валютных расход", "валютные расходы"),
}


def naked_numbers(text: str) -> set[str]:
    stripped = textcheck._ISO_DATE.sub(" ", text or "")
    stripped = textcheck._DOT_DATE.sub(" ", stripped)
    stripped = textcheck._NAMED_DATE.sub(" ", stripped)
    return textcheck.numbers_in(stripped)


def _issue(claim_id: str, kind: str, severity: str, hint: str) -> dict[str, str]:
    return {"claim_id": claim_id, "type": kind, "severity": severity, "hint": hint}


def _empty_consequence_issue(
    items: list | None,
    *,
    cue: str,
    label: str,
    has_mechanisms: bool,
    neutral_ok: bool,
    news_absent_only: bool,
    blob: str,
) -> dict[str, str] | None:
    """Пустые risks/opportunities: early-return таблица решений, без вложенных elif."""
    if items is None:
        return _issue("", "incomplete", "blocking", f"{label} не заданы")
    if items:
        return None
    if not has_mechanisms:
        if cue not in blob:
            return _issue(
                "",
                "incomplete",
                "blocking",
                f"{label} не названы и не отмечены как отсутствующие",
            )
        return None
    if neutral_ok:
        return None
    if news_absent_only or cue not in blob:
        return _issue(
            "",
            "incomplete",
            "blocking",
            f"пустые {label} при механизмах: нужны клиентские формулировки "
            f"или явный нейтральный эффект при текущих признаках",
        )
    # cue есть в assumptions, но нет нейтрального эффекта / news-absent закрытия
    return _issue(
        "",
        "incomplete",
        "blocking",
        f"пустые {label} нельзя закрывать только отсутствием в тексте новости",
    )


def decide(issues: list[dict[str, str]]) -> str:
    if any(item["severity"] == "blocking" for item in issues):
        return "repair"
    if issues:
        return "caveats"
    return "pass"


def _user_map(state: dict) -> dict[str, str]:
    return {str(item["field"]): str(item["value"]) for item in state.get("user_provided") or []}


def _asserts(text: str, terms: tuple[str, ...]) -> bool:
    for sentence in re.split(r"[.!?]+", text or ""):
        lowered = sentence.lower()
        if any(term in lowered for term in terms) and not _CUE.search(sentence):
            return True
    return False


def _sources(state: dict) -> dict[str, str]:
    return dict((state.get("evidence") or {}).get("sources") or {})


def _corpus(claim: dict, sources: dict[str, str]) -> str:
    refs = list(claim.get("refs") or []) + list(claim.get("calc_refs") or []) + list(claim.get("evidence") or [])
    return "\n".join(str(sources[ref]) for ref in refs if ref in sources)


def _number_refs(claim: dict) -> list[str]:
    refs = list(claim.get("refs") or []) + list(claim.get("calc_refs") or []) + list(claim.get("evidence") or [])
    layer = claim.get("layer")
    if layer == "fact":
        return [ref for ref in refs if str(ref).startswith("fact:")]
    if layer == "model":
        return [ref for ref in refs if str(ref).startswith(("fact:", "card:", "calc:"))]
    return refs


def _number_corpus(claim: dict, sources: dict[str, str]) -> str:
    return "\n".join(str(sources[ref]) for ref in _number_refs(claim) if ref in sources)


def _forbidden(text: str, mechanism_id: str | None) -> bool:
    if _PARTY.search(text or ""):
        return True
    if not _ADVICE.search(text or ""):
        return False
    if mechanism_id in _ALLOWED_CONTRACT and _CONTRACT_PHRASE.search(text or ""):
        remainder = _CONTRACT_PHRASE.sub(" ", text or "")
        if not _BUY_VERB.search(remainder) and not _PRODUCT_NAME.search(remainder):
            return False
    return True


def _features_in(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {item for item in _DIVERGENCE_FEATURES if item in lowered}


def verify_stage1(draft: dict | None, state: dict, config: AgentConfig) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if not isinstance(draft, dict) or not isinstance(draft.get("claims"), list):
        return {"decision": "repair", "issues": [_issue("", "schema", "blocking", "черновик не похож на реестр утверждений")]}
    claims = draft["claims"]
    ids = [str(item.get("id") or "") for item in claims]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        issues.append(_issue("", "schema", "blocking", "идентификаторы утверждений"))
    by_id = {str(item.get("id")): item for item in claims}
    for section in ("what_happened", "why_it_matters", "what_to_watch"):
        for claim_id in draft.get(section) or []:
            if claim_id not in by_id:
                issues.append(_issue(str(claim_id), "schema", "blocking", f"{section}: нет утверждения"))
    if draft.get("impact_level") not in {"low", "medium", "high"}:
        issues.append(_issue("", "schema", "blocking", "уровень влияния"))
    if config.disclaimer not in str(draft.get("disclaimer") or ""):
        issues.append(_issue("", "disclaimer", "blocking", "нет дисклеймера"))
    sources = _sources(state)
    features = ((state.get("views") or {}).get("profile") or {}).get("features") or {}
    structural = set(((state.get("views") or {}).get("profile") or {}).get("structurally_missing") or [])
    provided = _user_map(state)
    cards = {item["card_id"]: item for item in (state.get("evidence") or {}).get("cards") or []}
    reliability = str(state.get("reliability") or "none")
    mechanisms = set(state.get("mechanism_ids") or [])
    texts = [str(draft.get("headline") or "")] + list(draft.get("assumptions") or [])
    for claim in claims:
        texts.append(str(claim.get("text") or ""))
    for scenario in draft.get("scenarios") or []:
        texts.append(str(scenario.get("logic") or ""))
        if scenario.get("plausibility") not in {"low", "medium", "high"} or "probability" in scenario:
            issues.append(_issue("", "schema", "blocking", "сценарий без вероятности и с правдоподобностью"))
        for ref in scenario.get("support_refs") or []:
            if ref not in sources:
                issues.append(_issue("", "missing_ref", "blocking", f"сценарий ссылается на {ref}"))
    if len(draft.get("scenarios") or []) > config.max_scenarios:
        issues.append(_issue("", "schema", "blocking", "слишком много сценариев"))
    for claim in claims:
        claim_id = str(claim.get("id"))
        text = str(claim.get("text") or "")
        layer = claim.get("layer")
        if layer not in {"fact", "alfa", "model", "hypothesis"}:
            issues.append(_issue(claim_id, "schema", "blocking", "слой"))
            continue
        for ref in list(claim.get("refs") or []) + list(claim.get("calc_refs") or []) + list(claim.get("evidence") or []):
            if ref not in sources:
                issues.append(_issue(claim_id, "missing_ref", "blocking", f"нет источника {ref}"))
        if layer == "fact" and not any(str(ref).startswith("fact:") for ref in claim.get("refs") or []):
            issues.append(_issue(claim_id, "layer_violation", "blocking", "fact требует ссылку fact:"))
        if layer == "hypothesis":
            evidence_refs = list(claim.get("evidence") or [])
            confidence = claim.get("confidence")
            if not evidence_refs and confidence is not None and float(confidence) > 0.5:
                issues.append(_issue(claim_id, "incomplete", "blocking", "гипотеза без опоры и с уверенностью выше 0.5"))
        corpus = _corpus(claim, sources)
        number_corpus = _number_corpus(claim, sources)
        for number in naked_numbers(text):
            if number not in naked_numbers(number_corpus):
                issues.append(_issue(claim_id, "number", "blocking", f"число {number} не найдено в источнике"))
        for date in textcheck._dates_in_text(text):
            if date not in textcheck._dates_in_text(corpus):
                issues.append(_issue(claim_id, "date", "blocking", "дата не найдена в источнике"))
        if layer == "alfa":
            refs = list(claim.get("refs") or [])
            # card_id в фикстурах уже card:doc:n; старый короткий id — card-key.
            def _card_for(ref: str):
                key = str(ref)
                if key in cards:
                    return cards[key]
                if key.startswith("card:"):
                    return cards.get(key.split(":", 1)[1])
                return None

            if not any(str(ref).startswith("card:") and _card_for(str(ref)) is not None for ref in refs):
                issues.append(_issue(claim_id, "layer_violation", "blocking", "alfa только из карточки"))
            if "альфа" not in text.lower():
                issues.append(_issue(claim_id, "layer_violation", "blocking", "нет атрибуции Альфа-Банку"))
            if not textcheck._dates_in_text(text):
                issues.append(_issue(claim_id, "layer_violation", "blocking", "у карточки нет даты"))
            for ref in refs:
                card = _card_for(str(ref))
                if card and card.get("stale"):
                    issues.append(_issue(claim_id, "stale", "minor", "карточка старше порога"))
        if layer == "model":
            if not claim.get("feature_refs") and not claim.get("mechanism_id"):
                issues.append(_issue(claim_id, "layer_violation", "blocking", "модельному выводу нужен признак или механизм"))
            if _BANK.search(text) and "не позиция банка" not in text.lower():
                issues.append(_issue(claim_id, "layer_violation", "blocking", "модельный вывод нельзя атрибутировать банку"))
            for ref in claim.get("feature_refs") or []:
                if ref not in features:
                    issues.append(_issue(claim_id, "missing_ref", "blocking", f"нет признака {ref}"))
            mechanism_id = claim.get("mechanism_id")
            if mechanism_id and mechanism_id not in mechanisms:
                issues.append(_issue(claim_id, "missing_ref", "blocking", "неизвестный механизм"))
        if _forbidden(text, str(claim.get("mechanism_id") or "") or None):
            issues.append(_issue(claim_id, "forbidden", "blocking", "запрещённая формулировка"))
        if reliability in {"low", "none"} and _CATEGORICAL.search(text):
            issues.append(_issue(claim_id, "categorical", "blocking", "категоричность при низкой надёжности"))
        for field_name, terms in _FIELD_TERMS.items():
            if structural and field_name not in structural:
                continue
            value = provided.get(field_name)
            filled = value not in (None, "", "unknown")
            if not filled and _asserts(text, terms):
                issues.append(_issue(claim_id, "profile_field", "blocking", f"утверждение о {field_name} без ответа пользователя"))
    for claim_id in draft.get("bank_block_ids") or []:
        claim = by_id.get(str(claim_id))
        if claim is None or claim.get("layer") != "alfa":
            issues.append(_issue(str(claim_id), "layer_violation", "blocking", "в блок банка попал чужой слой"))
    covered = {claim.get("mechanism_id") for claim in claims if claim.get("mechanism_id")}
    for mechanism_id in mechanisms:
        if mechanism_id not in covered:
            issues.append(_issue("", "incomplete", "blocking", f"не покрыт механизм {mechanism_id}"))
    blob = " ".join(str(item) for item in draft.get("assumptions") or []).lower()
    has_mechanisms = bool(mechanisms)
    neutral_ok = bool(
        re.search(r"нейтральн\w*\s+эффект|при текущих признак", blob, re.IGNORECASE)
    )
    news_absent_only = bool(
        re.search(
            r"в тексте (события )?не назван|отдельн\w+\s+риск\w*\s+в тексте|отдельн\w+\s+возможност\w*\s+в тексте",
            blob,
            re.IGNORECASE,
        )
    ) and not neutral_ok
    for cue, label, items in (
        ("риск", "риски", draft.get("risks")),
        ("возможност", "возможности", draft.get("opportunities")),
    ):
        issue = _empty_consequence_issue(
            items,
            cue=cue,
            label=label,
            has_mechanisms=has_mechanisms,
            neutral_ok=neutral_ok,
            news_absent_only=news_absent_only,
            blob=blob,
        )
        if issue is not None:
            issues.append(issue)
    if not cards and not any(claim.get("layer") == "alfa" for claim in claims):
        issues.append(_issue("", "no_analytics", "minor", "нет карточки Альфа-Банка по теме"))
    for number in naked_numbers("\n".join(texts[:1] + list(draft.get("assumptions") or []))):
        # заголовок и допущения не должны приносить новые числа
        if number not in naked_numbers("\n".join(str(sources[key]) for key in sources)):
            issues.append(_issue("", "number", "blocking", f"число {number} в заголовке или допущении"))
    return {"decision": decide(issues), "issues": issues}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[а-яёa-z0-9]+", (text or "").lower()) if len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _fact_surfaces(state: dict) -> list[str]:
    evidence = state.get("evidence") or {}
    event = evidence.get("event") or {}
    surfaces: list[str] = []
    title = str(event.get("title") or "").strip()
    if title:
        surfaces.append(title)
    for fact in event.get("facts") or []:
        claim = str(fact.get("claim") or "").strip()
        if claim:
            surfaces.append(claim)
    return surfaces


def _is_paraphrase_of_fact(text: str, state: dict) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    claim_tokens = _tokens(lowered)
    for raw in _fact_surfaces(state):
        raw_l = raw.lower().strip()
        if not raw_l:
            continue
        surface = _tokens(raw_l)
        overlap = _jaccard(claim_tokens, surface)
        if overlap >= 0.85:
            return True
        if overlap >= 0.7 and (lowered == raw_l or lowered in raw_l or raw_l in lowered):
            return True
    return False


def _norm_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _extraction_risks(state: dict) -> set[str]:
    event = (state.get("evidence") or {}).get("event") or {}
    return {_norm_phrase(str(item)) for item in event.get("risks") or [] if _norm_phrase(str(item))}


def _draft_mechanisms(draft: dict) -> set[str]:
    return {
        str(claim.get("mechanism_id"))
        for claim in draft.get("claims") or []
        if claim.get("mechanism_id")
    }


def _has_consequence_scenario(draft: dict) -> bool:
    for scenario in draft.get("scenarios") or []:
        logic = str(scenario.get("logic") or "").strip()
        if len(logic) < 24:
            continue
        if _SCENARIO_STUB.search(logic) and len(logic) < 90:
            continue
        return True
    return False


def _is_boilerplate_rationale(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if not _RATIONALE_BOILERPLATE.search(lowered):
        return False
    if "ключевые фактор" in lowered:
        return False
    if "не хватает данных" in lowered:
        return False
    cleaned = _RATIONALE_BOILERPLATE.sub(" ", lowered)
    cleaned = re.sub(r"[.\s]+", " ", cleaned).strip()
    # осталась только дефиниция рубрики / пусто — без факторов профиля
    return len(_tokens(cleaned)) <= 12


def _has_profile_or_calc(state: dict) -> bool:
    features = ((state.get("views") or {}).get("profile") or {}).get("features") or {}
    return bool(features) or bool(state.get("calc_results"))


def heuristic_critic(draft: dict, state: dict) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    level = str(draft.get("impact_level") or "")
    rationale = str(draft.get("impact_rationale") or "")
    rationale_l = rationale.lower()
    high_words = ("высокое влияние", "влияние высокое", "критично высок")
    low_words = ("влияние незначительное", "влияние низкое", "можно игнорировать")
    if level == "low" and any(word in rationale_l for word in high_words):
        issues.append(_issue("impact", "impact_inconsistent", "blocking", "текст обоснования выше уровня рубрики"))
    if level == "high" and any(word in rationale_l for word in low_words):
        issues.append(_issue("impact", "impact_inconsistent", "blocking", "текст обоснования ниже уровня рубрики"))
    if level == "medium" and any(word in rationale_l for word in high_words + low_words):
        issues.append(_issue("impact", "impact_inconsistent", "blocking", "текст обоснования не совпадает со средним уровнем"))
    if _is_boilerplate_rationale(rationale):
        issues.append(
            _issue(
                "impact",
                "impact_rationale_boilerplate",
                "blocking",
                "impact_rationale только «Уровень посчитан по рубрике» без факторов",
            )
        )
    why_ids = {str(item) for item in draft.get("why_it_matters") or []}
    for claim in draft.get("claims") or []:
        text = str(claim.get("text") or "")
        claim_id = str(claim.get("id") or "")
        if _OVERCLAIM.search(text) or (
            str(state.get("reliability") or "") == "high" and _BARE_TOCHNO.search(text) and not _OVERCLAIM.search(text)
        ):
            issues.append(_issue(claim_id, "overclaim", "blocking", "слишком сильная формулировка"))
        if _MISLEAD.search(text):
            issues.append(_issue(claim_id, "misleading", "blocking", "формулировка приуменьшает эффект"))
        if _LEAP.search(text):
            issues.append(_issue(claim_id, "unjustified_inference", "blocking", "вывод не следует из источников"))
        if claim.get("layer") == "model" and claim_id in why_ids and _is_paraphrase_of_fact(text, state):
            issues.append(_issue(claim_id, "paraphrase_only", "blocking", "модельный вывод только пересказывает факт"))
    state_mechs = {str(item) for item in state.get("mechanism_ids") or [] if item}
    if len(state_mechs) >= 2 and len(_draft_mechanisms(draft) & state_mechs) < 2:
        issues.append(
            _issue(
                "",
                "single_factor",
                "blocking",
                "в draft один механизм при len(mechanism_ids) >= 2",
            )
        )
    risks = [str(item).strip() for item in draft.get("risks") or [] if str(item).strip()]
    opps = [str(item).strip() for item in draft.get("opportunities") or [] if str(item).strip()]
    if not risks and not opps and not _has_consequence_scenario(draft):
        issues.append(
            _issue(
                "",
                "no_consequence",
                "blocking",
                "нет риска, возможности и сценария с логикой последствий",
            )
        )
    extraction = _extraction_risks(state)
    if risks and extraction and _has_profile_or_calc(state):
        draft_risk_set = {_norm_phrase(item) for item in risks}
        if draft_risk_set and draft_risk_set <= extraction:
            severity = "blocking" if level == "high" else "minor"
            issues.append(
                _issue(
                    "",
                    "news_only_risks",
                    severity,
                    "risks ⊆ extraction risks при наличии profile features / calc",
                )
            )
    return issues


def should_call_critic(state: dict, draft: dict, stage1: dict, config: AgentConfig, *, force: bool = False) -> bool:
    if force or state.get("force_critic"):
        return True
    if stage1.get("issues"):
        return True
    if draft.get("impact_level") == "high":
        return True
    if draft.get("diverges_from_bank_view"):
        return True
    if state.get("suspicious"):
        return True
    trace_id = str(state.get("trace_id") or "0")
    sample = int(trace_id[:8], 16) if len(trace_id) >= 8 else 0
    return (sample % 1000) < int(config.monitor_rate * 1000)


def run_critic(draft: dict, state: dict, config: AgentConfig, llm: Any, *, llm_enabled: bool) -> dict[str, Any]:
    if llm_enabled and llm is not None:
        class CriticIssue(BaseModel):
            model_config = ConfigDict(extra="forbid")

            claim_id: str = ""
            type: str = Field(description=_CRITIC_TYPE_LIST)
            severity: str = "minor"
            hint: str = ""

        class CriticOut(BaseModel):
            model_config = ConfigDict(extra="forbid")

            issues: list[CriticIssue] = Field(default_factory=list)

        path = config.prompt_path("critic")
        system = path.read_text(encoding="utf-8") if path.exists() else "Проверь реестр утверждений."
        sources = (state.get("evidence") or {}).get("sources") or {}
        compact = [{"id": key, "text": clip_text(str(value), 500)} for key, value in sources.items()]
        registry = {
            "headline": draft.get("headline"),
            "claims": draft.get("claims"),
            "what_happened": draft.get("what_happened"),
            "why_it_matters": draft.get("why_it_matters"),
            "what_to_watch": draft.get("what_to_watch"),
            "impact_level": draft.get("impact_level"),
            "impact_rationale": draft.get("impact_rationale"),
            "assumptions": draft.get("assumptions"),
            "risks": draft.get("risks"),
            "opportunities": draft.get("opportunities"),
            "scenarios": draft.get("scenarios"),
        }
        result = llm.complete(
            role=config.roles.get("critic", "critic"),
            model="",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"registry": registry, "sources": compact}, ensure_ascii=False, default=str)},
            ],
            response_schema=CriticOut,
            temperature=0.0,
            trace_id=state.get("trace_id"),
            prompt_version=config.prompt_versions.get("critic"),
        )
        if result.ok and result.parsed is not None and not isinstance(result.parsed, str):
            parsed = result.parsed
            raw_issues = parsed.issues if hasattr(parsed, "issues") else parsed.get("issues", [])
            issues = []
            for item in raw_issues:
                payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
                kind = str(payload.get("type") or "ungrounded")
                if kind not in _CRITIC_TYPES:
                    severity = "minor"
                else:
                    severity = payload.get("severity") if payload.get("severity") in {"minor", "blocking"} else "blocking"
                issues.append(_issue(str(payload.get("claim_id") or ""), kind, severity, str(payload.get("hint") or "")))
            return {"issues": issues, "backend": "llm"}
        return {"issues": [], "backend": "unavailable", "error": result.error}
    return {"issues": heuristic_critic(draft, state), "backend": "heuristic"}


def critic_mandatory(state: dict, draft: dict) -> bool:
    flags = set(state.get("flags") or [])
    return bool(
        draft.get("impact_level") == "high"
        or draft.get("diverges_from_bank_view")
        or state.get("suspicious")
        or "suspicious_instructions" in flags
    )


def apply_critic_status(flags: list[str], draft: dict, state: dict, stage2: dict, decision: str) -> tuple[list[str], str]:
    output = list(flags)
    if stage2.get("backend") == "unavailable" and critic_mandatory(state, draft):
        if "critic_unavailable" not in output:
            output.append("critic_unavailable")
        if decision == "pass":
            decision = "caveats"
    return output, decision


def bank_divergences(draft: dict, state: dict) -> list[dict[str, Any]]:
    """Разные числа по одной теме не усредняются. Обе формулировки остаются."""
    found: list[dict[str, Any]] = []
    claims = list(draft.get("claims") or [])
    models = [item for item in claims if item.get("layer") == "model"]
    alfas = [item for item in claims if item.get("layer") == "alfa"]
    for model in models:
        model_features = _features_in(str(model.get("text") or ""))
        model_numbers = naked_numbers(str(model.get("text") or ""))
        if not model_features or not model_numbers:
            continue
        for alfa in alfas:
            shared = model_features & _features_in(str(alfa.get("text") or ""))
            alfa_numbers = naked_numbers(str(alfa.get("text") or ""))
            if shared and alfa_numbers and model_numbers != alfa_numbers:
                found.append(
                    {
                        "feature": sorted(shared)[0],
                        "kind": "model_vs_alfa",
                        "positions": [
                            {"id": model.get("id"), "numbers": sorted(model_numbers)},
                            {"id": alfa.get("id"), "numbers": sorted(alfa_numbers)},
                        ],
                    }
                )
    sources = _sources(state)
    facts = {key: value for key, value in sources.items() if str(key).startswith("fact:")}
    cards = {key: value for key, value in sources.items() if str(key).startswith("card:")}
    for fact_id, fact_text in facts.items():
        fact_features = _features_in(str(fact_text))
        fact_numbers = naked_numbers(str(fact_text))
        if not fact_features or not fact_numbers:
            continue
        for card_id, card_text in cards.items():
            shared = fact_features & _features_in(str(card_text))
            card_numbers = naked_numbers(str(card_text))
            if shared and card_numbers and fact_numbers != card_numbers:
                found.append(
                    {
                        "feature": sorted(shared)[0],
                        "kind": "fact_vs_card",
                        "positions": [
                            {"id": fact_id, "numbers": sorted(fact_numbers)},
                            {"id": card_id, "numbers": sorted(card_numbers)},
                        ],
                    }
                )
    return found


def merge_replacements(draft: dict, replacements: list[Any], sources: dict[str, str]) -> tuple[dict, bool, bool]:
    """Подмена только тех id, у которых слой совпал. Новый механизм и чужие ссылки отбрасываются."""
    output = copy.deepcopy(draft)
    by_id = {claim.get("id"): claim for claim in output.get("claims") or []}
    accepted = 0
    unchanged = 0
    for raw in replacements or []:
        repl = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        original = by_id.get(repl.get("id"))
        if original is None or repl.get("layer") != original.get("layer"):
            continue
        repl_mechanism = repl.get("mechanism_id")
        if repl_mechanism and repl_mechanism != original.get("mechanism_id"):
            continue

        def known(refs: list[str] | None) -> list[str]:
            return [ref for ref in (refs or []) if ref in sources]

        new_text = str(repl.get("text") if repl.get("text") is not None else original.get("text") or "")
        new_refs = known(repl.get("refs"))
        new_calc = known(repl.get("calc_refs"))
        new_evidence = known(repl.get("evidence"))
        same = (
            new_text == str(original.get("text") or "")
            and new_refs == list(original.get("refs") or [])
            and new_calc == list(original.get("calc_refs") or [])
            and new_evidence == list(original.get("evidence") or [])
        )
        original["text"] = new_text
        original["refs"] = new_refs
        original["calc_refs"] = new_calc
        original["evidence"] = new_evidence
        original["mechanism_id"] = original.get("mechanism_id")
        accepted += 1
        unchanged += int(same)
    return output, accepted > 0, accepted > 0 and unchanged == accepted


def settle_blocking(draft: dict, issues: list[dict[str, str]]) -> dict:
    """После двух проходов блокирующее утверждение удаляется или становится слабой гипотезой."""
    blocking = {
        item["claim_id"]
        for item in issues
        if item.get("severity") == "blocking" and item.get("claim_id") not in {"", "impact"}
    }
    output = copy.deepcopy(draft)
    kept: list[dict] = []
    converted: list[str] = []
    removed: set[str] = set()
    for claim in output.get("claims") or []:
        claim_id = str(claim.get("id") or "")
        if claim_id not in blocking:
            kept.append(claim)
            continue
        text = str(claim.get("text") or "")
        if naked_numbers(text) or textcheck._dates_in_text(text):
            removed.add(claim_id)
            continue
        converted_claim = dict(claim)
        converted_claim["layer"] = "hypothesis"
        raw_confidence = converted_claim.get("confidence")
        converted_claim["confidence"] = min(0.3, float(raw_confidence) if raw_confidence is not None else 0.3)
        converted_claim["refs"] = []
        converted_claim["calc_refs"] = []
        converted_claim["evidence"] = []
        kept.append(converted_claim)
        converted.append(claim_id)
        removed.add(claim_id)
    output["claims"] = kept

    def scrub(ids: list[str] | None) -> list[str]:
        return [item for item in (ids or []) if item not in removed]

    output["what_happened"] = scrub(output.get("what_happened"))
    output["why_it_matters"] = scrub(output.get("why_it_matters"))
    output["what_to_watch"] = scrub(output.get("what_to_watch"))
    for claim_id in converted:
        if claim_id not in output["what_to_watch"]:
            output["what_to_watch"].append(claim_id)
    output["bank_block_ids"] = [claim["id"] for claim in kept if claim.get("layer") == "alfa"]
    output["repair_dropped"] = sorted(removed - set(converted))
    return output


def apply_repair(draft: dict, issues: list[dict[str, str]], state: dict, config: AgentConfig) -> dict[str, Any]:
    blocking_ids = {
        item["claim_id"]
        for item in issues
        if item["severity"] == "blocking" and item["claim_id"] not in {"", "impact"}
    }
    kept = [claim for claim in draft.get("claims") or [] if claim.get("id") not in blocking_ids]
    safe = template_draft(state, config)

    def layer(items: list[dict], name: str) -> list[dict]:
        return [item for item in items if item.get("layer") == name]

    if not layer(kept, "fact"):
        kept.extend(layer(safe["claims"], "fact"))
    if not layer(kept, "alfa"):
        kept.extend(layer(safe["claims"], "alfa"))
    have = {item.get("mechanism_id") for item in kept if item.get("layer") == "model"}
    for claim in safe["claims"]:
        if claim.get("layer") == "model" and claim.get("mechanism_id") not in have:
            kept.append(claim)
            have.add(claim.get("mechanism_id"))
    if not layer(kept, "hypothesis"):
        kept.extend(layer(safe["claims"], "hypothesis"))
    force_safe_rationale = any(
        item["type"] in {"impact_inconsistent", "impact_rationale_boilerplate"} for item in issues
    )
    rationale = (
        safe["impact_rationale"]
        if force_safe_rationale
        else draft.get("impact_rationale") or safe["impact_rationale"]
    )
    # safe уже несёт клиентские risks/opportunities/scenarios — при no_consequence /
    # news_only_risks / single_factor код-repair подменяет последствия шаблоном, не grounding.
    rebuilt = dict(safe)
    rebuilt.update(
        {
            "claims": kept,
            "impact_rationale": rationale,
            "impact_level": safe["impact_level"],
            "impact_definition": safe["impact_definition"],
            "what_happened": [item["id"] for item in kept if item.get("layer") == "fact"],
            "why_it_matters": [item["id"] for item in kept if item.get("layer") in {"model", "alfa"}],
            "what_to_watch": [item["id"] for item in kept if item.get("layer") == "hypothesis"],
            "bank_block_ids": [item["id"] for item in kept if item.get("layer") == "alfa"],
            "repair_dropped": sorted(blocking_ids),
        }
    )
    return rebuilt


def contradictions(cards: list[dict]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict]] = {}
    for card in cards:
        metric = card.get("metric")
        if metric and card.get("value") is not None:
            grouped.setdefault(str(metric), []).append(card)
    found = []
    for metric, items in grouped.items():
        values = {str(item.get("value")) for item in items}
        if len(values) > 1:
            found.append(
                {
                    "metric": metric,
                    "positions": [{"card_id": item["card_id"], "value": item.get("value")} for item in items],
                }
            )
    return found
