"""Source/clarify draft helpers для follow-up чата (без agent-цикла)."""

from __future__ import annotations

import re
from typing import Any

from alfa.core.grounding import insight_sources

_TOKEN = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ%]{3,}")
_NO_SOURCE = "Прямая ссылка на первоисточник в сохранённых данных не найдена."
_CLARIFY_FALLBACK = "На основе уже собранной аналитики по этому событию и компании."


def model_anchor(state: dict) -> dict[str, Any]:
    features = ((state.get("views") or {}).get("profile") or {}).get("features") or {}
    feature = next((name for name, value in features.items() if value is not None), None)
    mechanisms = list(state.get("mechanism_ids") or [])
    if feature:
        return {"feature_refs": [feature], "mechanism_id": mechanisms[0] if mechanisms else None}
    if mechanisms:
        return {"feature_refs": [], "mechanism_id": mechanisms[0]}
    return {"feature_refs": ["industry_group"], "mechanism_id": None}


def model_claim(state: dict, claim_id: str, text: str) -> dict:
    anchor = model_anchor(state)
    return {
        "id": claim_id,
        "layer": "model",
        "text": text,
        "refs": [],
        "feature_refs": anchor["feature_refs"],
        "mechanism_id": anchor["mechanism_id"],
        "calc_refs": [],
    }


def append_claim(draft: dict, claim: dict, section: str) -> None:
    claims = list(draft.get("claims") or [])
    if any(item.get("id") == claim["id"] for item in claims):
        claim = dict(claim)
        claim["id"] = f"{claim['id']}_2"
    claims.append(claim)
    draft["claims"] = claims
    ids = list(draft.get(section) or [])
    if claim["id"] not in ids:
        ids.append(claim["id"])
    draft[section] = ids


def apply_source(state: dict, draft: dict) -> None:
    rows = source_meta(state)
    if not rows:
        append_claim(draft, model_claim(state, "c_source_empty", _NO_SOURCE), "what_happened")
        return
    refs = [str(row["id"]) for row in rows[:5]]
    publishers = sorted(
        {
            str(row["publisher"]).strip()
            for row in rows[:5]
            if row.get("publisher") and str(row["publisher"]).strip()
        }
    )
    # Без цифр в model-тексте: иначе verify требует их в corpus источников.
    if publishers:
        text = f"Вывод опирается на материалы ({', '.join(publishers)}). Ссылки — в блоке sources."
    else:
        text = "Вывод опирается на первоисточники ниже. Ссылки — в блоке sources."
    claim = model_claim(state, "c_source", text)
    claim["refs"] = refs
    append_claim(draft, claim, "what_happened")
    for index, row in enumerate(rows[:5]):
        ref = str(row["id"])
        if not ref.startswith("fact:"):
            continue
        # fact-claim с ref удерживает url/span в insight.sources.
        fact_text = str(row.get("evidence_span") or row.get("title") or ref).strip() or ref
        if len(fact_text) > 240:
            fact_text = fact_text[:237].rstrip() + "…"
        append_claim(
            draft,
            {"id": f"c_source_link_{index}", "layer": "fact", "text": fact_text, "refs": [ref]},
            "what_happened",
        )


def apply_clarify(state: dict, draft: dict, question: str) -> None:
    matched = match_claims(draft, question)
    if matched:
        best = matched[0]
        text = str(best.get("text") or "").strip()
        claim = {
            "id": "c_clarify",
            "layer": "model",
            "text": ("По вашему уточнению: " + text) if text else _CLARIFY_FALLBACK,
            "refs": list(best.get("refs") or []),
            "feature_refs": list(best.get("feature_refs") or []),
            "mechanism_id": best.get("mechanism_id"),
            "calc_refs": list(best.get("calc_refs") or []),
            "evidence": list(best.get("evidence") or []),
        }
        if not claim["feature_refs"] and not claim["mechanism_id"]:
            anchor = model_anchor(state)
            claim["feature_refs"] = anchor["feature_refs"]
            claim["mechanism_id"] = anchor["mechanism_id"]
        append_claim(draft, claim, "why_it_matters")
        return
    summary, refs, calc_refs = clarify_summary(draft)
    claim = model_claim(state, "c_clarify", summary)
    claim["refs"] = refs[:8]
    claim["calc_refs"] = calc_refs[:4]
    append_claim(draft, claim, "why_it_matters")


def match_claims(draft: dict, question: str) -> list[dict]:
    tokens = {token.lower() for token in _TOKEN.findall(question or "")}
    if not tokens:
        return []
    scored: list[tuple[int, dict]] = []
    for claim in draft.get("claims") or []:
        text = str(claim.get("text") or "").lower()
        if not text:
            continue
        score = sum(1 for token in tokens if token in text)
        if score:
            scored.append((score, claim))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("id") or "")))
    return [claim for _score, claim in scored]


def clarify_summary(draft: dict) -> tuple[str, list[str], list[str]]:
    bits: list[str] = []
    used_claims: list[dict] = []
    # Только тексты claims (не headline): числа уже привязаны к refs/calc_refs.
    by_id = {item.get("id"): item for item in draft.get("claims") or []}
    for section in ("what_happened", "why_it_matters", "what_to_watch"):
        for claim_id in draft.get(section) or []:
            claim = by_id.get(claim_id)
            if not claim:
                continue
            text = str(claim.get("text") or "").strip()
            if text and text not in bits:
                bits.append(text)
                used_claims.append(claim)
            if len(bits) >= 3:
                break
        if len(bits) >= 3:
            break
    refs: list[str] = []
    calc_refs: list[str] = []
    for claim in used_claims:
        for ref in claim.get("refs") or []:
            if ref not in refs:
                refs.append(ref)
        for ref in claim.get("calc_refs") or []:
            if ref not in calc_refs:
                calc_refs.append(ref)
    if not used_claims:
        for claim in draft.get("claims") or []:
            if claim.get("layer") != "fact":
                continue
            for ref in claim.get("refs") or []:
                if ref not in refs:
                    refs.append(ref)
    if not bits:
        return _CLARIFY_FALLBACK, refs, calc_refs
    return _CLARIFY_FALLBACK + " " + " ".join(bits[:3]), refs, calc_refs


def source_meta(state: dict) -> list[dict[str, Any]]:
    locators = dict((state.get("evidence") or {}).get("locators") or {})
    if not locators:
        return []
    used: list[str] = []
    for claim in (state.get("draft") or {}).get("claims") or []:
        used.extend(claim.get("refs") or [])
        used.extend(claim.get("evidence") or [])
    if not used:
        used = [
            key
            for key in sorted(locators)
            if str(key).startswith(("fact:", "event:", "card:", "external:")) and "@" not in str(key)
        ]
    rows = insight_sources(locators, used)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        meta = locators.get(row["id"]) or {}
        item = dict(row)
        if not item.get("url"):
            item["url"] = meta.get("url")
        if not item.get("evidence_span"):
            item["evidence_span"] = meta.get("evidence_span")
        item["publisher"] = meta.get("publisher")
        enriched.append(item)
    return [row for row in enriched if row.get("id")]
