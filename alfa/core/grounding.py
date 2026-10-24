"""Отбор, объединение и привязка фактов к первоисточникам.

Факт — атомарное утверждение с evidence_span ⊆ текста источника и локатором
(url + char_start/char_end). Эхо-источники кластера сливаются по совпадающим
числам и датам; расхождения не усредняются.
"""

from __future__ import annotations

import re
from typing import Any

from alfa.core import textcheck
from alfa.core.schemas import Fact

_DECIMAL = re.compile(r"\d+[,.]\d+")
_WS = re.compile(r"\s+")


def locate_span(text: str, span: str) -> tuple[int, int] | None:
    """Позиции evidence_span в исходном тексте. Сначала точное совпадение, затем по нормализованным пробелам."""
    if not span or not text:
        return None
    exact = text.find(span)
    if exact >= 0:
        return exact, exact + len(span)
    needle = textcheck.normalize_ws(span)
    if not needle:
        return None
    hay = textcheck.normalize_ws(text)
    pos = hay.find(needle)
    if pos < 0:
        # Мягкий поиск: первые 8–12 значимых слов.
        words = [w for w in needle.split(" ") if w][:12]
        if len(words) < 3:
            return None
        soft = " ".join(words[:8])
        pos = hay.find(soft)
        if pos < 0:
            return None
        needle = soft
    # Восстанавливаем смещение в исходной строке по префиксу нормализованного текста.
    prefix_norm = hay[:pos]
    approx = 0
    built = ""
    for index, char in enumerate(text):
        approx = index
        built = textcheck.normalize_ws(text[: index + 1])
        if built == prefix_norm or built.startswith(prefix_norm) and len(built) >= len(prefix_norm):
            if built == prefix_norm:
                approx = index + 1
                break
    start = approx
    end = min(len(text), start + len(span))
    # Расширяем end, пока нормализованный фрагмент не покроет needle.
    while end <= len(text) and textcheck.normalize_ws(text[start:end]) != needle:
        if textcheck.normalize_ws(text[start:end]).startswith(needle):
            break
        end += 1
        if end - start > len(span) + 40:
            break
    if not textcheck.span_in_text(text[start:end] if start < end else "", text):
        return start, min(len(text), start + len(span))
    return start, end


def split_sentences(raw: str) -> list[str]:
    protected: list[str] = []

    def hide(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\u0000{len(protected) - 1}\u0000"

    shielded = _DECIMAL.sub(hide, raw or "")
    parts = re.split(r"[.!?]+", shielded)
    output: list[str] = []
    for part in parts:
        text = part
        for index, token in enumerate(protected):
            text = text.replace(f"\u0000{index}\u0000", token)
        text = text.strip(" \n\t;:—-")
        if text:
            output.append(text)
    return output


def _numbers(text: str) -> set[str]:
    return textcheck.numbers_in(text or "")


def _dates(text: str) -> set[tuple[int, int, int]]:
    return textcheck._dates_in_text(text or "")


def _fact_key(claim: str, value: str | None = None, date: str | None = None) -> str:
    nums = sorted(_numbers(value or "") | _numbers(claim))
    dates = sorted(_dates(date or "") | _dates(claim))
    if nums or dates:
        return "n:" + "|".join(nums) + ";d:" + "|".join(f"{y}-{m:02d}-{d:02d}" for y, m, d in dates)
    return "t:" + textcheck.normalize_ws(claim).lower()[:120]


def build_locator(
    *,
    source_id: str,
    title: str,
    url: str | None,
    evidence_span: str,
    text: str,
    source_type: str = "external_news",
    publisher: str | None = None,
    page: int | None = None,
    bbox_rel: list[float] | None = None,
) -> dict[str, Any]:
    start_end = locate_span(text, evidence_span) if evidence_span and text else None
    locator_bits = []
    if publisher:
        locator_bits.append(publisher)
    if start_end is not None:
        locator_bits.append(f"chars {start_end[0]}-{start_end[1]}")
    if page is not None:
        locator_bits.append(f"стр. {page}")
    if bbox_rel:
        locator_bits.append("bbox=" + ",".join(f"{v:.2f}" for v in bbox_rel))
    return {
        "id": source_id,
        "type": source_type,
        "title": title or source_id,
        "url": url or None,
        "locator": "; ".join(locator_bits) if locator_bits else None,
        "evidence_span": evidence_span or "",
        "char_start": start_end[0] if start_end else None,
        "char_end": start_end[1] if start_end else None,
        "publisher": publisher,
        "page": page,
        "bbox_rel": list(bbox_rel) if bbox_rel else None,
    }


def fact_payload(
    *,
    fact_id: str,
    claim: str,
    evidence_span: str,
    text: str,
    url: str | None = None,
    title: str = "",
    publisher: str | None = None,
    value: str | None = None,
    unit: str | None = None,
    date: str | None = None,
    subject: str | None = None,
    unverified: bool | None = None,
    source_ids: list[str] | None = None,
    echo_publishers: list[str] | None = None,
) -> dict[str, Any]:
    span = (evidence_span or "").strip() or (claim or "").strip()
    claim_text = (claim or "").strip()
    if claim_text and not claim_text.endswith("."):
        claim_text += "."
    # evidence_span должен быть подстрокой источника; claim — атомарное утверждение.
    if text and span and not textcheck.span_in_text(span, text):
        # Пробуем взять предложение целиком из текста.
        for sentence in split_sentences(text):
            if textcheck.normalize_ws(claim_text).rstrip(".") in textcheck.normalize_ws(sentence):
                span = sentence
                break
        if not textcheck.span_in_text(span, text):
            span = ""
    start_end = locate_span(text, span) if span and text else None
    if unverified is None:
        unverified = (not span) or (not textcheck.span_in_text(span, text) if text else True)
        if value not in (None, "") and text and not textcheck.number_in_text(value, text):
            unverified = True
        if date not in (None, "") and text and not textcheck.date_in_text(date, text):
            unverified = True
    refs = list(source_ids or [f"fact:{fact_id}"])
    # Валидация через Fact — единый контракт extraction / get_event / ranking.
    return Fact.model_validate(
        {
            "id": fact_id,
            "claim": claim_text,
            "evidence_span": span,
            "value": value,
            "unit": unit,
            "date": date,
            "subject": subject,
            "unverified": bool(unverified),
            "url": url or None,
            "char_start": start_end[0] if start_end else None,
            "char_end": start_end[1] if start_end else None,
            "source_ids": refs,
            "echo_publishers": list(echo_publishers or []),
            "echo_urls": [],
            "title": title,
            "publisher": publisher,
        }
    ).model_dump()


def atomic_facts_from_text(
    text: str,
    *,
    title: str = "",
    url: str | None = None,
    publisher: str | None = None,
    skip_injection=None,
    max_facts: int = 8,
) -> list[dict[str, Any]]:
    """Режет текст на атомарные факты: span ⊆ text, не весь документ как один claim."""
    body = (text or "").strip()
    if not body:
        claim = (title or "Событие получено").strip()
        return [
            fact_payload(
                fact_id="f1",
                claim=claim,
                evidence_span=claim if claim in (title or "") else "",
                text=title or "",
                url=url,
                title=title,
                publisher=publisher,
                unverified=True,
            )
        ]
    sentences = split_sentences(body)
    kept: list[str] = []
    for sentence in sentences:
        if skip_injection is not None and skip_injection(sentence):
            continue
        kept.append(sentence)
    if not kept:
        kept = [title] if title else [body[:240]]
    # Не схлопываем весь текст в один факт, даже если предложений мало.
    facts: list[dict[str, Any]] = []
    for index, sentence in enumerate(kept[:max_facts], start=1):
        # evidence_span — точная подстрока; claim — то же атомарное предложение.
        span = sentence
        if span not in body:
            # Восстанавливаем оригинальный фрагмент через locate.
            located = locate_span(body, span)
            if located:
                span = body[located[0] : located[1]]
        facts.append(
            fact_payload(
                fact_id=f"f{index}",
                claim=sentence,
                evidence_span=span,
                text=body,
                url=url,
                title=title,
                publisher=publisher,
            )
        )
    return facts


def facts_from_extraction(
    extracted: list[Any],
    *,
    text: str,
    title: str = "",
    url: str | None = None,
    publisher: str | None = None,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, item in enumerate(extracted or [], start=1):
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
            span = str(item.get("evidence_span") or "").strip()
            value = item.get("value")
            unit = item.get("unit")
            date = item.get("date")
            subject = item.get("subject")
            unverified = item.get("unverified")
        else:
            claim = str(getattr(item, "claim", "") or "").strip()
            span = str(getattr(item, "evidence_span", "") or "").strip()
            value = getattr(item, "value", None)
            unit = getattr(item, "unit", None)
            date = getattr(item, "date", None)
            subject = getattr(item, "subject", None)
            unverified = getattr(item, "unverified", None)
        if not claim and not span:
            continue
        facts.append(
            fact_payload(
                fact_id=f"f{index}",
                claim=claim or span,
                evidence_span=span or claim,
                text=text,
                url=url,
                title=title,
                publisher=publisher,
                value=None if value in (None, "") else str(value),
                unit=None if unit in (None, "") else str(unit),
                date=None if date in (None, "") else str(date),
                subject=None if subject in (None, "") else str(subject),
                unverified=unverified,
            )
        )
    return facts


def merge_echo_facts(
    primary: list[dict[str, Any]],
    echoes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Сливает эхо с primary по числам/датам.

    Совпадающие факты получают дополнительные source_ids и echo_publishers.
    Расхождения чисел не усредняются — пишутся в disagreements.
    Уникальные маркеры эха в primary не тащим: иначе evidence_span чужого текста
    попадёт в карточку события.
    """
    merged = [dict(item) for item in primary]
    by_key: dict[str, dict[str, Any]] = {}
    for fact in merged:
        by_key[_fact_key(fact.get("claim") or "", fact.get("value"), fact.get("date"))] = fact
    disagreements: list[dict[str, Any]] = []
    for echo in echoes:
        publisher = str(echo.get("publisher") or echo.get("source") or "")
        echo_url = echo.get("url")
        echo_title = str(echo.get("title") or "")
        echo_text = str(echo.get("text") or "")
        echo_facts = echo.get("facts") or atomic_facts_from_text(
            echo_text, title=echo_title, url=echo_url, publisher=publisher
        )
        for raw in echo_facts:
            key = _fact_key(raw.get("claim") or "", raw.get("value"), raw.get("date"))
            # Ключи вида t:… без чисел/дат не считаем подтверждением — слишком шумно.
            if key.startswith("t:"):
                continue
            host = by_key.get(key)
            if host is None:
                continue
            refs = list(host.get("source_ids") or [f"fact:{host['id']}"])
            echo_ref = f"fact:{host['id']}@{publisher or 'echo'}"
            if echo_ref not in refs:
                refs.append(echo_ref)
            host["source_ids"] = refs
            pubs = list(host.get("echo_publishers") or [])
            if publisher and publisher not in pubs:
                pubs.append(publisher)
            host["echo_publishers"] = pubs
            host_nums = _numbers(str(host.get("claim") or "")) | _numbers(str(host.get("value") or ""))
            echo_nums = _numbers(str(raw.get("claim") or "")) | _numbers(str(raw.get("value") or ""))
            if host_nums and echo_nums and host_nums != echo_nums:
                disagreements.append(
                    {
                        "primary_id": host["id"],
                        "echo_publisher": publisher,
                        "primary_numbers": sorted(host_nums),
                        "echo_numbers": sorted(echo_nums),
                        "echo_claim": raw.get("claim"),
                        "echo_url": echo_url,
                    }
                )
            elif echo_url and not host.get("echo_urls"):
                host["echo_urls"] = [echo_url]
            elif echo_url:
                urls = list(host.get("echo_urls") or [])
                if echo_url not in urls:
                    urls.append(echo_url)
                host["echo_urls"] = urls
    return merged, disagreements


def summarize_facts(facts: list[dict[str, Any]], *, max_items: int = 5) -> str:
    """Краткая сводка по отобранным фактам без выдумывания чисел."""
    parts: list[str] = []
    for fact in facts[:max_items]:
        claim = str(fact.get("claim") or "").strip()
        if not claim:
            continue
        pubs = list(fact.get("echo_publishers") or [])
        if pubs:
            claim = claim.rstrip(".") + f" (подтверждено: {', '.join(pubs)})."
        parts.append(claim)
    return " ".join(parts)


def source_text_map(facts: list[dict[str, Any]], echoes: list[dict[str, Any]] | None = None) -> dict[str, str]:
    """id → текст для verify (числа/даты должны находиться в этом тексте)."""
    mapping: dict[str, str] = {}
    for fact in facts:
        fact_id = f"fact:{fact['id']}"
        span = str(fact.get("evidence_span") or fact.get("claim") or "")
        mapping[fact_id] = span
        for ref in fact.get("source_ids") or []:
            if ref != fact_id and ref not in mapping:
                mapping[str(ref)] = span
    for echo in echoes or []:
        event_id = echo.get("event_id")
        if event_id:
            mapping[f"event:{event_id}"] = str(echo.get("text") or echo.get("title") or "")
    return mapping


def locator_map(
    facts: list[dict[str, Any]],
    *,
    event_title: str = "",
    event_url: str | None = None,
    event_text: str = "",
    publisher: str | None = None,
    echoes: list[dict[str, Any]] | None = None,
    cards: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    locators: dict[str, dict[str, Any]] = {}
    for fact in facts:
        fact_id = f"fact:{fact['id']}"
        locators[fact_id] = build_locator(
            source_id=fact_id,
            title=str(fact.get("title") or event_title or fact_id),
            url=fact.get("url") or event_url,
            evidence_span=str(fact.get("evidence_span") or ""),
            text=event_text or str(fact.get("evidence_span") or ""),
            source_type="external_news",
            publisher=fact.get("publisher") or publisher,
        )
        locators[fact_id]["char_start"] = fact.get("char_start")
        locators[fact_id]["char_end"] = fact.get("char_end")
        for ref in fact.get("source_ids") or []:
            if ref not in locators:
                locators[str(ref)] = dict(locators[fact_id])
                locators[str(ref)]["id"] = str(ref)
    for echo in echoes or []:
        event_id = str(echo.get("event_id") or "")
        if not event_id:
            continue
        key = f"event:{event_id}"
        echo_text = str(echo.get("text") or "")
        span = echo_text[:240] if echo_text else str(echo.get("title") or "")
        locators[key] = build_locator(
            source_id=key,
            title=str(echo.get("title") or event_id),
            url=echo.get("url"),
            evidence_span=span,
            text=echo_text or span,
            source_type="external_news",
            publisher=echo.get("publisher") or echo.get("source"),
        )
    for card in cards or []:
        card_id = str(card.get("card_id") or "")
        if not card_id:
            continue
        claim = str(card.get("claim") or "")
        locators[card_id] = build_locator(
            source_id=card_id,
            title=claim[:120] or card_id,
            url=None,
            evidence_span=claim,
            text=claim,
            source_type="bank_analytics",
            publisher="alfa",
            page=card.get("page"),
            bbox_rel=card.get("bbox_rel"),
        )
    return locators


def insight_sources(locators: dict[str, dict[str, Any]], used_refs: list[str]) -> list[dict[str, Any]]:
    """Уникальные источники, на которые ссылается инсайт."""
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in used_refs:
        meta = locators.get(ref)
        if meta is None:
            continue
        key = str(meta.get("id") or ref)
        # Схлопываем fact:f1@rbc → канонический fact:f1 для реестра, но url эха сохраняем отдельно.
        canon = key.split("@", 1)[0]
        if canon in seen and canon.startswith("fact:"):
            continue
        if key in seen:
            continue
        seen.add(canon if canon.startswith("fact:") else key)
        ordered.append(
            {
                "id": canon if canon.startswith(("fact:", "card:", "event:", "calc:", "external:")) else key,
                "type": meta.get("type") or "external_news",
                "title": meta.get("title") or key,
                "locator": meta.get("locator"),
                "url": meta.get("url"),
                "evidence_span": meta.get("evidence_span") or None,
                "char_start": meta.get("char_start"),
                "char_end": meta.get("char_end"),
            }
        )
    return ordered
