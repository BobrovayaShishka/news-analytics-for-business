"""Проверка, что цифры, даты и evidence_span реально есть в тексте новости.

Промах не выкидывает факт: он помечается unverified и попадает в validation.*.
"""

from __future__ import annotations

import re
from datetime import datetime

from alfa.core.schemas import EventExtraction, ExtractionStatus, Fact

_WS = re.compile(r"\s+")
_NUMBER = re.compile(
    r"(?<![\d])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?%?",
)
_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
_NAMED_DATE = re.compile(
    r"(?<!\d)(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(20\d{2})(?!\d)",
    flags=re.IGNORECASE,
)
_DOT_DATE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](20\d{2})(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)")


def normalize_ws(text: str) -> str:
    cleaned = (
        text.replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
        .replace("\u2007", " ")
    )
    return _WS.sub(" ", cleaned).strip()


def _canonical_number(token: str) -> str | None:
    raw = (
        token.replace("\u00a0", "")
        .replace("\u202f", "")
        .replace(" ", "")
        .replace("%", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    raw = raw.replace(",", ".")
    if raw.startswith("+"):
        raw = raw[1:]
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return None
    if "." in raw:
        whole, frac = raw.split(".", 1)
        frac = frac.rstrip("0")
        sign = "-" if whole.startswith("-") else ""
        whole_i = abs(int(whole))
        if not frac:
            return f"{sign}{whole_i}"
        return f"{sign}{whole_i}.{frac}"
    return str(int(raw))


def _numbers_in(text: str) -> set[str]:
    found: set[str] = set()
    for match in _NUMBER.finditer(normalize_ws(text)):
        canon = _canonical_number(match.group(0))
        if canon is not None:
            found.add(canon)
    return found


def _number_pieces(value: str) -> list[str]:
    pieces: list[str] = []
    for match in _NUMBER.finditer(normalize_ws(value)):
        canon = _canonical_number(match.group(0))
        if canon is not None:
            pieces.append(canon)
    return pieces


def number_in_text(value: str | None, text: str) -> bool:
    if value is None or str(value).strip() == "":
        return True
    pieces = _number_pieces(str(value))
    if not pieces:
        return normalize_ws(str(value)).lower() in normalize_ws(text).lower()
    present = _numbers_in(text)
    return all(piece in present for piece in pieces)


def _parse_date(value: str) -> datetime | None:
    text = normalize_ws(value)
    iso = _ISO_DATE.search(text)
    if iso:
        return datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    dotted = _DOT_DATE.search(text)
    if dotted:
        return datetime(int(dotted.group(3)), int(dotted.group(2)), int(dotted.group(1)))
    named = _NAMED_DATE.search(text.lower())
    if named:
        return datetime(int(named.group(3)), _MONTHS[named.group(2).lower()], int(named.group(1)))
    if re.fullmatch(r"20\d{2}", text):
        return datetime(int(text), 1, 1)
    return None


def _dates_in_text(text: str) -> set[tuple[int, int, int]]:
    found: set[tuple[int, int, int]] = set()
    normalized = normalize_ws(text)
    for match in _ISO_DATE.finditer(normalized):
        found.add((int(match.group(1)), int(match.group(2)), int(match.group(3))))
    for match in _DOT_DATE.finditer(normalized):
        found.add((int(match.group(3)), int(match.group(2)), int(match.group(1))))
    for match in _NAMED_DATE.finditer(normalized.lower()):
        found.add(
            (int(match.group(3)), _MONTHS[match.group(2).lower()], int(match.group(1)))
        )
    return found


def _years_in_text(text: str) -> set[int]:
    return {int(item) for item in re.findall(r"(?<!\d)(20\d{2})(?!\d)", normalize_ws(text))}


def date_in_text(value: str | None, text: str) -> bool:
    if value is None or str(value).strip() == "":
        return True
    raw = normalize_ws(str(value))
    parsed = _parse_date(raw)
    if parsed is None:
        return raw.lower() in normalize_ws(text).lower()
    if re.fullmatch(r"20\d{2}", raw):
        return parsed.year in _years_in_text(text)
    return (parsed.year, parsed.month, parsed.day) in _dates_in_text(text)


def span_in_text(span: str | None, text: str) -> bool:
    if span is None or str(span).strip() == "":
        return False
    return normalize_ws(span) in normalize_ws(text)


def apply_textcheck(extraction: EventExtraction, text: str) -> EventExtraction:
    """Помечает непроверенные факты. Ничего не удаляет и не подставляет."""
    validation_numbers: list[str] = []
    validation_dates: list[str] = []
    missing_spans: list[str] = []
    facts: list[Fact] = []
    for fact in extraction.facts:
        missing = not span_in_text(fact.evidence_span, text)
        bad_number = fact.value not in (None, "") and not number_in_text(fact.value, text)
        bad_date = fact.date not in (None, "") and not date_in_text(fact.date, text)
        if missing:
            missing_spans.append(fact.evidence_span or fact.claim)
        if bad_number:
            validation_numbers.append(str(fact.value))
        if bad_date:
            validation_dates.append(str(fact.date))
        facts.append(fact.model_copy(update={"unverified": missing or bad_number or bad_date}))

    for note in list(extraction.risks) + list(extraction.opportunities):
        if note.evidence_span and not span_in_text(note.evidence_span, text):
            missing_spans.append(note.evidence_span)

    status = extraction.extraction_status
    if str(getattr(status, "value", status)) == "ok" and (
        extraction.truncated or any(item.unverified for item in facts) or missing_spans
    ):
        status = ExtractionStatus("partial")

    return extraction.model_copy(
        update={
            "facts": facts,
            "validation": extraction.validation.model_copy(
                update={
                    "unverified_numbers": validation_numbers,
                    "unverified_dates": validation_dates,
                    "missing_spans": missing_spans,
                }
            ),
            "extraction_status": status,
        }
    )
