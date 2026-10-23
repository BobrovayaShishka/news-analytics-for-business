"""Нормализация, очистка и флаги качества. Текст новости остаётся данными."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlsplit, urlunsplit, urlencode

from alfa.core.ids import make_news_id
from alfa.data.interfaces import IngestError
from pipelines.news.models import Draft

CYRILLIC_LANG_SHARE = 0.4
LATIN_LANG_SHARE = 0.5
SIMHASH_BITS = 64
SIMHASH_GRAMS = 3
BGE_MAX_SEQ_LENGTH = 512
BGE_ENCODE_BATCH = 8
_WORD = re.compile(r"\w+", flags=re.UNICODE)

_FIXED_TZ = {
    "europe/moscow": timezone(timedelta(hours=3)),
    "msk": timezone(timedelta(hours=3)),
    "utc": timezone.utc,
}
_TRACKING = {
    "fbclid",
    "gclid",
    "yclid",
    "ymclid",
    "ysclid",
    "_openstat",
    "openstat",
    "mc_cid",
    "mc_eid",
    "ref_src",
    "from",
    "clid",
    "_cldee",
    "roistat",
    "roistat_visit",
    "rb_clickid",
    "yadclid",
    "yadordid",
    "fb_action_ids",
    "fb_action_types",
    "fb_source",
    "share",
    "share_token",
    "shared_token",
}
_TITLE_TAIL = re.compile(
    r"\s+[-|–—]\s+(рбк|интерфакс|ведомости|коммерсантъ|коммерсант|минфин( россии)?|банк россии|фас( россии)?)\s*$",
    re.IGNORECASE,
)
_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def source_timezone(name: str | None):
    key = (name or "Europe/Moscow").strip()
    fixed = _FIXED_TZ.get(key.lower())
    if fixed is not None:
        return fixed
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(key)
    except Exception:
        return _FIXED_TZ["europe/moscow"]


def parse_datetime(value: str | None, tz) -> datetime | None:
    if not value or not str(value).strip():
        return None
    raw = str(value).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None:
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def resolve_published(page: datetime | None, rss: datetime | None) -> datetime | None:
    """Дата страницы важнее ленты. Если нет нигде — None, без подстановки «сейчас»."""
    return page or rss


def canonical_url(url: str) -> str:
    raw = (url or "").strip()
    if raw.startswith("urn:"):
        return raw
    parts = urlsplit(raw)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return raw
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    kept = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if not _drop_query_key(key)]
    kept.sort()
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, urlencode(kept, doseq=True), ""))


def _drop_query_key(key: str) -> bool:
    low = key.lower()
    return low.startswith("utm_") or low in _TRACKING


def normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = (
        cleaned.replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
        .replace("\u2007", " ")
        .replace("\ufeff", "")
    )
    cleaned = cleaned.translate(
        str.maketrans(
            {
                "\u201c": '"',
                "\u201d": '"',
                "\u00ab": '"',
                "\u00bb": '"',
                "\u2018": "'",
                "\u2019": "'",
                "\u2013": "-",
                "\u2014": "-",
            }
        )
    )
    cleaned = _CONTROLS.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def clean_title(title: str, settings: dict) -> str:
    text = normalize_text(title)
    text = _TITLE_TAIL.sub("", text).strip()
    for name in settings.get("title_sources") or []:
        text = re.sub(rf"\s+[-|–—]\s+{re.escape(str(name))}\s*$", "", text, flags=re.IGNORECASE).strip()
    return text


def clean_text(text: str, patterns: list[str], settings: dict) -> str:
    normalized = normalize_text(text)
    for pattern in list(settings.get("strip_patterns") or []) + list(patterns):
        normalized = re.sub(pattern, " ", normalized)
    return normalize_text(normalized)


def word_tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower().replace("ё", "е"))


def detect_lang(text: str, settings: dict | None = None) -> str:
    lang_cfg = (settings or {}).get("lang") or {}
    cyrillic_share = float(lang_cfg.get("cyrillic_share", CYRILLIC_LANG_SHARE))
    latin_share = float(lang_cfg.get("latin_share", LATIN_LANG_SHARE))
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "und"
    cyr = sum(1 for char in letters if "а" <= char.lower() <= "я" or char.lower() == "ё")
    if cyr / len(letters) >= cyrillic_share:
        return "ru"
    lat = sum(1 for char in letters if "a" <= char.lower() <= "z")
    if lat / len(letters) >= latin_share:
        return "en"
    return "other"


def simhash_value(text: str, bits: int = SIMHASH_BITS, grams: int = SIMHASH_GRAMS) -> int:
    tokens = word_tokens(text)
    if len(tokens) >= grams:
        grams_list = [" ".join(tokens[index : index + grams]) for index in range(len(tokens) - grams + 1)]
    else:
        grams_list = tokens
    if not grams_list:
        return 0
    acc = [0] * bits
    for gram in grams_list:
        digest = hashlib.blake2s(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(bits):
            acc[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit, weight in enumerate(acc):
        if weight > 0:
            out |= 1 << bit
    return out


def simhash_hex(text: str, bits: int = SIMHASH_BITS, grams: int = SIMHASH_GRAMS) -> str:
    width = bits // 4
    return f"{simhash_value(text, bits, grams) & ((1 << bits) - 1):0{width}x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _needs_js(html: str, text: str, min_chars: int) -> bool:
    if len(text) >= min_chars:
        return False
    low = (html or "").lower()
    if any(marker in low for marker in ("enable javascript", "нужен javascript", "включите javascript")):
        return True
    return low.count("<script") >= 8 and len(text) < min(min_chars, 200)


def _suspicious(text: str, settings: dict) -> bool:
    for pattern in settings.get("suspicious_patterns") or []:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def _paywall(text: str, html: str, settings: dict) -> bool:
    blob = f"{text}\n{html}".lower().replace("ё", "е")
    return any(marker in blob for marker in settings.get("paywall_markers") or [])


def _irrelevant(url: str, title: str, text: str, settings: dict) -> bool:
    rules = settings.get("prefilter") or {}
    hay_url = (url or "").lower()
    for part in rules.get("url_parts") or []:
        if str(part).lower() in hay_url:
            return True
    blob = f"{title}\n{text}".lower().replace("ё", "е")
    for word in rules.get("blacklist") or []:
        if str(word).lower().replace("ё", "е") in blob:
            return True
    return False


def quality_marks(
    *,
    text: str,
    title: str,
    url: str,
    published: datetime | None,
    lang: str,
    html: str,
    settings: dict,
    http_truncated: bool,
    original_len: int,
) -> tuple[list[str], str]:
    flags: list[str] = []
    limit = int(settings["max_text_chars"])
    min_chars = int(settings["min_text_chars"])
    if original_len < min_chars:
        flags.append("too_short")
    if http_truncated or original_len > limit:
        flags.append("truncated")
    if published is None:
        flags.append("date_missing")
    if lang not in set(settings.get("target_langs") or ["ru"]):
        flags.append("non_target_lang")
    if _paywall(text, html, settings):
        flags.append("paywall_suspected")
    if _needs_js(html, text, min_chars):
        flags.append("needs_js")
    if _suspicious(f"{title}\n{text}", settings):
        flags.append("suspicious_instructions")
    prefilter = "likely_irrelevant" if _irrelevant(url, title, text, settings) else "unknown"
    return sorted(set(flags)), prefilter


def build_draft(
    *,
    source: str,
    source_type: str,
    trust_tier: str,
    url: str,
    title: str,
    text: str,
    html: str,
    published_at: datetime | None,
    origin: str,
    topic_hints: list[str],
    strip_patterns: list[str],
    settings: dict,
    http_truncated: bool = False,
    fetched_at: datetime | None = None,
    raw_html: bytes | None = None,
) -> Draft:
    cleaned = clean_text(text, strip_patterns, settings)
    if origin == "on_demand" and not cleaned.strip():
        raise IngestError("текст не удалось получить")
    title_c = clean_title(title, settings)
    if not title_c:
        title_c = (cleaned.split("\n", 1)[0][:180] if cleaned else "") or "Без заголовка"
    canonical = canonical_url(url)
    lang = detect_lang(cleaned or title_c, settings)
    original_len = len(cleaned)
    full = cleaned[: int(settings["hard_text_chars"])]
    served = full[: int(settings["max_text_chars"])]
    flags, prefilter = quality_marks(
        text=full,
        title=title_c,
        url=canonical,
        published=published_at,
        lang=lang,
        html=html,
        settings=settings,
        http_truncated=http_truncated,
        original_len=original_len,
    )
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()
    return Draft(
        id=make_news_id(source, canonical),
        source=source,
        source_type=source_type,
        trust_tier=trust_tier,
        url=url,
        canonical_url=canonical,
        title=title_c,
        text=full,
        served_text=served,
        text_len=original_len,
        published_at=published_at,
        fetched_at=(fetched_at or datetime.now(timezone.utc)).replace(microsecond=0),
        lang=lang,
        content_hash=digest,
        flags=flags,
        prefilter=prefilter,
        origin=origin,
        topic_hints=list(topic_hints),
        text_hash=digest,
        simhash=simhash_hex(
            full,
            bits=int((settings.get("dedup") or {}).get("simhash_bits", SIMHASH_BITS)),
            grams=int((settings.get("dedup") or {}).get("simhash_grams", SIMHASH_GRAMS)),
        ),
        raw_html=raw_html,
        http_truncated=http_truncated,
    )
