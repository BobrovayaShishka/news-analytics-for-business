"""Идентификаторы.

Грамматика (все части, кроме префикса, без двоеточий):

    news_id  := "news:" source ":" sha1_10
    source   := [A-Za-z0-9_-]+
    sha1_10  := [0-9a-f]{10}     # первые 10 hex-символов SHA1 от URL в UTF-8
    doc_id   := [A-Za-z0-9_-]+
    chunk_id := "an:" doc_id ":p" page ":c" n
    card_id  := "card:" doc_id ":" n
    page, n  := [0-9]+

Источник в news id — слаг, не отображаемое имя. Символы вне [A-Za-z0-9_-]
в слаг не входят, иначе разбор по двоеточиям неоднозначен.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_SOURCE = r"[A-Za-z0-9_-]+"
_DOC = r"[A-Za-z0-9_-]+"
_SHA = r"[0-9a-f]{10}"
_NUM = r"[0-9]+"

NEWS_RE = re.compile(rf"^news:({_SOURCE}):({_SHA})$")
CHUNK_RE = re.compile(rf"^an:({_DOC}):p({_NUM}):c({_NUM})$")
CARD_RE = re.compile(rf"^card:({_DOC}):({_NUM})$")
SLUG_RE = re.compile(rf"^{_SOURCE}$")


def _require_slug(value: str, what: str) -> str:
    if not SLUG_RE.fullmatch(value or ""):
        raise ValueError(f"{what} must match [A-Za-z0-9_-]+, got {value!r}")
    return value


def make_news_id(source: str, url: str) -> str:
    slug = _require_slug(source, "source")
    if not url:
        raise ValueError("url is required")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"news:{slug}:{digest}"


@dataclass(frozen=True)
class NewsId:
    source: str
    digest: str

    @property
    def value(self) -> str:
        return f"news:{self.source}:{self.digest}"


def parse_news_id(value: str) -> NewsId:
    match = NEWS_RE.fullmatch(value or "")
    if not match:
        raise ValueError(f"not a news id: {value!r}")
    return NewsId(source=match.group(1), digest=match.group(2))


def make_chunk_id(doc_id: str, page: int, n: int) -> str:
    slug = _require_slug(doc_id, "doc_id")
    if page < 0 or n < 0:
        raise ValueError("page and n must be >= 0")
    return f"an:{slug}:p{page}:c{n}"


@dataclass(frozen=True)
class ChunkId:
    doc_id: str
    page: int
    n: int

    @property
    def value(self) -> str:
        return make_chunk_id(self.doc_id, self.page, self.n)


def parse_chunk_id(value: str) -> ChunkId:
    match = CHUNK_RE.fullmatch(value or "")
    if not match:
        raise ValueError(f"not a chunk id: {value!r}")
    return ChunkId(doc_id=match.group(1), page=int(match.group(2)), n=int(match.group(3)))


def make_card_id(doc_id: str, n: int) -> str:
    slug = _require_slug(doc_id, "doc_id")
    if n < 0:
        raise ValueError("n must be >= 0")
    return f"card:{slug}:{n}"


@dataclass(frozen=True)
class CardId:
    doc_id: str
    n: int

    @property
    def value(self) -> str:
        return make_card_id(self.doc_id, self.n)


def parse_card_id(value: str) -> CardId:
    match = CARD_RE.fullmatch(value or "")
    if not match:
        raise ValueError(f"not a card id: {value!r}")
    return CardId(doc_id=match.group(1), n=int(match.group(2)))
