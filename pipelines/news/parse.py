"""Разбор RSS, JSON API и HTML. JavaScript не исполняется."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from urllib.parse import urljoin

from pipelines.news.textutil import parse_datetime

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


@dataclass
class RssItem:
    url: str
    title: str
    summary: str
    published: datetime | None


@dataclass
class HtmlExtract:
    title: str
    text: str
    published_raw: str
    engine: str


def parse_rss(payload: bytes, tz) -> list[RssItem]:
    parsed = _parse_feedparser(payload, tz)
    if parsed:
        return parsed
    return _parse_xml(payload, tz)


def _parse_feedparser(payload: bytes, tz) -> list[RssItem]:
    try:
        import feedparser
    except ImportError:
        return []
    try:
        feed = feedparser.parse(payload)
    except Exception:
        return []
    items: list[RssItem] = []
    for entry in feed.entries:
        title = unescape(_plain(getattr(entry, "title", "") or ""))
        link = (getattr(entry, "link", "") or "").strip()
        if not title or not link:
            continue
        summary = ""
        if getattr(entry, "content", None):
            summary = _plain(entry.content[0].get("value") or "")
        if not summary:
            summary = _plain(getattr(entry, "summary", "") or "")
        published = None
        raw_date = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if raw_date:
            published = parse_datetime(str(raw_date), tz)
        if published is None and getattr(entry, "published_parsed", None):
            stamp = entry.published_parsed
            published = datetime(*stamp[:6], tzinfo=timezone.utc).replace(microsecond=0)
        items.append(RssItem(url=link, title=title, summary=unescape(summary), published=published))
    return items


def _parse_xml(payload: bytes, tz) -> list[RssItem]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    items: list[RssItem] = []
    for node in root.findall("channel/item"):
        title = unescape(_plain(node.findtext("title") or ""))
        link = (node.findtext("link") or "").strip()
        if not title or not link:
            continue
        summary = node.findtext("description") or ""
        encoded = node.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if encoded is not None and (encoded.text or "").strip():
            summary = encoded.text or ""
        published = parse_datetime(node.findtext("pubDate") or "", tz)
        items.append(RssItem(url=link, title=title, summary=unescape(_plain(summary)), published=published))
    if items:
        return items
    for node in root.findall("atom:entry", _NS):
        title = unescape(_plain(node.findtext("atom:title", default="", namespaces=_NS)))
        link = ""
        for link_node in node.findall("atom:link", _NS):
            rel = link_node.get("rel") or "alternate"
            if rel == "alternate" and link_node.get("href"):
                link = link_node.get("href") or ""
                break
        if not title or not link:
            continue
        summary = node.findtext("atom:summary", default="", namespaces=_NS) or node.findtext(
            "atom:content", default="", namespaces=_NS
        )
        published = parse_datetime(
            node.findtext("atom:published", default="", namespaces=_NS)
            or node.findtext("atom:updated", default="", namespaces=_NS),
            tz,
        )
        items.append(RssItem(url=link, title=title, summary=unescape(_plain(summary or "")), published=published))
    return items


def parse_api(payload: bytes, tz) -> list[RssItem]:
    try:
        data = json.loads(payload.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    rows = data.get("items") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    items: list[RssItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = _plain(str(row.get("title") or ""))
        url = str(row.get("url") or "").strip()
        if not title or not url:
            continue
        text = str(row.get("text") or row.get("summary") or "")
        published = parse_datetime(str(row.get("published_at") or ""), tz)
        items.append(RssItem(url=url, title=title, summary=text, published=published))
    return items


def extract_links(html: str, base_url: str, selectors: dict | None) -> list[str]:
    tree = _tree(html)
    if tree is None:
        return []
    selector = (selectors or {}).get("item_link") or "a"
    nodes = _select(tree, selector)
    links: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        href = (node.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
    return links


def extract_html(html: str, selectors: dict | None = None, *, engines: tuple[str, ...] | None = None) -> HtmlExtract:
    selectors = selectors or {}
    order = engines or ("trafilatura", "readability", "selectors", "lxml")
    meta_title, published_raw = _meta(html, selectors)
    best: HtmlExtract | None = None
    for engine in order:
        text = _engine_text(engine, html, selectors)
        if not text:
            continue
        candidate = HtmlExtract(title=meta_title, text=text, published_raw=published_raw, engine=engine)
        if len(text) >= 80:
            return candidate
        if best is None or len(text) > len(best.text):
            best = candidate
    if best is not None:
        return best
    return HtmlExtract(title=meta_title, text="", published_raw=published_raw, engine="none")


def _engine_text(engine: str, html: str, selectors: dict) -> str:
    if engine == "trafilatura":
        return _trafilatura_text(html)
    if engine == "readability":
        return _readability_text(html)
    tree = _tree(html)
    if tree is None:
        return ""
    _drop_noise(tree)
    if engine == "selectors":
        return _node_text(_select(tree, selectors.get("text") or "article"))
    if engine == "lxml":
        for selector in ("article", "main", "body"):
            text = _node_text(_select(tree, selector))
            if text:
                return text
    return ""


def _trafilatura_text(html: str) -> str:
    try:
        import trafilatura
    except ImportError:
        return ""
    try:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
    except Exception:
        return ""
    return _plain(extracted or "")


def _readability_text(html: str) -> str:
    try:
        from readability import Document
    except ImportError:
        return ""
    try:
        summary = Document(html).summary()
    except Exception:
        return ""
    tree = _tree(summary or "")
    if tree is None:
        return ""
    _drop_noise(tree)
    return _plain(tree.text_content())


def _meta(html: str, selectors: dict) -> tuple[str, str]:
    tree = _tree(html)
    if tree is None:
        return "", ""
    title = ""
    if selectors.get("title"):
        title = _node_text(_select(tree, selectors["title"]))
    if not title:
        title = _attr(tree, "//meta[@property='og:title']/@content")
    if not title:
        title = _node_text(_select(tree, "h1"))
    if not title:
        title = _node_text(_select(tree, "title"))
    published = ""
    if selectors.get("date"):
        nodes = _select(tree, selectors["date"])
        if nodes:
            published = nodes[0].get("datetime") or _plain(nodes[0].text_content())
    if not published:
        published = _attr(tree, "//meta[@property='article:published_time']/@content")
    if not published:
        published = _attr(tree, "//meta[@itemprop='datePublished']/@content")
    if not published:
        published = _attr(tree, "//time/@datetime")
    if not published:
        published = _jsonld_date(tree)
    return _plain(title), published.strip()


def _jsonld_date(tree) -> str:
    for node in tree.xpath("//script[@type='application/ld+json']"):
        raw = node.text or ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        found = _find_date(payload)
        if found:
            return found
    return ""


def _find_date(payload) -> str:
    if isinstance(payload, list):
        for item in payload:
            found = _find_date(item)
            if found:
                return found
        return ""
    if not isinstance(payload, dict):
        return ""
    for key in ("datePublished", "dateCreated", "uploadDate"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    graph = payload.get("@graph")
    if graph is not None:
        return _find_date(graph)
    return ""


def _tree(html: str):
    if not html:
        return None
    try:
        from lxml import html as lhtml
    except ImportError:
        return None
    parser = lhtml.HTMLParser(encoding="utf-8")
    try:
        return lhtml.fromstring(html.encode("utf-8", errors="replace"), parser=parser)
    except Exception:
        return None


def _drop_noise(tree) -> None:
    paths = [
        "//script",
        "//style",
        "//noscript",
        "//svg",
        "//iframe",
        "//form",
        "//nav",
        "//footer",
        "//aside",
        "//*[contains(@class, 'caption')]",
    ]
    for path in paths:
        for node in tree.xpath(path):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)


def _select(tree, selector: str):
    if not selector:
        return []
    if selector.startswith("//"):
        try:
            return tree.xpath(selector)
        except Exception:
            return []
    try:
        return tree.cssselect(selector)
    except Exception:
        return []


def _node_text(nodes) -> str:
    parts = []
    for node in nodes:
        if isinstance(node, str):
            parts.append(node)
            continue
        parts.append(node.text_content())
    return _plain("\n".join(parts))


def _attr(tree, path: str) -> str:
    found = tree.xpath(path)
    if not found:
        return ""
    value = found[0]
    return str(value).strip() if value else ""


def _plain(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", unescape(value or ""))
    return re.sub(r"\s+", " ", text).strip()
