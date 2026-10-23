"""Сбор, разбор и запись. Один сбойный источник не останавливает остальные."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from alfa.core.logging import get_logger
from alfa.core.timeutil import SECONDS_PER_DAY, SECONDS_PER_HOUR
from alfa.data.interfaces import IngestError, RunReport
from pipelines.news.dedup import recluster
from pipelines.news.http import FetchResult, fetch_checked
from pipelines.news.models import Draft, SourceSpec
from pipelines.news.parse import extract_html, extract_links, parse_api, parse_rss
from pipelines.news.store import NewsStore
from pipelines.news.textutil import build_draft, normalize_text, parse_datetime, resolve_published, source_timezone

logger = get_logger("news")
_CHARSET = re.compile(r"charset=([\w.-]+)", re.IGNORECASE)


def log_event(event: str, **fields) -> None:
    logger.info("%s", json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))


class NewsPipeline:
    def __init__(self, settings: dict, sources: list[SourceSpec], store: NewsStore, transport, embedder):
        self.settings = settings
        self.sources = sources
        self.store = store
        self.transport = transport
        self.embedder = embedder
        self.raw_dir = Path(settings["raw_dir"])
        if not self.raw_dir.is_absolute():
            from alfa.core.config import resolve_path

            self.raw_dir = resolve_path(self.raw_dir)
        self._robots: dict[str, RobotFileParser | bool | None] = {}
        self._robots_lock = threading.Lock()
        self._by_name = {spec.name: spec for spec in sources}

    def ingest_text(self, text: str, title: str | None = None):
        body = text or ""
        digest = hashlib.sha1(normalize_text(body).encode("utf-8")).hexdigest()[:16]
        draft = build_draft(
            source="user",
            source_type="user_submitted",
            trust_tier="unverified",
            url=f"urn:user-text:{digest}",
            title=title or "",
            text=body,
            html="",
            published_at=None,
            origin="on_demand",
            topic_hints=[],
            strip_patterns=[],
            settings=self.settings,
        )
        return self._save_one(draft)

    def ingest_url(self, url: str):
        try:
            result = fetch_checked(url, self.transport, self.settings, user_supplied=True)
        except IngestError:
            raise
        if result.error or result.status == 0 or result.status >= 400 or not result.body:
            raise IngestError("текст не удалось получить")
        html = _decode(result)
        extracted = extract_html(html, {})
        if not extracted.text.strip():
            raise IngestError("текст не удалось получить")
        published = parse_datetime(extracted.published_raw, source_timezone("Europe/Moscow"))
        draft = build_draft(
            source="user",
            source_type="user_submitted",
            trust_tier="unverified",
            url=result.url or url,
            title=extracted.title,
            text=extracted.text,
            html=html,
            published_at=published,
            origin="on_demand",
            topic_hints=[],
            strip_patterns=[],
            settings=self.settings,
            http_truncated=result.truncated,
            raw_html=result.body,
        )
        if "needs_js" in draft.flags and "too_short" in draft.flags:
            raise IngestError("текст не удалось получить")
        return self._save_one(draft)

    def accept_many(self, drafts: list[Draft]):
        ids = []
        for draft in drafts:
            self.store.upsert(draft, self._write_raw(draft))
            ids.append(draft.id)
        if ids:
            recluster(self.store, self.embedder, self.settings, ids)
        return [self._require(item_id) for item_id in ids]

    def run_ingest(self, sources: list[str] | None = None, backfill_days: int | None = None) -> RunReport:
        started = datetime.now(timezone.utc).replace(microsecond=0)
        started_mono = time.perf_counter()
        self._purge_raw()
        run_id = started.strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha1(str(started_mono).encode()).hexdigest()[:6]
        queue = self._starting_sources(sources)
        seen: set[str] = set()
        problems: list[str] = []
        per_source: list[dict] = []
        drafts: list[Draft] = []
        while queue:
            spec = queue.pop(0)
            if spec.name in seen:
                continue
            seen.add(spec.name)
            try:
                batch, stats = self._collect(spec, backfill_days)
                drafts.extend(batch)
                created = self._persist(batch)
                stats["new_docs"] = created
                self._observe_success(spec, stats)
            except Exception as exc:
                stats = _empty_stats(spec.name)
                stats["errors"].append(str(exc))
                stats["new_docs"] = 0
                problems.append(spec.name)
                self._observe_failure(spec, str(exc))
                log_event("source_failed", source=spec.name, error=str(exc))
            per_source.append(stats)
            for extra in self._fallbacks_to_run(spec.fallback_group):
                if extra.name not in seen:
                    queue.append(extra)
        focus = [draft.id for draft in drafts]
        clustered = 0
        if focus:
            try:
                clustered = recluster(self.store, self.embedder, self.settings, focus)
            except Exception as exc:
                problems.append("dedup")
                log_event("dedup_failed", error=str(exc))
        finished = datetime.now(timezone.utc).replace(microsecond=0)
        report = self._report(
            run_id=run_id,
            started=started,
            finished=finished,
            per_source=per_source,
            problems=problems,
            drafts=drafts,
            clustered=clustered,
            elapsed=time.perf_counter() - started_mono,
        )
        self.store.save_run(report.run_id, report.started_at, report.finished_at, _report_dict(report))
        log_event("ingest_finished", run_id=report.run_id, new_docs=report.new_docs, problems=report.problems)
        return report

    def due_source_names(self, now: datetime | None = None) -> list[str]:
        moment = now or datetime.now(timezone.utc)
        due = []
        for spec in self.sources:
            if not self._effective(spec):
                continue
            state = self.store.get_state(spec.name)
            if state and state.get("last_attempt_at"):
                last = datetime.fromisoformat(state["last_attempt_at"])
                interval = int(state["poll_interval_s"])
                if (moment - last).total_seconds() < interval:
                    continue
            due.append(spec.name)
        return due

    def seconds_until_due(self, now: datetime | None = None) -> float:
        moment = now or datetime.now(timezone.utc)
        waits: list[float] = []
        for spec in self.sources:
            if not self._effective(spec):
                continue
            state = self.store.get_state(spec.name)
            if not state or not state.get("last_attempt_at"):
                return 0.0
            last = datetime.fromisoformat(state["last_attempt_at"])
            remain = int(state["poll_interval_s"]) - (moment - last).total_seconds()
            waits.append(max(0.0, remain))
        if not waits:
            return 30.0
        return min(waits)

    def _starting_sources(self, names: list[str] | None) -> list[SourceSpec]:
        if names is None:
            specs = [spec for spec in self.sources if self._effective(spec)]
            specs.sort(key=lambda spec: spec.role != "primary")
            return specs
        missing = [name for name in names if name not in self._by_name]
        if missing:
            raise IngestError("неизвестный источник: " + ", ".join(missing))
        return [self._by_name[name] for name in names]

    def _effective(self, spec: SourceSpec) -> bool:
        if spec.enabled:
            return True
        state = self.store.get_state(spec.name)
        return bool(state and state["fallback_active"])

    def _collect(self, spec: SourceSpec, backfill_days: int | None) -> tuple[list[Draft], dict]:
        started = time.perf_counter()
        if spec.type == "html":
            items, found, downloaded = self._html_items(spec)
        elif spec.type == "api":
            items, found, downloaded = self._api_items(spec)
        else:
            items, found, downloaded = self._rss_items(spec)
        cutoff = None
        if backfill_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=backfill_days)
        kept = []
        for item in items:
            if cutoff and item.published and item.published < cutoff:
                continue
            kept.append(item)
        loaded = self._load_pages(spec, kept)
        drafts: list[Draft] = []
        empty = 0
        flagged = 0
        for item, page in loaded:
            published = resolve_published(page["published"], item.published)
            text = page["text"] or item.summary
            title = item.title or page["title"]
            draft = build_draft(
                source=spec.name,
                source_type="news",
                trust_tier=spec.trust_tier,
                url=page["final_url"] or item.url,
                title=title,
                text=text,
                html=page["html"],
                published_at=published,
                origin="scheduled",
                topic_hints=spec.topic_hints,
                strip_patterns=spec.strip_patterns,
                settings=self.settings,
                http_truncated=page["truncated"],
                raw_html=page["raw"],
            )
            drafts.append(draft)
            if not draft.text.strip() or "too_short" in draft.flags:
                empty += 1
            if draft.flags:
                flagged += 1
        stats = {
            "name": spec.name,
            "found": found,
            "downloaded": downloaded + page_downloads(loaded),
            "parsed": len(drafts),
            "empty_or_short": empty,
            "flagged": flagged,
            "errors": [],
            "elapsed_s": round(time.perf_counter() - started, 3),
        }
        return drafts, stats

    def _rss_items(self, spec: SourceSpec):
        if not spec.feed_url:
            raise IngestError(f"{spec.name}: нет feed_url")
        if not self._robots_allowed(spec.feed_url):
            raise IngestError(f"robots.txt запрещает {spec.feed_url}")
        result = self._cached_get(spec.feed_url)
        if result.not_modified:
            return [], 0, 1
        if result.error or result.status >= 400 or not result.body:
            raise IngestError(result.error or f"HTTP {result.status}")
        items = parse_rss(result.body, source_timezone(spec.timezone))
        capped = items[: int(self.settings["max_items_per_source"])]
        return capped, len(capped), 1

    def _api_items(self, spec: SourceSpec):
        if not spec.feed_url:
            raise IngestError(f"{spec.name}: нет feed_url")
        if not self._robots_allowed(spec.feed_url):
            raise IngestError(f"robots.txt запрещает {spec.feed_url}")
        result = self._cached_get(spec.feed_url)
        if result.not_modified:
            return [], 0, 1
        if result.error or result.status >= 400 or not result.body:
            raise IngestError(result.error or f"HTTP {result.status}")
        items = parse_api(result.body, source_timezone(spec.timezone))
        capped = items[: int(self.settings["max_items_per_source"])]
        return capped, len(capped), 1

    def _html_items(self, spec: SourceSpec):
        if not spec.list_url:
            raise IngestError(f"{spec.name}: нет list_url")
        if not self._robots_allowed(spec.list_url):
            raise IngestError(f"robots.txt запрещает {spec.list_url}")
        result = self._cached_get(spec.list_url)
        if result.not_modified:
            return [], 0, 1
        if result.error or result.status >= 400 or not result.body:
            raise IngestError(result.error or f"HTTP {result.status}")
        from pipelines.news.parse import RssItem

        links = extract_links(_decode(result), spec.list_url, spec.selectors)
        links = links[: int(self.settings["max_items_per_source"])]
        items = [RssItem(url=link, title="", summary="", published=None) for link in links]
        return items, len(items), 1

    def _load_pages(self, spec: SourceSpec, items) -> list[tuple]:
        workers = max(1, int(self.settings["workers"]))

        def one(item):
            return item, self._load_page(spec, item)

        if workers == 1 or len(items) <= 1:
            return [one(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, items))

    def _load_page(self, spec: SourceSpec, item) -> dict:
        empty = {"html": "", "text": "", "title": "", "published": None, "truncated": False, "raw": None, "final_url": item.url, "downloaded": False}
        if not item.url or item.url.startswith("urn:"):
            return empty
        if not self._robots_allowed(item.url):
            return empty
        try:
            result = self._cached_get(item.url)
        except IngestError as exc:
            log_event("page_skipped", source=spec.name, url=item.url, error=str(exc))
            return empty
        if result.not_modified or result.error or result.status >= 400 or not result.body:
            return empty
        html = _decode(result)
        extracted = extract_html(html, spec.selectors)
        published = parse_datetime(extracted.published_raw, source_timezone(spec.timezone))
        return {
            "html": html,
            "text": extracted.text,
            "title": extracted.title,
            "published": published,
            "truncated": result.truncated,
            "raw": result.body,
            "final_url": result.url or item.url,
            "downloaded": True,
        }

    def _cached_get(self, url: str) -> FetchResult:
        conditional = self.store.get_cache(url)
        result = fetch_checked(url, self.transport, self.settings, user_supplied=False, conditional=conditional)
        if result.status == 200:
            self.store.put_cache(url, result.headers.get("etag"), result.headers.get("last-modified"), result.status)
        return result

    def _robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        host = parsed.netloc.lower()
        with self._robots_lock:
            if host not in self._robots:
                robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
                result = self.transport.get(robots_url)
                if result.error or result.status in {0} or result.status >= 500:
                    self._robots[host] = None
                elif result.status == 404:
                    self._robots[host] = True
                else:
                    parser = RobotFileParser()
                    parser.parse(result.body.decode("utf-8", errors="replace").splitlines())
                    self._robots[host] = parser
            cached = self._robots[host]
        if cached is None:
            return False
        if cached is True:
            return True
        return cached.can_fetch(self.settings["user_agent"], url)

    def _persist(self, drafts: list[Draft]) -> int:
        created = 0
        for draft in drafts:
            result = self.store.upsert(draft, self._write_raw(draft))
            if result["created"]:
                created += 1
        return created

    def _save_one(self, draft: Draft):
        self.store.upsert(draft, self._write_raw(draft))
        recluster(self.store, self.embedder, self.settings, [draft.id])
        return self._require(draft.id)

    def _require(self, news_id: str):
        doc = self.store.get_doc(news_id)
        if doc is None:
            raise KeyError(news_id)
        return doc

    def _write_raw(self, draft: Draft) -> str | None:
        if not draft.raw_html:
            return None
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{draft.id.replace(':', '_')}.html.gz"
        with gzip.open(path, "wb") as handle:
            handle.write(draft.raw_html)
        return str(path)

    def _purge_raw(self) -> None:
        if not self.raw_dir.exists():
            return
        cutoff = time.time() - int(self.settings["raw_ttl_days"]) * SECONDS_PER_DAY
        for path in self.raw_dir.glob("*.gz"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

    def _blank_state(self, spec: SourceSpec) -> dict:
        return {
            "name": spec.name,
            "state": "ok",
            "last_success_at": None,
            "last_error": "",
            "last_attempt_at": None,
            "consecutive_errors": 0,
            "empty_ratio": 0.0,
            "new_docs": 0,
            "baseline_new_docs": 0,
            "poll_interval_s": spec.poll_interval_s,
            "fallback_active": 0,
        }

    def _classify(self, state: dict, *, parsed: int) -> str:
        health = self.settings["health"]
        if int(state["consecutive_errors"]) >= int(health["down_errors"]):
            return "down"
        if int(state["consecutive_errors"]) >= int(health["degraded_errors"]):
            return "degraded"
        if parsed >= 3 and float(state["empty_ratio"]) >= float(health["empty_ratio_degraded"]):
            return "degraded"
        baseline = int(state["baseline_new_docs"])
        if baseline >= 5 and int(state["new_docs"]) < float(health["low_volume_ratio"]) * baseline:
            return "degraded"
        return "ok"

    def _observe_success(self, spec: SourceSpec, stats: dict) -> None:
        state = self.store.get_state(spec.name) or self._blank_state(spec)
        parsed = int(stats["parsed"])
        state["consecutive_errors"] = 0
        state["last_error"] = ""
        state["last_success_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        state["last_attempt_at"] = state["last_success_at"]
        state["new_docs"] = int(stats["new_docs"])
        state["empty_ratio"] = (int(stats["empty_or_short"]) / parsed) if parsed else 0.0
        if state["new_docs"] > 0:
            if int(state["baseline_new_docs"]) <= 0:
                state["baseline_new_docs"] = state["new_docs"]
            else:
                state["baseline_new_docs"] = int(round(0.5 * state["baseline_new_docs"] + 0.5 * state["new_docs"]))
        state["state"] = self._classify(state, parsed=parsed)
        if not state["fallback_active"]:
            state["poll_interval_s"] = spec.poll_interval_s
        self.store.save_state(state)
        self._sync_fallbacks(spec)

    def _observe_failure(self, spec: SourceSpec, error: str) -> None:
        state = self.store.get_state(spec.name) or self._blank_state(spec)
        state["consecutive_errors"] = int(state["consecutive_errors"]) + 1
        state["last_error"] = error
        state["last_attempt_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        state["new_docs"] = 0
        state["empty_ratio"] = 1.0
        state["state"] = self._classify(state, parsed=0)
        self.store.save_state(state)
        self._sync_fallbacks(spec)

    def _sync_fallbacks(self, spec: SourceSpec) -> None:
        group = spec.fallback_group
        if not group or spec.role != "primary":
            return
        state = self.store.get_state(spec.name)
        if state and state["state"] in {"degraded", "down"}:
            self._enable_fallbacks(group, spec.name)
            return
        if state and state["state"] == "ok":
            self._maybe_disable_fallbacks(group)

    def _enable_fallbacks(self, group: str, because: str) -> None:
        boost = float(self.settings["health"]["fallback_boost"])
        for spec in self.sources:
            if spec.role != "fallback" or spec.fallback_group != group:
                continue
            state = self.store.get_state(spec.name) or self._blank_state(spec)
            if state["fallback_active"]:
                continue
            state["fallback_active"] = 1
            state["poll_interval_s"] = max(60, int(spec.poll_interval_s * boost))
            self.store.save_state(state)
            self.store.add_journal("fallback_on", spec.name, {"because": because, "group": group})
            log_event("fallback_on", source=spec.name, because=because, group=group)

    def _maybe_disable_fallbacks(self, group: str) -> None:
        primaries = [spec for spec in self.sources if spec.role == "primary" and spec.fallback_group == group and spec.enabled]
        for spec in primaries:
            state = self.store.get_state(spec.name)
            if state and state["state"] in {"degraded", "down"}:
                return
        for spec in self.sources:
            if spec.role != "fallback" or spec.fallback_group != group:
                continue
            state = self.store.get_state(spec.name)
            if not state or not state["fallback_active"]:
                continue
            state["fallback_active"] = 0
            state["poll_interval_s"] = spec.poll_interval_s
            self.store.save_state(state)
            self.store.add_journal("fallback_off", spec.name, {"group": group})
            log_event("fallback_off", source=spec.name, group=group)

    def _fallbacks_to_run(self, group: str) -> list[SourceSpec]:
        if not group:
            return []
        ready = []
        for spec in self.sources:
            if spec.role == "fallback" and spec.fallback_group == group and self._effective(spec):
                ready.append(spec)
        return ready

    def _report(self, *, run_id, started, finished, per_source, problems, drafts, clustered, elapsed) -> RunReport:
        total = self.store.count_documents()
        echoes = sum(1 for row in self.store.all_rows() if not row["is_primary"])
        ages = []
        for draft in drafts:
            if draft.published_at is not None:
                ages.append((started - draft.published_at.astimezone(timezone.utc)).total_seconds() / SECONDS_PER_HOUR)
        attempted = len(per_source) or 1
        failed = sum(1 for item in per_source if item.get("errors"))
        fresh = statistics.median(ages) if ages else None
        return RunReport(
            run_id=run_id,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            found=sum(int(item["found"]) for item in per_source),
            downloaded=sum(int(item["downloaded"]) for item in per_source),
            parsed=sum(int(item["parsed"]) for item in per_source),
            empty_or_short=sum(int(item["empty_or_short"]) for item in per_source),
            flagged=sum(int(item["flagged"]) for item in per_source),
            clustered=clustered,
            new_docs=sum(int(item["new_docs"]) for item in per_source),
            problems=problems,
            sources=[{key: value for key, value in item.items()} for item in per_source],
            duplicate_ratio=(echoes / total) if total else 0.0,
            freshness_hours=fresh,
            coverage=(attempted - failed) / attempted,
            sample_clusters=sample_clusters(self.store, int(self.settings["report_sample_size"]), run_id),
            embedder=getattr(self.embedder, "name", "unknown"),
            elapsed_s=round(elapsed, 3),
        )


def _decode(result: FetchResult) -> str:
    match = _CHARSET.search(result.headers.get("content-type", ""))
    encoding = match.group(1) if match else "utf-8"
    return result.body.decode(encoding, errors="replace").lstrip("\ufeff")


def _empty_stats(name: str) -> dict:
    return {
        "name": name,
        "found": 0,
        "downloaded": 0,
        "parsed": 0,
        "empty_or_short": 0,
        "flagged": 0,
        "errors": [],
        "elapsed_s": 0.0,
        "new_docs": 0,
    }


def page_downloads(loaded: list[tuple]) -> int:
    return sum(1 for _item, page in loaded if page.get("downloaded"))


def sample_clusters(store: NewsStore, limit: int, seed: str) -> list[dict]:
    import random

    index = store.cluster_index()
    multi = [item for item in index if item["size"] > 1]
    pool = multi or index
    rng = random.Random(seed)
    rng.shuffle(pool)
    samples = []
    for item in pool[:limit]:
        rows = store.cluster_rows(item["cluster_id"])
        samples.append(
            {
                "cluster_id": item["cluster_id"],
                "size": item["size"],
                "members": [
                    {"id": row["id"], "source": row["source"], "title": row["title"], "is_primary": bool(row["is_primary"])}
                    for row in rows
                ],
            }
        )
    return samples


def _report_dict(report: RunReport) -> dict:
    return {
        "run_id": report.run_id,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "found": report.found,
        "downloaded": report.downloaded,
        "parsed": report.parsed,
        "empty_or_short": report.empty_or_short,
        "flagged": report.flagged,
        "clustered": report.clustered,
        "new_docs": report.new_docs,
        "problems": report.problems,
        "sources": report.sources,
        "duplicate_ratio": report.duplicate_ratio,
        "freshness_hours": report.freshness_hours,
        "coverage": report.coverage,
        "sample_clusters": report.sample_clusters,
        "embedder": report.embedder,
        "elapsed_s": report.elapsed_s,
    }

