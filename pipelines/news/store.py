"""SQLite: документы, ревизии, кластеры, запуски и здоровье источников."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alfa.core.schemas import NewsDoc
from alfa.data.interfaces import eligible_for_extraction

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,
    trust_tier TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    served_text TEXT NOT NULL,
    text_len INTEGER NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    updated_at TEXT,
    revision INTEGER NOT NULL,
    lang TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_ref TEXT,
    flags TEXT NOT NULL,
    prefilter TEXT NOT NULL,
    cluster_id TEXT,
    is_primary INTEGER NOT NULL DEFAULT 1,
    echo_count INTEGER NOT NULL DEFAULT 0,
    echo_sources TEXT NOT NULL DEFAULT '[]',
    extraction_status TEXT NOT NULL,
    origin TEXT NOT NULL,
    topic_hints TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    simhash TEXT NOT NULL,
    UNIQUE(source, canonical_url)
);
CREATE TABLE IF NOT EXISTS revisions (
    id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    PRIMARY KEY (id, revision)
);
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id TEXT PRIMARY KEY,
    primary_id TEXT,
    echo_count INTEGER NOT NULL,
    echo_sources TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    report TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_state (
    name TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    last_success_at TEXT,
    last_error TEXT,
    last_attempt_at TEXT,
    consecutive_errors INTEGER NOT NULL,
    empty_ratio REAL NOT NULL,
    new_docs INTEGER NOT NULL,
    baseline_new_docs INTEGER NOT NULL,
    poll_interval_s INTEGER NOT NULL,
    fallback_active INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS http_cache (
    url TEXT PRIMARY KEY,
    etag TEXT,
    last_modified TEXT,
    fetched_at TEXT,
    status INTEGER
);
CREATE TABLE IF NOT EXISTS extraction_queue (
    news_id TEXT PRIMARY KEY,
    queued_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    source TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(text_hash);
CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents(cluster_id);
CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def row_to_doc(row) -> NewsDoc:
    published = row["published_at"] or None
    updated = row["updated_at"] or None
    return NewsDoc(
        id=row["id"],
        source=row["source"],
        url=row["url"],
        canonical_url=row["canonical_url"],
        published_at=datetime.fromisoformat(published) if published else None,
        title=row["title"],
        text=row["served_text"],
        text_len=int(row["text_len"]),
        source_type=row["source_type"],
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
        updated_at=datetime.fromisoformat(updated) if updated else None,
        revision=int(row["revision"]),
        lang=row["lang"],
        trust_tier=row["trust_tier"],
        content_hash=row["content_hash"],
        raw_ref=row["raw_ref"],
        flags=json.loads(row["flags"]),
        prefilter=row["prefilter"],
        cluster_id=row["cluster_id"],
        is_primary=bool(row["is_primary"]),
        echo_count=int(row["echo_count"]),
        echo_sources=json.loads(row["echo_sources"]),
        extraction_status=row["extraction_status"],
        origin=row["origin"],
        topic_hints=json.loads(row["topic_hints"]),
    )


class NewsStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert(self, draft, raw_ref: str | None) -> dict:
        with self._lock:
            existing = self._conn.execute("SELECT * FROM documents WHERE id = ?", (draft.id,)).fetchone()
            now = _now()
            if existing is None:
                status = _initial_status(draft)
                self._insert_document(draft, revision=1, status=status, raw_ref=raw_ref, updated_at=None)
                if status == "pending":
                    self._enqueue(draft.id, now)
                self._conn.commit()
                return {"id": draft.id, "revision": 1, "created": True, "changed": True}
            if existing["content_hash"] == draft.content_hash:
                self._conn.execute(
                    "UPDATE documents SET fetched_at = ?, raw_ref = COALESCE(?, raw_ref) WHERE id = ?",
                    (now, raw_ref, draft.id),
                )
                self._conn.commit()
                return {"id": draft.id, "revision": int(existing["revision"]), "created": False, "changed": False}
            self._conn.execute(
                """
                INSERT INTO revisions (id, revision, content_hash, title, text, captured_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    existing["id"],
                    int(existing["revision"]),
                    existing["content_hash"],
                    existing["title"],
                    existing["text"],
                    now,
                ),
            )
            revision = int(existing["revision"]) + 1
            status = _initial_status(draft)
            self._conn.execute(
                """
                UPDATE documents SET
                    title = ?, text = ?, served_text = ?, text_len = ?, published_at = ?,
                    fetched_at = ?, updated_at = ?, revision = ?, lang = ?, content_hash = ?,
                    raw_ref = COALESCE(?, raw_ref), flags = ?, prefilter = ?, extraction_status = ?,
                    topic_hints = ?, text_hash = ?, simhash = ?, url = ?, trust_tier = ?, origin = ?
                WHERE id = ?
                """,
                (
                    draft.title,
                    draft.text,
                    draft.served_text,
                    draft.text_len,
                    _iso(draft.published_at),
                    now,
                    now,
                    revision,
                    draft.lang,
                    draft.content_hash,
                    raw_ref,
                    json.dumps(draft.flags, ensure_ascii=False),
                    draft.prefilter,
                    status,
                    json.dumps(draft.topic_hints, ensure_ascii=False),
                    draft.text_hash,
                    draft.simhash,
                    draft.url,
                    draft.trust_tier,
                    draft.origin,
                    draft.id,
                ),
            )
            if status == "pending":
                self._enqueue(draft.id, now)
            else:
                self._conn.execute("DELETE FROM extraction_queue WHERE news_id = ?", (draft.id,))
            self._conn.commit()
            return {"id": draft.id, "revision": revision, "created": False, "changed": True}

    def _insert_document(self, draft, *, revision: int, status: str, raw_ref: str | None, updated_at: str | None) -> None:
        self._conn.execute(
            """
            INSERT INTO documents (
                id, source, source_type, trust_tier, url, canonical_url, title, text, served_text,
                text_len, published_at, fetched_at, updated_at, revision, lang, content_hash, raw_ref,
                flags, prefilter, cluster_id, is_primary, echo_count, echo_sources, extraction_status,
                origin, topic_hints, text_hash, simhash
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, NULL, 1, 0, '[]', ?,
                ?, ?, ?, ?
            )
            """,
            (
                draft.id,
                draft.source,
                draft.source_type,
                draft.trust_tier,
                draft.url,
                draft.canonical_url,
                draft.title,
                draft.text,
                draft.served_text,
                draft.text_len,
                _iso(draft.published_at),
                _iso(draft.fetched_at),
                updated_at,
                revision,
                draft.lang,
                draft.content_hash,
                raw_ref,
                json.dumps(draft.flags, ensure_ascii=False),
                draft.prefilter,
                status,
                draft.origin,
                json.dumps(draft.topic_hints, ensure_ascii=False),
                draft.text_hash,
                draft.simhash,
            ),
        )

    def _enqueue(self, news_id: str, now: str) -> None:
        self._conn.execute(
            "INSERT INTO extraction_queue (news_id, queued_at) VALUES (?, ?) ON CONFLICT(news_id) DO NOTHING",
            (news_id, now),
        )

    def get_row(self, news_id: str):
        with self._lock:
            return self._conn.execute("SELECT * FROM documents WHERE id = ?", (news_id,)).fetchone()

    def get_doc(self, news_id: str) -> NewsDoc | None:
        row = self.get_row(news_id)
        if row is None:
            return None
        return row_to_doc(row)

    def get_many(self, ids: list[str]) -> list:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        with self._lock:
            return self._conn.execute(f"SELECT * FROM documents WHERE id IN ({marks})", ids).fetchall()

    def all_rows(self) -> list:
        with self._lock:
            return self._conn.execute("SELECT * FROM documents ORDER BY fetched_at, id").fetchall()

    def rows_for_dedup(self, focus_ids: list[str], window_hours: int) -> list:
        focus = self.get_many(focus_ids)
        if not focus:
            return []
        stamps = [datetime.fromisoformat(row["published_at"] or row["fetched_at"]) for row in focus]
        start = (min(stamps) - timedelta(hours=window_hours)).isoformat()
        end = (max(stamps) + timedelta(hours=window_hours)).isoformat()
        hashes = [row["text_hash"] for row in focus]
        urls = [row["canonical_url"] for row in focus]
        hash_marks = ",".join("?" * len(hashes))
        url_marks = ",".join("?" * len(urls))
        sql = f"""
            SELECT * FROM documents
            WHERE text_hash IN ({hash_marks})
               OR canonical_url IN ({url_marks})
               OR COALESCE(published_at, fetched_at) BETWEEN ? AND ?
        """
        with self._lock:
            rows = self._conn.execute(sql, [*hashes, *urls, start, end]).fetchall()
        unique = {row["id"]: row for row in rows}
        return list(unique.values())

    def update_cluster(self, news_id: str, *, cluster_id: str, is_primary: bool, echo_count: int, echo_sources: list[str]) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE documents
                SET cluster_id = ?, is_primary = ?, echo_count = ?, echo_sources = ?
                WHERE id = ?
                """,
                (cluster_id, int(is_primary), echo_count, json.dumps(echo_sources, ensure_ascii=False), news_id),
            )
            self._conn.commit()

    def save_cluster(self, cluster_id: str, primary_id: str, echo_count: int, echo_sources: list[str], updated_at: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO clusters (cluster_id, primary_id, echo_count, echo_sources, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cluster_id) DO UPDATE SET
                    primary_id = excluded.primary_id,
                    echo_count = excluded.echo_count,
                    echo_sources = excluded.echo_sources,
                    updated_at = excluded.updated_at
                """,
                (cluster_id, primary_id, echo_count, json.dumps(echo_sources, ensure_ascii=False), updated_at),
            )
            self._conn.commit()

    def refresh_queue(self, ids: list[str]) -> None:
        now = _now()
        with self._lock:
            if not ids:
                self._conn.commit()
                return
            marks = ",".join("?" * len(ids))
            rows = self._conn.execute(f"SELECT * FROM documents WHERE id IN ({marks})", ids).fetchall()
            for row in rows:
                if row["extraction_status"] == "done":
                    continue
                doc = row_to_doc(row)
                if eligible_for_extraction(doc):
                    self._conn.execute(
                        "UPDATE documents SET extraction_status = 'pending' WHERE id = ?",
                        (doc.id,),
                    )
                    self._enqueue(doc.id, now)
                else:
                    self._conn.execute(
                        "UPDATE documents SET extraction_status = 'skipped' WHERE id = ?",
                        (doc.id,),
                    )
                    self._conn.execute("DELETE FROM extraction_queue WHERE news_id = ?", (doc.id,))
            self._conn.commit()

    def queue_rows(self) -> list:
        with self._lock:
            return self._conn.execute(
                """
                SELECT documents.* FROM extraction_queue
                JOIN documents ON documents.id = extraction_queue.news_id
                WHERE documents.extraction_status = 'pending'
                ORDER BY extraction_queue.queued_at
                """
            ).fetchall()

    def cluster_rows(self, cluster_id: str) -> list:
        with self._lock:
            return self._conn.execute(
                """
                SELECT * FROM documents WHERE cluster_id = ?
                ORDER BY is_primary DESC, COALESCE(published_at, fetched_at), id
                """,
                (cluster_id,),
            ).fetchall()

    def revisions(self, news_id: str) -> list:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM revisions WHERE id = ? ORDER BY revision",
                (news_id,),
            ).fetchall()

    def count_documents(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def duplicate_identity_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT source, canonical_url FROM documents
                    GROUP BY source, canonical_url HAVING COUNT(*) > 1
                )
                """
            ).fetchone()
        return int(row[0])

    def get_cache(self, url: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM http_cache WHERE url = ?", (url,)).fetchone()
        if row is None:
            return None
        return {"etag": row["etag"], "last_modified": row["last_modified"]}

    def put_cache(self, url: str, etag: str | None, last_modified: str | None, status: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO http_cache (url, etag, last_modified, fetched_at, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    etag = excluded.etag,
                    last_modified = excluded.last_modified,
                    fetched_at = excluded.fetched_at,
                    status = excluded.status
                """,
                (url, etag, last_modified, _now(), status),
            )
            self._conn.commit()

    def get_state(self, name: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM source_state WHERE name = ?", (name,)).fetchone()
        return dict(row) if row is not None else None

    def save_state(self, state: dict) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO source_state (
                    name, state, last_success_at, last_error, last_attempt_at, consecutive_errors,
                    empty_ratio, new_docs, baseline_new_docs, poll_interval_s, fallback_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    state = excluded.state,
                    last_success_at = excluded.last_success_at,
                    last_error = excluded.last_error,
                    last_attempt_at = excluded.last_attempt_at,
                    consecutive_errors = excluded.consecutive_errors,
                    empty_ratio = excluded.empty_ratio,
                    new_docs = excluded.new_docs,
                    baseline_new_docs = excluded.baseline_new_docs,
                    poll_interval_s = excluded.poll_interval_s,
                    fallback_active = excluded.fallback_active
                """,
                (
                    state["name"],
                    state["state"],
                    state.get("last_success_at"),
                    state.get("last_error") or "",
                    state.get("last_attempt_at"),
                    int(state["consecutive_errors"]),
                    float(state["empty_ratio"]),
                    int(state["new_docs"]),
                    int(state["baseline_new_docs"]),
                    int(state["poll_interval_s"]),
                    int(state["fallback_active"]),
                ),
            )
            self._conn.commit()

    def all_states(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM source_state ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def add_journal(self, event: str, source: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO journal (ts, event, source, payload) VALUES (?, ?, ?, ?)",
                (_now(), event, source, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()

    def journal(self, event: str | None = None) -> list[dict]:
        with self._lock:
            if event:
                rows = self._conn.execute(
                    "SELECT * FROM journal WHERE event = ? ORDER BY id",
                    (event,),
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM journal ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def save_run(self, report_id: str, started_at: str, finished_at: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, started_at, finished_at, report) VALUES (?, ?, ?, ?)",
                (report_id, started_at, finished_at, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()

    def latest_run(self) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return json.loads(row["report"])

    def cluster_index(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT cluster_id, COUNT(*) AS size FROM documents
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
                """
            ).fetchall()
        return [{"cluster_id": row["cluster_id"], "size": int(row["size"])} for row in rows]


def _initial_status(draft) -> str:
    if draft.prefilter == "likely_irrelevant" or "needs_js" in draft.flags:
        return "skipped"
    return "pending"
