"""Публичный сервис: матрица, батч, лента, объяснение, инкремент."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alfa.ranking.config import RankingConfig, load_ranking_config, repo_path
from alfa.ranking.engine import as_utc, event_is_active, explain_score, score_arrays
from alfa.ranking.matrix import (
    FeatureMatrix,
    assemble_matrix,
    default_inputs,
    load_group_table,
)
from alfa.ranking.models import CardView, ClientRow, Explanation, Feed, FeedItem, RankEvent, ScoredPair
from alfa.ranking.rerank import apply_rerank

_SERVICE: "RankingService | None" = None


def guard_client(client_id: str) -> None:
    from alfa.access import AccessError
    from alfa import access as access_mod

    current = access_mod._session.get()
    if not current or current != client_id:
        raise AccessError("client_id берётся из сессии и не совпадает с аргументом")


class ImpressionLog:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], tuple[int, str]] = {}

    def blocks(self, client_id: str, event: RankEvent) -> bool:
        found = self.rows.get((client_id, event.cluster_id or event.event_id))
        if found is None:
            return False
        revision, signature = found
        material = event.revision != revision and event.fact_signature != signature
        return not material

    def blocked_ids(self, event: RankEvent) -> set[str]:
        cluster = event.cluster_id or event.event_id
        return {client_id for (client_id, key) in self.rows if key == cluster and self.blocks(client_id, event)}

    def mark(self, client_id: str, event: RankEvent) -> None:
        self.rows[(client_id, event.cluster_id or event.event_id)] = (event.revision, event.fact_signature)


class RankingService:
    def __init__(
        self,
        rows: list[ClientRow],
        events: list[RankEvent],
        cards: list[CardView] | None = None,
        config: RankingConfig | None = None,
        *,
        cache_dir: Path | None = None,
        group_table: dict | None = None,
        as_of: datetime | None = None,
        persist_dir: Path | None = None,
    ) -> None:
        self.rows = list(rows)
        self.events = list(events)
        self.cards = list(cards or [])
        self.config = config or load_ranking_config()
        self.cache_dir = cache_dir
        self.group_table = group_table if group_table is not None else load_group_table(
            repo_path(self.config.scoring["paths"]["groups"])
        )
        self.as_of = as_utc(as_of or datetime(2026, 10, 22, 12, tzinfo=timezone.utc))
        self.persist_dir = persist_dir
        self.impressions = ImpressionLog()
        self._matrix: FeatureMatrix | None = None
        self._matrix_fp = ""
        self._cache: dict[tuple, object] = {}
        self._seen_profiles: dict[str, str] | None = None
        self._seen_events: dict[str, str] = {}

    @classmethod
    def load_default(cls) -> "RankingService":
        config = load_ranking_config()
        rows = default_inputs(config)
        from alfa.ranking.catalog import catalog_cards, catalog_events, catalog_rows

        if not rows:
            rows = catalog_rows()
        return cls(rows, catalog_events(), catalog_cards(), config)

    def profile_fingerprint(self) -> str:
        blob = []
        for row in self.rows:
            blob.append(
                (
                    row.client_id,
                    row.profile_version,
                    row.client_type,
                    row.conflicting_records,
                    row.has_credit,
                    row.has_leasing,
                    row.has_deposit,
                    row.has_guarantees,
                    row.has_factoring,
                    row.has_payroll,
                    row.has_rko,
                    row.industry_group,
                    tuple(row.themes or []),
                    row.okved_code,
                    row.region,
                    row.size_class,
                    row.workers_class,
                    row.fx_exposure,
                    tuple(row.fts_countries or []),
                    row.group_id,
                    row.group_size_reported,
                    row.has_payment_data,
                    row.reliability,
                    row.n_edges,
                    row.known_counterparty_volume_share,
                    tuple(sorted((row.industry_mix_out or {}).items())),
                    tuple(sorted((row.industry_mix_in or {}).items())),
                )
            )
        return hashlib.sha256(repr(blob).encode("utf-8")).hexdigest()[:16]

    def build_feature_matrix(self, as_of: datetime) -> FeatureMatrix:
        self.as_of = as_utc(as_of)
        fingerprint = self.profile_fingerprint()
        if self._matrix is None or self._matrix_fp != fingerprint:
            self._matrix = assemble_matrix(
                self.rows,
                self.config,
                self.as_of,
                cache_dir=self.cache_dir,
                group_table=self.group_table,
            )
            self._matrix_fp = fingerprint
            self._cache.clear()
        return self._matrix

    def event_map(self) -> dict[str, RankEvent]:
        return {event.event_id: event for event in self.events}

    def _arrays(self, event: RankEvent, as_of: datetime, enabled: set[str] | None = None):
        matrix = self.build_feature_matrix(as_of)
        card_key = tuple(
            (card.card_id, as_utc(card.published_at).isoformat(), tuple(card.topics)) for card in self.cards
        )
        key = (
            event.event_id,
            event.signature(),
            as_utc(as_of).isoformat(),
            self.config.config_version,
            self.config.weight_version,
            self._matrix_fp,
            card_key,
            tuple(sorted(enabled)) if enabled else None,
        )
        cached = self._cache.get(key)
        if cached is None or enabled is not None:
            # В кэше базовый скоринг без впечатлений: показ режется в ленте.
            cached = score_arrays(
                matrix,
                event,
                as_of,
                self.config,
                self.cards,
                blocked=None,
                enabled=enabled,
                cache_dir=self.cache_dir,
            )
            if enabled is None:
                self._cache[key] = cached
        return matrix, cached

    def _materialize(self, event: RankEvent, as_of: datetime, enabled: set[str] | None = None) -> list[ScoredPair]:
        matrix, arrays = self._arrays(event, as_of, enabled)
        if arrays.filtered or len(matrix) == 0:
            return []
        indexes = [i for i, flag in enumerate(arrays.evidence.tolist()) if flag]
        indexes.sort(key=lambda i: (-float(arrays.score[i]), str(matrix.client_ids[i])))
        pairs: list[ScoredPair] = []
        for index in indexes:
            view = explain_score(matrix, arrays, self.config, event, index)
            pairs.append(
                ScoredPair(
                    event_id=event.event_id,
                    client_id=str(matrix.client_ids[index]),
                    score=view["score"],
                    relevance=view["relevance"],
                    importance=view["importance"],
                    components=view["components"],
                    matched_rules=view["matched_rules"],
                    reliability=view["reliability"],
                    priority=view["priority"],
                    flags=view["flags"],
                    mechanism_ids=view["mechanism_ids"],
                    cluster_id=event.cluster_id or event.event_id,
                    revision=event.revision,
                    fact_signature=event.fact_signature,
                    hypotheses=view["hypotheses"],
                    missing_features=view["missing_features"],
                    score_confidence=view["score_confidence"],
                    config_version=self.config.config_version,
                    weight_version=self.config.weight_version,
                    group_signal=view["group_signal"],
                    graph_signal=view["graph_signal"],
                    analytics_refs=view["analytics_refs"],
                    above_threshold=view["above_threshold"],
                )
            )
        return pairs

    def score_event(
        self,
        event_id: str,
        as_of: datetime,
        top_n: int | None = None,
        clients: list[str] | None = None,
        enabled: set[str] | None = None,
    ) -> list[ScoredPair]:
        event = self.event_map()[event_id]
        self.as_of = as_utc(as_of)
        pairs = self._materialize(event, as_of, enabled)
        if clients is not None:
            allowed = set(clients)
            pairs = [pair for pair in pairs if pair.client_id in allowed]
        if top_n is not None:
            pairs = pairs[:top_n]
        if self.persist_dir is not None and enabled is None:
            above = [pair for pair in pairs if pair.above_threshold]
            persist_pairs(above, self.persist_dir)
        return pairs

    def clients_for_event(self, event_id: str, as_of: datetime, top_n: int | None = None) -> list[ScoredPair]:
        if top_n is None:
            top_n = int(self.config.scoring["feed"]["clients_per_event"])
        pairs = [pair for pair in self.score_event(event_id, as_of) if pair.above_threshold]
        return pairs[:top_n]

    def score_all(self, as_of: datetime) -> list[ScoredPair]:
        found: list[ScoredPair] = []
        for event in self.events:
            found.extend(self.score_event(event.event_id, as_of))
        self._seen_profiles = {row.client_id: row.profile_version for row in self.rows}
        self._seen_events = {event.event_id: event.signature() for event in self.events}
        return found

    def score_incremental(self, as_of: datetime) -> list[ScoredPair]:
        if self._seen_profiles is None:
            return self.score_all(as_of)
        changed_clients = {
            row.client_id
            for row in self.rows
            if self._seen_profiles.get(row.client_id) != row.profile_version or row.client_id not in self._seen_profiles
        }
        changed_events = [event for event in self.events if self._seen_events.get(event.event_id) != event.signature()]
        if not changed_clients and not changed_events:
            return []
        self._matrix = None
        self._cache.clear()
        output: list[ScoredPair] = []
        changed_ids = {event.event_id for event in changed_events}
        targets = changed_events if changed_events and not changed_clients else self.events
        for event in targets:
            pairs = self.score_event(event.event_id, as_of)
            if changed_clients and not changed_events:
                pairs = [pair for pair in pairs if pair.client_id in changed_clients]
            elif changed_clients and changed_events:
                pairs = [pair for pair in pairs if pair.event_id in changed_ids or pair.client_id in changed_clients]
            output.extend(pairs)
        self._seen_profiles = {row.client_id: row.profile_version for row in self.rows}
        self._seen_events = {event.event_id: event.signature() for event in self.events}
        return output

    def _coverage(self, client_id: str) -> float:
        matrix = self.build_feature_matrix(self.as_of)
        if client_id not in matrix.index:
            return 0.0
        index = matrix.row(client_id)
        names = list(self.config.scoring["coverage_features"])
        flags = [bool(matrix.known[name][index]) for name in names if name in matrix.known]
        if not flags:
            return 0.0
        return sum(flags) / len(flags)

    def _candidates(self, client_id: str, as_of: datetime) -> list[ScoredPair]:
        found: list[ScoredPair] = []
        for event in self.events:
            if not event_is_active(event, as_of, self.config):
                continue
            if self.impressions.blocks(client_id, event):
                continue
            for pair in self._materialize(event, as_of):
                if pair.client_id == client_id and pair.above_threshold:
                    found.append(pair)
        return found

    def build_feed(
        self,
        client_id: str,
        as_of: datetime,
        limit: int | None = None,
        role: str | None = None,
        *,
        record_impressions: bool = True,
    ) -> Feed:
        guard_client(client_id)
        self.as_of = as_utc(as_of)
        reasons = self.config.scoring["feed"]["empty_reasons"]
        low = float(self.config.scoring["feed"]["low_coverage_known_fraction"])
        if client_id not in {row.client_id for row in self.rows if row.client_type == "CLIENT" and not row.conflicting_records}:
            return self._empty_feed(client_id, reasons["low_data"], role)
        candidates = self._candidates(client_id, as_of)
        events = self.event_map()
        features = {client_id: self.feature_view(client_id)}
        candidates = apply_rerank(
            candidates,
            self.config,
            features_by_client=features,
            events=events,
            llm=None,
        )
        limit = int(self.config.scoring["feed"]["max_per_day"] if limit is None else limit)
        max_mechanism = int(self.config.scoring["feed"]["max_same_mechanism"])
        preferred = set((self.config.scoring.get("roles") or {}).get(role or "", []))

        def sort_key(pair: ScoredPair):
            liked = 0 if any(item in preferred for item in pair.mechanism_ids) else 1
            return (pair.priority != "urgent", -pair.score, liked, pair.event_id)

        ranked = sorted(candidates, key=sort_key)
        mechanism_count: Counter[str] = Counter()
        seen_clusters: set[str] = set()
        chosen: list[ScoredPair] = []
        for pair in ranked:
            if len(chosen) >= limit:
                break
            mechanisms = pair.mechanism_ids or ["_none"]
            if any(mechanism_count[item] >= max_mechanism for item in mechanisms):
                continue
            if pair.cluster_id in seen_clusters:
                continue
            chosen.append(pair)
            for item in mechanisms:
                mechanism_count[item] += 1
            seen_clusters.add(pair.cluster_id)
            if record_impressions:
                self.impressions.mark(client_id, events[pair.event_id])
        items = [self._item(pair) for pair in chosen]
        if items:
            return Feed(
                client_id=client_id,
                as_of=self.as_of,
                items=items,
                empty_state=False,
                empty_reason=None,
                config_version=self.config.config_version,
                weight_version=self.config.weight_version,
                role=role,
            )
        reason = reasons["low_data"] if self._coverage(client_id) < low else reasons["no_events"]
        return self._empty_feed(client_id, reason, role)

    def explain(self, event_id: str, client_id: str) -> Explanation:
        guard_client(client_id)
        event = self.event_map()[event_id]
        matrix = self.build_feature_matrix(self.as_of)
        if client_id not in matrix.index:
            raise KeyError(client_id)
        _matrix, arrays = self._arrays(event, self.as_of)
        view = explain_score(matrix, arrays, self.config, event, matrix.row(client_id))
        return Explanation(
            event_id=event_id,
            client_id=client_id,
            fired_rules=view["matched_rules"],
            components=view["components"],
            group_signal=view["group_signal"],
            graph_signal=view["graph_signal"],
            missing_features=view["missing_features"],
            reliability=view["reliability"],
            score_confidence=view["score_confidence"],
            config_version=self.config.config_version,
            weight_version=self.config.weight_version,
            score=view["score"],
            relevance=view["relevance"],
            importance=view["importance"],
            priority=view["priority"],
            flags=view["flags"],
            hypotheses=view["hypotheses"],
            eligible=view["eligible"],
            above_threshold=view["above_threshold"],
        )

    def feature_view(self, client_id: str) -> dict:
        matrix = self.build_feature_matrix(self.as_of)
        if client_id not in matrix.index:
            return {}
        row = matrix.rows[matrix.row(client_id)]
        return {
            "industry_group": row.industry_group,
            "themes": list(row.themes or []),
            "size_class": row.size_class,
            "region": row.region,
            "fx_known": row.fx_exposure == "known",
            "fts_countries": list(row.fts_countries or []),
            "has_credit": row.has_credit,
            "has_leasing": row.has_leasing,
            "has_deposit": row.has_deposit,
            "has_payroll": row.has_payroll,
            "has_rko": row.has_rko,
            "has_guarantees": row.has_guarantees,
            "has_factoring": row.has_factoring,
            "has_group": bool(row.group_id),
            "reliability": row.reliability,
        }

    def _item(self, pair: ScoredPair) -> FeedItem:
        reasons = []
        for mechanism in pair.mechanism_ids:
            text = self.config.reason(mechanism)
            if text not in reasons:
                reasons.append(text)
        return FeedItem(
            event_id=pair.event_id,
            client_id=pair.client_id,
            score=pair.score,
            priority=pair.priority,
            mechanism_ids=list(pair.mechanism_ids),
            reasons=reasons,
            reliability=pair.reliability,
            analytics_refs=list(pair.analytics_refs),
            flags=list(pair.flags),
            hypotheses=list(pair.hypotheses),
        )

    def _empty_feed(self, client_id: str, reason: str, role: str | None) -> Feed:
        return Feed(
            client_id=client_id,
            as_of=self.as_of,
            items=[],
            empty_state=True,
            empty_reason=reason,
            config_version=self.config.config_version,
            weight_version=self.config.weight_version,
            role=role,
        )


def persist_pairs(pairs: list[ScoredPair], directory: Path) -> dict[str, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    jsonl_path = directory / "scored_pairs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(pair.model_dump_json() + "\n")
    parquet_path: Path | None = directory / "scored_pairs.parquet"
    try:
        import pyarrow  # noqa: F401

        rows = []
        for pair in pairs:
            payload = pair.model_dump(mode="json")
            rows.append(
                {
                    "event_id": payload["event_id"],
                    "client_id": payload["client_id"],
                    "score": payload["score"],
                    "relevance": payload["relevance"],
                    "importance": payload["importance"],
                    "components": json.dumps(payload["components"], ensure_ascii=False),
                    "matched_rules": json.dumps(payload["matched_rules"], ensure_ascii=False),
                    "reliability": payload["reliability"],
                    "priority": payload["priority"],
                    "flags": json.dumps(payload["flags"], ensure_ascii=False),
                    "config_version": payload["config_version"],
                    "weight_version": payload["weight_version"],
                }
            )
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None
    return {"jsonl": str(jsonl_path), "parquet": None if parquet_path is None else str(parquet_path)}


def get_service() -> RankingService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = RankingService.load_default()
    return _SERVICE


def set_service(service: RankingService | None) -> None:
    global _SERVICE
    _SERVICE = service


@contextmanager
def using(service: RankingService):
    previous = _SERVICE
    set_service(service)
    try:
        yield service
    finally:
        set_service(previous)


def build_feature_matrix(as_of: datetime) -> FeatureMatrix:
    return get_service().build_feature_matrix(as_of)


def score_event(event_id: str, as_of: datetime, top_n: int | None = None, clients: list[str] | None = None) -> list[ScoredPair]:
    return get_service().score_event(event_id, as_of, top_n=top_n, clients=clients)


def clients_for_event(event_id: str, as_of: datetime, top_n: int | None = None) -> list[ScoredPair]:
    return get_service().clients_for_event(event_id, as_of, top_n=top_n)


def build_feed(client_id: str, as_of: datetime, limit: int | None = None, role: str | None = None) -> Feed:
    return get_service().build_feed(client_id, as_of, limit=limit, role=role)


def explain(event_id: str, client_id: str) -> Explanation:
    return get_service().explain(event_id, client_id)


def score_incremental(as_of: datetime) -> list[ScoredPair]:
    return get_service().score_incremental(as_of)


def event_from_parts(doc, extraction, magnitude: float = 0.0) -> RankEvent:
    """Сшивка NewsDoc и EventExtraction. Тип не выдумывается: пустой тип → irrelevant."""
    event_type = extraction.event_type.value if getattr(extraction, "event_type", None) else "irrelevant"
    status = extraction.extraction_status
    status_value = status.value if hasattr(status, "value") else str(status)
    published = doc.published_at or doc.fetched_at
    entities = [
        {"text": entity.text, "type": entity.type.value if hasattr(entity.type, "value") else str(entity.type)}
        for entity in extraction.entities
    ]
    facts = "|".join((fact.value or "") for fact in extraction.facts)
    return RankEvent(
        event_id=doc.id,
        event_type=event_type,
        published_at=published,
        extraction_status=status_value,
        is_business_relevant=bool(extraction.is_business_relevant),
        is_primary=bool(doc.is_primary),
        trust_tier=doc.trust_tier,
        cluster_id=doc.cluster_id or doc.id,
        revision=int(doc.revision),
        fact_signature=facts,
        magnitude=magnitude,
        n_sources=1 + len(doc.echo_sources),
        echo_count=int(doc.echo_count),
        entities=entities,
        title=doc.title,
        text=doc.text,
    )
