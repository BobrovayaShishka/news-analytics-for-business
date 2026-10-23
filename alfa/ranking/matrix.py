"""Матрица признаков CLIENT. Пропуски — NaN и маска known, не нули."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alfa.kb.index import HashingEmbedder
from alfa.ranking.config import RankingConfig, repo_path
from alfa.ranking.entities import country_code, themes_for_okved
from alfa.ranking.models import ClientRow
from alfa.settings import industry_group, load_okved_map, load_profile_config, norm_label, norm_place, payment_reliability, size_band

SEED_INDUSTRIES = (
    "construction_buildings",
    "gold_mining",
    "land_transport",
    "manufacturing_machinery",
    "mining_metal",
    "mining_oil_gas",
    "real_estate",
    "retail",
    "wholesale",
    "wholesale_fuel",
    "wholesale_metals",
)

PRODUCTS = (
    "has_credit",
    "has_leasing",
    "has_deposit",
    "has_guarantees",
    "has_factoring",
    "has_payroll",
    "has_rko",
)

RELIABILITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
RELIABILITY_NAME = {rank: name for name, rank in RELIABILITY_RANK.items()}


def resolve_themes(row: ClientRow, themes_cfg: dict) -> tuple[list[str], bool]:
    if row.themes is not None:
        return list(row.themes), True
    industry_known = bool(row.industry_group) and row.industry_group not in {"unknown", ""}
    themes = themes_for_okved(row.okved_code, row.industry_group, themes_cfg)
    return themes, bool(themes) or industry_known


def embedding_text(row: ClientRow, themes: list[str]) -> str:
    """Текст для семантического канала. Готовое описание профиля важнее этой сборки."""
    if row.text:
        return row.text
    parts: list[str] = []
    if row.industry_group and row.industry_group not in {"unknown", ""}:
        parts.append(f"отрасль {row.industry_group}")
    if themes:
        parts.append("темы " + " ".join(themes))
    if row.region:
        parts.append(f"регион {row.region}")
    if row.size_class:
        parts.append(f"размер {row.size_class}")
    if row.workers_class is not None:
        parts.append(f"численность {row.workers_class}")
    for name in PRODUCTS:
        value = getattr(row, name)
        if value is True:
            parts.append(name.removeprefix("has_"))
        elif value is False:
            parts.append("нет " + name.removeprefix("has_"))
    if row.fx_exposure == "known":
        parts.append("валютный контур известен")
        if row.fts_countries:
            parts.append("страны " + " ".join(row.fts_countries))
    elif row.fx_exposure is None:
        parts.append("валютный контур неизвестен")
    return " ".join(parts) or "профиль без признаков"


def embed_texts(texts: list[str], cache_dir: Path | None, dim: int) -> tuple[np.ndarray, str]:
    """Детерминированный hashing, не bge-m3. Один и тот же текст → один и тот же вектор."""
    backend = "hashing-non-bge"
    embedder = HashingEmbedder(dim=dim)
    if not texts:
        return np.zeros((0, dim), dtype=np.float32), backend
    digest = hashlib.sha1((f"{backend}\n{dim}\n" + "\n".join(texts)).encode("utf-8")).hexdigest()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{digest}.npy"
        if path.exists():
            cached = np.load(path)
            return np.asarray(cached, dtype=np.float32), backend
    vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    if cache_dir is not None:
        np.save(cache_dir / f"{digest}.npy", vectors)
    return vectors, backend


@dataclass
class FeatureMatrix:
    as_of: datetime
    client_ids: np.ndarray
    index: dict[str, int]
    known: dict[str, np.ndarray]
    values: dict[str, np.ndarray]
    missing: dict[str, np.ndarray]
    industry_code: np.ndarray
    industry_labels: list[str]
    industry_index: dict[str, int]
    industry_one_hot: np.ndarray
    theme_labels: list[str]
    theme_index: dict[str, int]
    theme_one_hot: np.ndarray
    region_code: np.ndarray
    region_labels: list[str]
    region_index: dict[str, int]
    size_code: np.ndarray
    size_labels: list[str]
    size_index: dict[str, int]
    size_one_hot: np.ndarray
    country_labels: list[str]
    country_index: dict[str, int]
    countries: np.ndarray
    mix_in: np.ndarray
    mix_out: np.ndarray
    group_code: np.ndarray
    group_any: dict[str, np.ndarray]
    group_any_known: dict[str, np.ndarray]
    group_member_count: np.ndarray
    has_group: np.ndarray
    reliability_rank: np.ndarray
    reliability_coeff: np.ndarray
    reliability_label: np.ndarray
    known_share: np.ndarray
    n_edges: np.ndarray
    embeddings: np.ndarray
    embedder_backend: str
    embed_seconds: float
    texts: list[str]
    currencies: list[set[str]]
    config_version: str
    weight_version: str
    rows: list[ClientRow]

    def __len__(self) -> int:
        return int(self.client_ids.shape[0])

    def row(self, client_id: str) -> int:
        return int(self.index[client_id])


def _size_labels() -> list[str]:
    return [str(item) for item in load_profile_config()["size_class"]["classes"]]


def _size_edges() -> list[float]:
    return [float(item) for item in load_profile_config()["computed"]["size_edges"]]


def size_from_revenue(revenue: float | None) -> str | None:
    if revenue is None:
        return None
    return size_band(revenue, _size_edges(), _size_labels())


def _workers(label: str | None) -> int | None:
    if not label:
        return None
    mapping = {norm_label(key): int(value) for key, value in load_profile_config()["workers_scale"]["map"].items()}
    return mapping.get(norm_label(label))


def prepare_rows(rows: list[ClientRow]) -> list[ClientRow]:
    """Только CLIENT, без конфликтующих записей и технических дублей."""
    seen: set[str] = set()
    kept: list[ClientRow] = []
    for row in rows:
        if row.client_id in seen:
            continue
        seen.add(row.client_id)
        if row.conflicting_records:
            continue
        if row.client_type != "CLIENT":
            continue
        kept.append(row)
    kept.sort(key=lambda item: item.client_id)
    return kept


def _blank_numeric(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float32)


def _overlay_groups(rows: list[ClientRow], table: dict[str, dict]) -> None:
    """Внешняя таблица групп дополняет агрегаты, не подменяет id членов."""
    if not table:
        return
    for row in rows:
        extra = table.get(row.group_id or "")
        if not extra:
            continue
        if extra.get("member_count") is not None:
            reported = int(extra["member_count"])
            row.group_size_reported = max(int(row.group_size_reported or 0), reported)
        if row.revenue is None and extra.get("revenue") is not None and False:
            # Выручку клиента из группы не выдумываем.
            pass


def load_group_table(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_parquet(path)
    table: dict[str, dict] = {}
    for record in frame.to_dict(orient="records"):
        group_id = str(record.get("group_id") or record.get("GROUP_ID") or "")
        if not group_id:
            continue
        table[group_id] = record
    return table


def assemble_matrix(
    rows: list[ClientRow],
    config: RankingConfig,
    as_of: datetime,
    *,
    cache_dir: Path | None = None,
    group_table: dict[str, dict] | None = None,
) -> FeatureMatrix:
    rows = prepare_rows(rows)
    _overlay_groups(rows, group_table or {})
    n = len(rows)
    theme_cfg = config.themes
    resolved = [resolve_themes(row, theme_cfg) for row in rows]
    texts = [embedding_text(row, themes) for row, (themes, _) in zip(rows, resolved, strict=True)]
    dim = int(config.scoring["embedding"]["dim"])
    embed_started = __import__("time").perf_counter()
    embeddings, backend = embed_texts(texts, cache_dir, dim)
    embed_seconds = __import__("time").perf_counter() - embed_started

    industry_labels = sorted(
        set(SEED_INDUSTRIES)
        | {row.industry_group for row in rows if row.industry_group and row.industry_group not in {"unknown", ""}}
        | {key for row in rows for key in (row.industry_mix_in or {})}
        | {key for row in rows for key in (row.industry_mix_out or {})}
    )
    industry_index = {label: i for i, label in enumerate(industry_labels)}
    theme_labels = sorted(
        {theme for themes, _ in resolved for theme in themes}
        | {str(item) for bucket in (theme_cfg.get("okved_prefixes") or {}).values() for item in bucket}
        | {str(item) for bucket in (theme_cfg.get("industry_group") or {}).values() for item in bucket}
    )
    theme_index = {label: i for i, label in enumerate(theme_labels)}
    countries_fixed = [str(code) for code in config.scoring.get("country_dictionary") or []]
    extra_countries = sorted(
        {
            code
            for row in rows
            for code in (row.fts_countries or [])
            if code not in countries_fixed
        }
    )
    country_labels = countries_fixed + extra_countries
    country_index = {label: i for i, label in enumerate(country_labels)}
    region_labels = sorted({row.region for row in rows if row.region})
    region_index = {label: i for i, label in enumerate(region_labels)}
    size_labels = _size_labels()
    size_index = {label: i for i, label in enumerate(size_labels)}

    known: dict[str, np.ndarray] = {}
    values: dict[str, np.ndarray] = {}
    for name in PRODUCTS:
        raw = [getattr(row, name) for row in rows]
        known[name] = np.fromiter((item is not None for item in raw), dtype=bool, count=n)
        values[name] = np.array(
            [np.nan if item is None else (1.0 if item else 0.0) for item in raw],
            dtype=np.float32,
        )

    industry_known = np.fromiter(
        (bool(row.industry_group) and row.industry_group not in {"unknown", ""} for row in rows),
        dtype=bool,
        count=n,
    )
    industry_code = np.array(
        [
            industry_index[row.industry_group] if industry_known[i] and row.industry_group in industry_index else -1
            for i, row in enumerate(rows)
        ],
        dtype=np.int16,
    )
    industry_one_hot = np.zeros((n, max(len(industry_labels), 1)), dtype=np.float32)
    hit = industry_code >= 0
    if hit.any():
        industry_one_hot[np.flatnonzero(hit), industry_code[hit]] = 1.0
    known["industry_group"] = industry_known
    values["industry_group"] = np.where(industry_known, industry_code, np.nan).astype(np.float32)

    theme_known = np.fromiter((flag for _, flag in resolved), dtype=bool, count=n)
    theme_one_hot = np.zeros((n, max(len(theme_labels), 1)), dtype=np.float32)
    theme_rows: list[int] = []
    theme_cols: list[int] = []
    for i, (themes, flag) in enumerate(resolved):
        if not flag:
            continue
        for theme in themes:
            column = theme_index.get(theme)
            if column is not None:
                theme_rows.append(i)
                theme_cols.append(column)
    if theme_rows:
        theme_one_hot[theme_rows, theme_cols] = 1.0
    known["industry_theme"] = theme_known
    theme_count = theme_one_hot.sum(axis=1).astype(np.float32)
    values["industry_theme"] = np.where(theme_known, theme_count, np.nan).astype(np.float32)

    size_known = np.fromiter((row.size_class in size_index for row in rows), dtype=bool, count=n)
    size_code = np.array(
        [size_index[row.size_class] if row.size_class in size_index else -1 for row in rows],
        dtype=np.int16,
    )
    size_one_hot = np.zeros((n, max(len(size_labels), 1)), dtype=np.float32)
    if size_known.any():
        size_one_hot[np.flatnonzero(size_known), size_code[size_known]] = 1.0
    known["size_class"] = size_known
    values["size_class"] = np.where(size_known, size_code, np.nan).astype(np.float32)

    workers = np.array(
        [np.nan if row.workers_class is None else float(row.workers_class) for row in rows],
        dtype=np.float32,
    )
    known["workers_class"] = np.fromiter((row.workers_class is not None for row in rows), dtype=bool, count=n)
    values["workers_class"] = workers

    region_known = np.fromiter((bool(row.region) for row in rows), dtype=bool, count=n)
    region_code = np.array(
        [region_index[row.region] if row.region in region_index else -1 for row in rows],
        dtype=np.int16,
    )
    known["region"] = region_known
    values["region"] = np.where(region_known, region_code, np.nan).astype(np.float32)

    fx_known = np.fromiter((row.fx_exposure == "known" for row in rows), dtype=bool, count=n)
    fx_values = np.full(n, np.nan, dtype=np.float32)
    fx_values[fx_known] = 1.0
    known["fx_exposure"] = fx_known
    values["fx_exposure"] = fx_values
    confidence_map = {"none": 0.0, "low": 0.5, "high": 1.0}
    fx_confidence = np.array(
        [confidence_map.get(row.fx_confidence, np.nan) for row in rows],
        dtype=np.float32,
    )
    known["fx_confidence"] = np.fromiter((row.fx_confidence in confidence_map for row in rows), dtype=bool, count=n)
    values["fx_confidence"] = fx_confidence

    countries = np.zeros((n, max(len(country_labels), 1)), dtype=bool)
    fts_known = np.fromiter((row.fts_countries is not None for row in rows), dtype=bool, count=n)
    country_rows: list[int] = []
    country_cols: list[int] = []
    for i, row in enumerate(rows):
        if not row.fts_countries:
            continue
        for code in row.fts_countries:
            column = country_index.get(code)
            if column is not None:
                country_rows.append(i)
                country_cols.append(column)
    if country_rows:
        countries[country_rows, country_cols] = True
    known["fts_countries"] = fts_known

    key_owner = np.array(
        [np.nan if row.key_owner_known is None else (1.0 if row.key_owner_known else 0.0) for row in rows],
        dtype=np.float32,
    )
    known["key_owner_known"] = np.fromiter((row.key_owner_known is not None for row in rows), dtype=bool, count=n)
    values["key_owner_known"] = key_owner

    pay_values = np.array(
        [np.nan if row.has_payment_data is None else (1.0 if row.has_payment_data else 0.0) for row in rows],
        dtype=np.float32,
    )
    known["has_payment_data"] = np.fromiter((row.has_payment_data is not None for row in rows), dtype=bool, count=n)
    values["has_payment_data"] = pay_values

    mix_in = np.zeros((n, max(len(industry_labels), 1)), dtype=np.float32)
    mix_out = np.zeros((n, max(len(industry_labels), 1)), dtype=np.float32)
    _scatter_mix(rows, industry_index, mix_in, mix_out)

    known_share = np.array(
        [np.nan if row.known_counterparty_volume_share is None else float(row.known_counterparty_volume_share) for row in rows],
        dtype=np.float32,
    )
    fts_cp = np.array(
        [np.nan if row.fts_counterparty_volume_share is None else float(row.fts_counterparty_volume_share) for row in rows],
        dtype=np.float32,
    )
    n_edges = np.array([np.nan if row.n_edges is None else float(row.n_edges) for row in rows], dtype=np.float32)
    known["known_counterparty_volume_share"] = np.isfinite(known_share)
    values["known_counterparty_volume_share"] = known_share
    known["fts_counterparty_volume_share"] = np.isfinite(fts_cp)
    values["fts_counterparty_volume_share"] = fts_cp
    known["n_edges"] = np.isfinite(n_edges)
    values["n_edges"] = n_edges

    coeff_table = config.scoring["graph"]["reliability_coeff"]
    reliability_label = np.array([row.reliability or "" for row in rows], dtype=object)
    reliability_known = np.fromiter((row.reliability in RELIABILITY_RANK for row in rows), dtype=bool, count=n)
    reliability_rank = np.array([RELIABILITY_RANK.get(row.reliability, 0) for row in rows], dtype=np.int16)
    reliability_coeff = np.array(
        [float(coeff_table[row.reliability]) if row.reliability in coeff_table else 0.0 for row in rows],
        dtype=np.float32,
    )
    known["reliability"] = reliability_known
    values["reliability"] = np.where(reliability_known, reliability_rank, np.nan).astype(np.float32)

    group_code, group_any, group_any_known, member_count, group_revenue, group_size_code, group_size_known = _group_layer(
        rows, resolved, size_index, group_table or {}
    )
    has_group = np.array([bool(row.group_id) for row in rows], dtype=bool)
    known["has_group"] = np.ones(n, dtype=bool)
    values["has_group"] = has_group.astype(np.float32)
    for name, column in group_any.items():
        values[f"group_any.{name}"] = np.where(group_any_known[name], column.astype(np.float32), np.nan).astype(np.float32)
        known[f"group_any.{name}"] = group_any_known[name]
    known["group_member_count"] = has_group
    values["group_member_count"] = np.where(has_group, member_count, np.nan).astype(np.float32)
    revenue_known = has_group & np.isfinite(group_revenue)
    known["group_revenue"] = revenue_known
    values["group_revenue"] = np.where(revenue_known, group_revenue, np.nan).astype(np.float32)
    known["group_size_class"] = group_size_known
    values["group_size_class"] = np.where(group_size_known, group_size_code, np.nan).astype(np.float32)

    intra = np.array(
        [np.nan if row.intra_group_payment_share is None else float(row.intra_group_payment_share) for row in rows],
        dtype=np.float32,
    )
    known["intra_group_payment_share"] = np.isfinite(intra)
    values["intra_group_payment_share"] = intra

    currencies = [set(row.currencies or []) for row in rows]
    missing = {name: np.logical_not(mask) for name, mask in known.items()}
    client_ids = np.array([row.client_id for row in rows], dtype=object)
    return FeatureMatrix(
        as_of=as_of,
        client_ids=client_ids,
        index={client_id: i for i, client_id in enumerate(client_ids.tolist())},
        known=known,
        values=values,
        missing=missing,
        industry_code=industry_code,
        industry_labels=industry_labels,
        industry_index=industry_index,
        industry_one_hot=industry_one_hot,
        theme_labels=theme_labels,
        theme_index=theme_index,
        theme_one_hot=theme_one_hot,
        region_code=region_code,
        region_labels=region_labels,
        region_index=region_index,
        size_code=size_code,
        size_labels=size_labels,
        size_index=size_index,
        size_one_hot=size_one_hot,
        country_labels=country_labels,
        country_index=country_index,
        countries=countries,
        mix_in=mix_in,
        mix_out=mix_out,
        group_code=group_code,
        group_any=group_any,
        group_any_known=group_any_known,
        group_member_count=member_count.astype(np.float32),
        has_group=has_group,
        reliability_rank=reliability_rank,
        reliability_coeff=reliability_coeff,
        reliability_label=reliability_label,
        known_share=known_share,
        n_edges=n_edges,
        embeddings=embeddings,
        embedder_backend=backend,
        embed_seconds=embed_seconds,
        texts=texts,
        currencies=currencies,
        config_version=config.config_version,
        weight_version=config.weight_version,
        rows=rows,
    )


def _scatter_mix(rows: list[ClientRow], industry_index: dict[str, int], mix_in: np.ndarray, mix_out: np.ndarray) -> None:
    in_rows: list[int] = []
    in_cols: list[int] = []
    in_vals: list[float] = []
    out_rows: list[int] = []
    out_cols: list[int] = []
    out_vals: list[float] = []
    for index, row in enumerate(rows):
        for name, share in (row.industry_mix_in or {}).items():
            column = industry_index.get(name)
            if column is not None:
                in_rows.append(index)
                in_cols.append(column)
                in_vals.append(float(share))
        for name, share in (row.industry_mix_out or {}).items():
            column = industry_index.get(name)
            if column is not None:
                out_rows.append(index)
                out_cols.append(column)
                out_vals.append(float(share))
    if in_rows:
        mix_in[in_rows, in_cols] = in_vals
    if out_rows:
        mix_out[out_rows, out_cols] = out_vals


def _encode_flag(value: bool | None) -> int:
    if value is None:
        return -1
    return 1 if value else 0


def _group_layer(rows, resolved, size_index, group_table):
    n = len(rows)
    group_ids = [row.group_id or "" for row in rows]
    labels = sorted({group_id for group_id in group_ids if group_id})
    code_of = {label: index for index, label in enumerate(labels)}
    group_code = np.array([code_of.get(group_id, -1) for group_id in group_ids], dtype=np.int32)
    names = ("has_credit", "has_leasing", "has_deposit", "has_fts", "has_payroll", "has_guarantees")
    group_any = {name: np.zeros(n, dtype=bool) for name in names}
    group_any_known = {name: np.zeros(n, dtype=bool) for name in names}
    member_count = np.zeros(n, dtype=np.float32)
    group_revenue = np.full(n, np.nan, dtype=np.float32)
    group_size_code = np.full(n, -1, dtype=np.int16)
    group_size_known = np.zeros(n, dtype=bool)
    chosen = np.flatnonzero(group_code >= 0)
    if chosen.size == 0:
        return group_code, group_any, group_any_known, member_count, group_revenue, group_size_code, group_size_known
    frame = pd.DataFrame(
        {
            "g": group_code[chosen],
            "reported": [int(rows[i].group_size_reported or 0) for i in chosen],
            "revenue": [np.nan if rows[i].revenue is None else float(rows[i].revenue) for i in chosen],
            "has_credit": [_encode_flag(rows[i].has_credit) for i in chosen],
            "has_leasing": [_encode_flag(rows[i].has_leasing) for i in chosen],
            "has_deposit": [_encode_flag(rows[i].has_deposit) for i in chosen],
            "has_payroll": [_encode_flag(rows[i].has_payroll) for i in chosen],
            "has_guarantees": [_encode_flag(rows[i].has_guarantees) for i in chosen],
            "has_fts": [
                _encode_flag(True if rows[i].fx_exposure == "known" else (False if rows[i].fx_exposure is not None else None))
                for i in chosen
            ],
        }
    )
    grouped = frame.groupby("g", sort=False)
    observed = grouped["reported"].transform("size").to_numpy()
    reported = grouped["reported"].transform("max").to_numpy()
    table_by_code = {code_of[key]: value for key, value in group_table.items() if key in code_of}
    if table_by_code:
        table_n = np.array([int((table_by_code.get(int(code)) or {}).get("member_count") or 0) for code in frame["g"]], dtype=np.float64)
    else:
        table_n = np.zeros(len(frame), dtype=np.float64)
    member_count[chosen] = np.maximum(np.maximum(observed, reported), table_n).astype(np.float32)
    revenue_sum = grouped["revenue"].transform("sum").to_numpy()
    revenue_known_count = grouped["revenue"].transform("count").to_numpy()
    if table_by_code:
        table_revenue = np.array(
            [
                np.nan
                if (table_by_code.get(int(code)) or {}).get("revenue") in (None, "")
                else float((table_by_code.get(int(code)) or {}).get("revenue"))
                for code in frame["g"]
            ],
            dtype=np.float64,
        )
        use_table = (revenue_known_count == 0) & np.isfinite(table_revenue)
        revenue_sum = np.where(use_table, table_revenue, revenue_sum)
        revenue_known_count = np.where(use_table, 1, revenue_known_count)
    edges = np.array(_size_edges(), dtype=np.float64)
    size_labels = _size_labels()
    known_revenue = revenue_known_count > 0
    group_revenue[chosen] = np.where(known_revenue, revenue_sum, np.nan).astype(np.float32)
    if known_revenue.any() and edges.size:
        slots = np.searchsorted(edges, np.nan_to_num(revenue_sum, nan=0.0), side="left")
        slots = np.clip(slots, 0, len(size_labels) - 1)
        coded = np.array([size_index.get(size_labels[int(slot)], -1) for slot in slots], dtype=np.int16)
        keep = known_revenue & (coded >= 0)
        group_size_code[chosen[keep]] = coded[keep]
        group_size_known[chosen[keep]] = True
    for name in names:
        maximum = grouped[name].transform("max").to_numpy()
        minimum = grouped[name].transform("min").to_numpy()
        is_known = (maximum == 1) | (minimum == 0)
        value = maximum == 1
        if table_by_code:
            forced = np.array([bool((table_by_code.get(int(code)) or {}).get(name) is True) for code in frame["g"]], dtype=bool)
            is_known = is_known | forced
            value = np.where(forced, True, value)
        group_any[name][chosen] = value
        group_any_known[name][chosen] = is_known
    _ = resolved
    return group_code, group_any, group_any_known, member_count, group_revenue, group_size_code, group_size_known


def row_from_built_profile(profile: dict, graph: dict | None, config: RankingConfig) -> ClientRow | None:
    if profile.get("conflicting_records"):
        return None
    attributes = profile["attributes"]
    if attributes.get("client_type") != "CLIENT":
        return None
    products = profile["products"]
    fx = profile["fx"]
    exposure = "known" if fx.get("exposure") == "known" else None
    countries = None
    if exposure == "known":
        countries = [country_code(str(item), config) for item in fx.get("countries") or []]
    industry = attributes.get("industry_group")
    if industry in {None, "", "unknown"}:
        industry = None
    size = attributes.get("size_class")
    if size in {None, "", "unknown"}:
        size = None
    graph_bits = _graph_bits(graph)
    return ClientRow(
        client_id=str(profile["client_id"]),
        client_type="CLIENT",
        conflicting_records=False,
        profile_version=str(profile.get("profile_version") or ""),
        has_credit=products.get("has_credit"),
        has_leasing=products.get("has_leasing"),
        has_deposit=products.get("has_deposit"),
        has_guarantees=products.get("has_guarantees"),
        has_factoring=products.get("has_factoring"),
        has_payroll=products.get("has_payroll"),
        has_rko=products.get("has_rko"),
        industry_group=industry,
        okved_code=attributes.get("okved_code"),
        region=attributes.get("region"),
        size_class=size,
        workers_class=attributes.get("workers_class"),
        revenue=attributes.get("revenue"),
        fx_exposure=exposure,
        fx_confidence=fx.get("confidence"),
        fts_countries=countries,
        group_id=attributes.get("group_id"),
        group_size_reported=attributes.get("group_size"),
        key_owner_known=None,
        **graph_bits,
    )


def _graph_bits(graph: dict | None) -> dict:
    if not graph or not graph.get("has_payment_data"):
        # Нет платёжных строк: известное отсутствие, доли не подменяются нулём.
        return {
            "has_payment_data": False,
            "reliability": "none",
            "industry_mix_in": {},
            "industry_mix_out": {},
        }
    fts_values = [
        graph.get("fts_counterparty_volume_share_in"),
        graph.get("fts_counterparty_volume_share_out"),
    ]
    fts_numbers = [float(item) for item in fts_values if isinstance(item, (int, float))]
    edges_in = graph.get("n_edges_in") or 0
    edges_out = graph.get("n_edges_out") or 0
    return {
        "has_payment_data": True,
        "industry_mix_in": dict(graph.get("industry_mix_in") or {}),
        "industry_mix_out": dict(graph.get("industry_mix_out") or {}),
        "fts_counterparty_volume_share": max(fts_numbers) if fts_numbers else None,
        "known_counterparty_volume_share": graph.get("known_counterparty_volume_share_total"),
        "reliability": graph.get("reliability_total") or "none",
        "n_edges": int(edges_in) + int(edges_out),
        "intra_group_payment_share": graph.get("same_group_volume_share"),
    }


def row_from_demo(payload: dict, config: RankingConfig) -> ClientRow:
    extract = payload.get("extract") or {}
    products = payload.get("alfa_products") or {}
    trade = payload.get("foreign_trade") or {}
    industry = payload.get("industry") or {}
    region = payload.get("region") or {}
    okved_map = load_okved_map()
    okved = industry.get("okved_code")
    group = industry_group(okved, okved_map) if okved else None
    if group in {None, "", "unknown"}:
        group = None
    fts_flag = trade.get("fts_flag")
    countries = None
    exposure = None
    if fts_flag is True:
        exposure = "known"
        countries = [country_code(str(item), config) for item in trade.get("countries_observed") or []]
    graph = _demo_graph(payload)
    revenue = extract.get("revenue_rub")
    return ClientRow(
        client_id=str(payload["company_id"]),
        client_type=str(extract.get("client_type") or "CLIENT"),
        profile_version="demo-json",
        has_credit=products.get("loans"),
        has_leasing=products.get("leasing"),
        has_deposit=products.get("deposits"),
        has_guarantees=products.get("guarantees"),
        has_factoring=products.get("factoring"),
        has_payroll=products.get("payroll"),
        has_rko=products.get("rko"),
        industry_group=group,
        okved_code=okved,
        region=norm_place(region.get("region")),
        size_class=size_from_revenue(float(revenue) if revenue is not None else None),
        workers_class=_workers(extract.get("employees_fns")),
        revenue=float(revenue) if revenue is not None else None,
        fx_exposure=exposure,
        fx_confidence="high" if countries else None,
        fts_countries=countries,
        currencies=[str(item).upper() for item in (payload.get("fx") or {}).get("currencies") or []],
        group_id=extract.get("group_id"),
        group_size_reported=extract.get("group_size"),
        key_owner_known=extract.get("key_owner_mark"),
        **graph,
    )


def _demo_graph(payload: dict) -> dict:
    block = payload.get("counterparties") or {}
    top = block.get("top") or []
    outbound = int(block.get("outbound_count") or 0)
    inbound = int(block.get("inbound_count") or 0)
    if not top and outbound + inbound == 0:
        return {"has_payment_data": False, "reliability": "none", "n_edges": 0}
    okved_map = load_okved_map()
    mix_in: dict[str, float] = {}
    mix_out: dict[str, float] = {}
    total = 0.0
    known = 0.0
    for item in top:
        amount = float(item.get("amount_rub") or 0.0)
        total += amount
        if item.get("in_company_table"):
            known += amount
        code = item.get("okved_code")
        if not code:
            continue
        group = industry_group(str(code), okved_map)
        if not group or group in {"unknown", "other"}:
            continue
        bucket = mix_out if item.get("direction") == "outbound" else mix_in
        bucket[group] = bucket.get(group, 0.0) + amount

    def _share(mix: dict[str, float]) -> dict[str, float]:
        base = sum(mix.values())
        if base <= 0:
            return {}
        return {key: value / base for key, value in mix.items()}

    share = None if total <= 0 else known / total
    edges = outbound + inbound
    return {
        "has_payment_data": True,
        "industry_mix_in": _share(mix_in),
        "industry_mix_out": _share(mix_out),
        "known_counterparty_volume_share": share,
        "reliability": payment_reliability(share, edges, load_profile_config()["reliability"]),
        "n_edges": edges,
    }


def load_profile_rows(profiles_path: Path, graph_path: Path, config: RankingConfig) -> list[ClientRow]:
    frame = pd.read_parquet(profiles_path)
    graphs: dict[str, dict] = {}
    if graph_path.exists():
        graph_frame = pd.read_parquet(graph_path)
        graphs = {str(row.client_id): json.loads(row.payload) for row in graph_frame.itertuples(index=False)}
    rows: list[ClientRow] = []
    for payload in frame["payload"]:
        profile = json.loads(payload)
        row = row_from_built_profile(profile, graphs.get(str(profile["client_id"])), config)
        if row is not None:
            rows.append(row)
    return rows


def _demo_paths(directory: Path) -> list[Path]:
    index_path = directory / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        paths: list[Path] = []
        for item in index.get("clients") or []:
            raw = Path(str(item["file"]))
            paths.append(raw if raw.is_absolute() else repo_path(raw))
        return paths
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.json") if path.name != "index.json")


def load_demo_rows(directory: Path, config: RankingConfig) -> list[ClientRow]:
    rows: list[ClientRow] = []
    for path in _demo_paths(directory):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "company_id" not in payload:
            continue
        rows.append(row_from_demo(payload, config))
    return rows


def default_inputs(config: RankingConfig) -> list[ClientRow]:
    paths = config.scoring["paths"]
    profiles = repo_path(paths["profiles"])
    graphs = repo_path(paths["graph_features"])
    if profiles.exists():
        return load_profile_rows(profiles, graphs, config)
    demo = load_demo_rows(repo_path(paths["clients_dir"]), config)
    if demo:
        return demo
    return []
