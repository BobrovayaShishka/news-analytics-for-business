"""Сборка profiles.parquet и graph_features.parquet одной командой."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from alfa.access import reload
from alfa.graph_features import build_graph_records, clean_payments
from alfa.profile_builder import apply_market_position, apply_profile_size, build_profiles, known_revenues
from alfa.settings import ROOT, load_profile_config, output_path, profile_config_hash, save_config

log = logging.getLogger("alfa.build")


def _read_companies(config: dict) -> pd.DataFrame:
    path = ROOT / config["inputs"]["companies"]
    return pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)


def _read_payments(config: dict) -> pd.DataFrame:
    path = ROOT / config["inputs"]["payments"]
    frame = pd.read_csv(path, sep=";", dtype={"PAYER": str, "BENEFICIARY": str})
    frame["PAY_AMT"] = pd.to_numeric(frame["PAY_AMT"], errors="coerce").fillna(0.0)
    return frame


def _ensure_computed(config: dict, edges: pd.DataFrame, profiles: list[dict]) -> dict:
    computed = config.setdefault("computed", {})
    changed = False
    if not computed.get("size_edges"):
        quantiles = [float(item) for item in config["size_class"]["quantiles"]]
        revenues = known_revenues(profiles)
        computed["size_edges"] = [float(item) for item in np.quantile(np.asarray(revenues, dtype=float), quantiles)]
        changed = True
    if computed.get("pay_amt_p99") is None:
        computed["pay_amt_p99"] = float(edges["PAY_AMT"].quantile(0.99))
        changed = True
    if changed:
        save_config(config)
    return config


def _write_records(records: list[dict], path) -> None:
    frame = pd.DataFrame(
        {
            "client_id": [item["client_id"] for item in records],
            "profile_version": [item["profile_version"] for item in records],
            "payload": [json.dumps(item, ensure_ascii=False) for item in records],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def build_all() -> dict:
    config = load_profile_config()
    log.info("читаю компании и платежи")
    companies = _read_companies(config)
    payments = _read_payments(config)
    edges, clean_stats = clean_payments(payments)
    profiles, profile_stats = build_profiles(companies, config)
    config = _ensure_computed(config, edges, profiles)
    version = profile_config_hash(config)
    apply_market_position(profiles)
    apply_profile_size(profiles, config, version)
    profile_ids = {item["client_id"] for item in profiles}
    seen = (set(payments["PAYER"]) | set(payments["BENEFICIARY"])) & profile_ids
    log.info("граф, рёбер %s", clean_stats["directed_edges"])
    records, graph_stats = build_graph_records(
        edges, profiles, config, version, float(config["computed"]["pay_amt_p99"]), profile_ids
    )
    _write_records(profiles, output_path(config, "profiles"))
    _write_records(records, output_path(config, "graph_features"))
    both = [
        item
        for item in records
        if item["n_edges_in"] and item["n_edges_out"] and item["known_counterparty_volume_share_total"] is not None
    ]
    median_known = float(np.median([item["known_counterparty_volume_share_total"] for item in both])) if both else None
    stats = {
        "profile_version": version,
        "profiles": profile_stats,
        "payments": clean_stats,
        "graph": graph_stats,
        "profile_ids_in_payment_file": len(seen),
        "companies_with_edges": len(records),
        "median_known_share_both_directions": median_known,
    }
    stats_path = output_path(config, "stats")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    reload()
    log.info("готово: %s профилей, %s компаний с рёбрами", len(profiles), len(records))
    return stats
