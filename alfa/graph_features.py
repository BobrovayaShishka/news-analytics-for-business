"""Признаки платёжного окружения. Доли считаются по обрезанному объёму."""

from __future__ import annotations

import logging

import pandas as pd

from alfa.settings import payment_reliability

log = logging.getLogger("alfa.graph")


def clean_payments(payments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    self_mask = payments["PAYER"] == payments["BENEFICIARY"]
    kept = payments.loc[~self_mask]
    before = len(kept)
    edges = kept.groupby(["PAYER", "BENEFICIARY"], as_index=False)[["PAY_AMT"]].sum()
    stats = {
        "payment_rows": int(len(payments)),
        "self_payments_removed": int(self_mask.sum()),
        "duplicate_rows_aggregated": int(before - len(edges)),
        "directed_edges": int(len(edges)),
    }
    return edges, stats


def degree_and_hubs(edges: pd.DataFrame, percentile: float) -> tuple[float, set[str]]:
    forward = edges[["PAYER", "BENEFICIARY"]]
    backward = edges[["BENEFICIARY", "PAYER"]].rename(columns={"BENEFICIARY": "PAYER", "PAYER": "BENEFICIARY"})
    undirected = pd.concat([forward, backward], ignore_index=True).drop_duplicates()
    degree = undirected.groupby("PAYER").size()
    cutoff = float(degree.quantile(percentile / 100))
    hubs = set(degree[degree > cutoff].index)
    return cutoff, hubs


def profile_attributes(profiles: list[dict]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        if profile["conflicting_records"]:
            continue
        attributes = profile["attributes"]
        fx = profile["fx"]
        fts_flag = fx["exposure"] == "known" and (fx["confidence"] == "high" or not fx["countries"])
        rows.append(
            {
                "counterparty": profile["client_id"],
                "in_profile": True,
                "industry_group": attributes["industry_group"],
                "region": attributes["region"],
                "fts_flag": fts_flag,
                "countries": "\x1f".join(fx["countries"]),
                "group_id": attributes["group_id"],
            }
        )
    return pd.DataFrame(rows).set_index("counterparty")


def _prepare(frame: pd.DataFrame, attributes: pd.DataFrame, hubs: set[str]) -> pd.DataFrame:
    attached = frame.join(attributes, on="counterparty")
    attached["in_profile"] = attached["in_profile"].fillna(False).astype(bool)
    attached["is_hub"] = attached["counterparty"].isin(hubs)
    attached["company_region"] = attached["company"].map(attributes["region"])
    attached["company_group"] = attached["company"].map(attributes["group_id"])
    attached["fts_flag"] = attached["fts_flag"].where(attached["in_profile"], False)
    attached["fts_flag"] = attached["fts_flag"].fillna(False).astype(bool)
    totals = attached.groupby("company", sort=False)["clip"].transform("sum")
    attached["share"] = 0.0
    positive = totals > 0
    attached.loc[positive, "share"] = attached.loc[positive, "clip"] / totals[positive]
    attached["rank"] = attached.groupby("company", sort=False)["clip"].rank(method="first", ascending=False)
    return attached


def _industry_mix(frame: pd.DataFrame, exclude_hubs: bool) -> tuple[dict[str, dict[str, float]], dict[str, dict], dict[str, float]]:
    known = frame["in_profile"]
    mix = frame[known & ~frame["is_hub"]] if exclude_hubs else frame[known]
    mixes: dict[str, dict[str, float]] = {}
    tops: dict[str, dict] = {}
    fts: dict[str, float] = {}
    if mix.empty:
        return mixes, tops, fts
    denominator = mix.groupby("company")["clip"].sum()
    grouped = mix.groupby(["company", "industry_group"], sort=False)["clip"].sum()
    for (company, industry), volume in grouped.items():
        if industry is None or (isinstance(industry, float) and pd.isna(industry)):
            continue
        base = float(denominator.loc[company])
        if base <= 0:
            continue
        mixes.setdefault(company, {})[str(industry)] = float(volume) / base
    for company, shares in mixes.items():
        name = max(shares, key=shares.get)
        tops[company] = {"industry_group": name, "share": shares[name]}
    fts_volume = mix.loc[mix["fts_flag"], :].groupby("company")["clip"].sum()
    for company, volume in fts_volume.items():
        base = float(denominator.loc[company])
        if base > 0:
            fts[company] = float(volume) / base
    present = set(denominator.index)
    for company in present:
        fts.setdefault(company, 0.0)
    return mixes, tops, fts


def _top_lists(frame: pd.DataFrame) -> tuple[dict[str, list], dict[str, list]]:
    public: dict[str, list] = {}
    trace: dict[str, list] = {}
    top = frame[frame["rank"] <= 5].sort_values(["company", "rank"])
    for row in top.itertuples(index=False):
        rank = int(row.rank)
        known = bool(row.in_profile)
        view = {
            "slot": f"counterparty_{rank}",
            "rank": rank,
            "volume_share": None if pd.isna(row.share) else float(row.share),
            "industry_group": None if not known or pd.isna(row.industry_group) else row.industry_group,
            "region": None if not known or pd.isna(row.region) else row.region,
            "fts_flag": bool(row.fts_flag) if known else None,
            "countries": row.countries.split("\x1f") if known and isinstance(row.countries, str) and row.countries else [],
        }
        public.setdefault(row.company, []).append(view)
        trace.setdefault(row.company, []).append(
            {"slot": view["slot"], "counterparty_id": row.counterparty, "volume_share": view["volume_share"]}
        )
    return public, trace


def _direction_pack(frame: pd.DataFrame, exclude_hubs: bool) -> dict[str, dict]:
    if frame.empty:
        return {}
    mixes, tops, fts = _industry_mix(frame, exclude_hubs)
    public, trace = _top_lists(frame)
    grouped = frame.groupby("company", sort=False)
    clip = grouped["clip"].sum()
    raw = grouped["raw"].sum()
    edges = grouped.size()
    counterparties = grouped["counterparty"].nunique()
    known_clip = frame.loc[frame["in_profile"]].groupby("company")["clip"].sum().to_dict()
    known_raw = frame.loc[frame["in_profile"]].groupby("company")["raw"].sum().to_dict()
    hub_clip = frame.loc[frame["is_hub"]].groupby("company")["clip"].sum().to_dict()
    top1 = grouped["share"].max().to_dict()
    top5 = frame.loc[frame["rank"] <= 5].groupby("company")["share"].sum().to_dict()
    hhi = frame.assign(share2=frame["share"] ** 2).groupby("company", sort=False)["share2"].sum().to_dict()
    pack = {}
    for company, clip_total in clip.items():
        clip_total = float(clip_total)
        raw_total = float(raw.at[company])
        known_value = float(known_clip.get(company, 0.0))
        known_raw_value = float(known_raw.get(company, 0.0))
        pack[company] = {
            "n_edges": int(edges.at[company]),
            "n_counterparties": int(counterparties.at[company]),
            "volume_clip": clip_total,
            "volume_raw": raw_total,
            "known_clip": known_value,
            "known_raw": known_raw_value,
            "known_share": None if clip_total <= 0 else known_value / clip_total,
            "known_share_raw": None if raw_total <= 0 else known_raw_value / raw_total,
            "hub_clip": float(hub_clip.get(company, 0.0)),
            "top": public.get(company, []),
            "trace": trace.get(company, []),
            "top1_share": None if clip_total <= 0 else float(top1[company]),
            "top5_share": None if clip_total <= 0 else float(top5.get(company, 0.0)),
            "hhi": None if clip_total <= 0 else float(hhi[company]),
            "industry_mix": mixes.get(company, {}),
            "top_industry": tops.get(company),
            "fts_share": fts.get(company),
        }
    return pack


def _context_shares(incoming: pd.DataFrame, outgoing: pd.DataFrame, exclude_hubs: bool) -> dict[str, dict]:
    frames = [frame for frame in (incoming, outgoing) if not frame.empty]
    if not frames:
        return {}
    both = pd.concat(frames, ignore_index=True)
    both = both.groupby(["company", "counterparty"], as_index=False).agg(
        clip=("clip", "sum"),
        in_profile=("in_profile", "max"),
        is_hub=("is_hub", "max"),
        region=("region", "first"),
        countries=("countries", "first"),
        group_id=("group_id", "first"),
        company_region=("company_region", "first"),
        company_group=("company_group", "first"),
    )
    known = both[both["in_profile"]]
    known_sum = known.groupby("company")["clip"].sum()
    result: dict[str, dict] = {company: {} for company in both["company"].unique()}
    if known.empty:
        return result
    same_rows = known[known["company_region"].notna() & (known["region"] == known["company_region"])]
    same_sum = same_rows.groupby("company")["clip"].sum()
    regional = known[known["region"].notna()]
    top_region = regional.groupby(["company", "region"])["clip"].sum().groupby("company").max()
    same_group_rows = known[known["company_group"].notna() & (known["group_id"] == known["company_group"])]
    same_group = same_group_rows.groupby("company")["clip"].sum()
    country_rows = known[~known["is_hub"]] if exclude_hubs else known
    countries: dict[str, dict[str, float]] = {}
    for row in country_rows.itertuples(index=False):
        if not isinstance(row.countries, str) or not row.countries:
            continue
        bucket = countries.setdefault(row.company, {})
        for country in row.countries.split("\x1f"):
            bucket[country] = bucket.get(country, 0.0) + float(row.clip)
    meta = known.drop_duplicates("company").set_index("company")
    for company, base in known_sum.items():
        base = float(base)
        item = result.setdefault(company, {})
        if base <= 0:
            continue
        company_region = meta.loc[company, "company_region"]
        company_group = meta.loc[company, "company_group"]
        if not pd.isna(company_region):
            item["same_region_share"] = float(same_sum.loc[company]) / base if company in same_sum.index else 0.0
        if company in top_region.index:
            item["top_region_share"] = float(top_region.loc[company]) / base
        if not pd.isna(company_group):
            item["same_group_volume_share"] = float(same_group.loc[company]) / base if company in same_group.index else 0.0
        ranked = countries.get(company, {})
        item["countries"] = [name for name, _ in sorted(ranked.items(), key=lambda pair: (-pair[1], pair[0]))[:3]]
    return result


def _empty_direction() -> dict:
    return {
        "n_edges": 0,
        "n_counterparties": 0,
        "volume_clip": 0.0,
        "volume_raw": 0.0,
        "known_clip": 0.0,
        "known_raw": 0.0,
        "known_share": None,
        "hub_clip": 0.0,
        "top": [],
        "trace": [],
        "top1_share": None,
        "top5_share": None,
        "hhi": None,
        "industry_mix": {},
        "top_industry": None,
        "fts_share": None,
    }


_VOLUME_FIELDS = (
    ("n_edges", "n_edges"),
    ("n_counterparties", "n_counterparties"),
    ("volume_clip", "volume_clip"),
    ("volume_raw", "volume_raw"),
    ("known_share", "known_counterparty_volume_share"),
)
_MIX_FIELDS = (
    ("top", "top_counterparties"),
    ("trace", "trace_top"),
    ("top1_share", "top1_share"),
    ("top5_share", "top5_share"),
    ("hhi", "hhi"),
    ("industry_mix", "industry_mix"),
    ("top_industry", "top_industry"),
    ("fts_share", "fts_counterparty_volume_share"),
)


def build_graph_records(
    edges: pd.DataFrame,
    profiles: list[dict],
    config: dict,
    version: str,
    amt_p99: float,
    profile_ids: set[str],
) -> tuple[list[dict], dict]:
    exclude_hubs = bool(config["payments"]["hubs"]["exclude_hubs"])
    log.info("степени и хабы")
    cutoff, hubs = degree_and_hubs(edges, float(config["payments"]["hubs"]["percentile"]))
    log.info("хабов %s, порог степени %.1f", len(hubs), cutoff)
    attributes = profile_attributes(profiles)
    work = edges.copy()
    work["clip"] = work["PAY_AMT"].clip(upper=amt_p99)
    work["raw"] = work["PAY_AMT"]

    def side(company_col: str, counterparty_col: str) -> pd.DataFrame:
        frame = work.rename(columns={company_col: "company", counterparty_col: "counterparty"})[
            ["company", "counterparty", "clip", "raw"]
        ]
        return _prepare(frame, attributes, hubs)

    log.info("присоединяю атрибуты контрагентов")
    incoming = side("BENEFICIARY", "PAYER")
    outgoing = side("PAYER", "BENEFICIARY")
    log.info("считаю направления")
    packed = {
        "in": _direction_pack(incoming, exclude_hubs),
        "out": _direction_pack(outgoing, exclude_hubs),
    }
    log.info("считаю общий контекст")
    context = _context_shares(incoming, outgoing, exclude_hubs)
    companies = (set(packed["in"]) | set(packed["out"])) & profile_ids

    records = []
    for company in companies:
        sides = {name: packed[name].get(company) or _empty_direction() for name in ("in", "out")}
        total_clip = sides["in"]["volume_clip"] + sides["out"]["volume_clip"]
        total_raw = sides["in"]["volume_raw"] + sides["out"]["volume_raw"]
        known_clip = sides["in"]["known_clip"] + sides["out"]["known_clip"]
        known_raw = sides["in"]["known_raw"] + sides["out"]["known_raw"]
        known_total = None if total_clip <= 0 else known_clip / total_clip
        extra = context.get(company, {})
        record = {
            "client_id": company,
            "profile_version": version,
            "has_payment_data": True,
            "missing_data": [],
        }
        for source, target in _VOLUME_FIELDS:
            for name in ("in", "out"):
                record[f"{target}_{name}"] = sides[name][source]
        record["known_counterparty_volume_share_total"] = known_total
        record["known_counterparty_volume_share_raw_total"] = None if total_raw <= 0 else known_raw / total_raw
        for source, target in _MIX_FIELDS:
            for name in ("in", "out"):
                record[f"{target}_{name}"] = sides[name][source]
        record["counterparty_countries_top3"] = extra.get("countries", [])
        record["same_region_share"] = extra.get("same_region_share")
        record["top_region_share"] = extra.get("top_region_share")
        record["same_group_volume_share"] = extra.get("same_group_volume_share")
        record["net_flow_ratio"] = None if total_clip <= 0 else (sides["in"]["volume_clip"] - sides["out"]["volume_clip"]) / total_clip
        record["hub_volume_share"] = None if total_clip <= 0 else (sides["in"]["hub_clip"] + sides["out"]["hub_clip"]) / total_clip
        rules = config["reliability"]
        record["reliability_in"] = payment_reliability(sides["in"]["known_share"], sides["in"]["n_edges"], rules)
        record["reliability_out"] = payment_reliability(sides["out"]["known_share"], sides["out"]["n_edges"], rules)
        record["reliability_total"] = payment_reliability(
            known_total, sides["in"]["n_edges"] + sides["out"]["n_edges"], rules
        )
        records.append(record)
    stats = {
        "hub_cutoff_degree": cutoff,
        "hubs": len(hubs),
        "pay_amt_p99": amt_p99,
    }
    return records, stats
