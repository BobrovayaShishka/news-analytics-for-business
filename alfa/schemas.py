from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Reliability = Literal["none", "low", "medium", "high"]


class Products(BaseModel):
    has_credit: bool | None = None
    has_leasing: bool | None = None
    has_deposit: bool | None = None
    has_guarantees: bool | None = None
    has_factoring: bool | None = None
    has_payroll: bool | None = None
    has_rko: bool | None = None


class FxProfile(BaseModel):
    exposure: Literal["known", "unknown"]
    confidence: Literal["high", "low", "none"]
    countries: list[str] = Field(default_factory=list)
    raw: str | None = None


class MarketPosition(BaseModel):
    """Ранг по выручке внутри ОКВЭД-2 среди клиентов банка. Доля рынка в выгрузке отсутствует."""

    method: Literal["revenue_rank_within_okved2"] = "revenue_rank_within_okved2"
    okved2: str | None = None
    rank: int | None = None
    peers: int | None = None
    share_percent: Literal[None] = None
    note: str = "Доля рынка в выгрузке отсутствует."


class Attributes(BaseModel):
    client_type: str | None = None
    okved_code: str | None = None
    okved_name: str | None = None
    okved_section: str | None = None
    industry_group: str
    region: str | None = None
    city: str | None = None
    age_class: int | None = None
    workers_class: int | None = None
    revenue: float | None = None
    revenue_unknown: bool
    size_class: str
    group_id: str | None = None
    group_size: int | None = None
    eco_stage: str | None = None
    org_type: str | None = None
    okfs_name: str | None = None


class Profile(BaseModel):
    client_id: str
    profile_version: str
    conflicting_records: bool
    conflicting_fields: list[str]
    source_row_count: int
    attributes: Attributes
    products: Products
    fx: FxProfile
    market_position: MarketPosition = Field(default_factory=MarketPosition)
    missing_fields: list[str]
    structurally_missing: list[str]
    text_description: str


class CounterpartyView(BaseModel):
    slot: str
    rank: int
    volume_share: float | None = None
    industry_group: str | None = None
    region: str | None = None
    fts_flag: bool | None = None
    countries: list[str] = Field(default_factory=list)


class IndustryShare(BaseModel):
    industry_group: str
    share: float


class GraphFeatures(BaseModel):
    client_id: str
    profile_version: str
    has_payment_data: bool
    missing_data: list[str] = Field(default_factory=list)
    n_edges_in: int | None = None
    n_edges_out: int | None = None
    n_counterparties_in: int | None = None
    n_counterparties_out: int | None = None
    volume_clip_in: float | None = None
    volume_clip_out: float | None = None
    volume_raw_in: float | None = None
    volume_raw_out: float | None = None
    known_counterparty_volume_share_in: float | None = None
    known_counterparty_volume_share_out: float | None = None
    known_counterparty_volume_share_total: float | None = None
    known_counterparty_volume_share_raw_total: float | None = None
    top_counterparties_in: list[CounterpartyView] = Field(default_factory=list)
    top_counterparties_out: list[CounterpartyView] = Field(default_factory=list)
    top1_share_in: float | None = None
    top1_share_out: float | None = None
    top5_share_in: float | None = None
    top5_share_out: float | None = None
    hhi_in: float | None = None
    hhi_out: float | None = None
    industry_mix_in: dict[str, float] = Field(default_factory=dict)
    industry_mix_out: dict[str, float] = Field(default_factory=dict)
    top_industry_in: IndustryShare | None = None
    top_industry_out: IndustryShare | None = None
    fts_counterparty_volume_share_in: float | None = None
    fts_counterparty_volume_share_out: float | None = None
    counterparty_countries_top3: list[str] = Field(default_factory=list)
    same_region_share: float | None = None
    top_region_share: float | None = None
    same_group_volume_share: float | None = None
    net_flow_ratio: float | None = None
    hub_volume_share: float | None = None
    reliability_in: Reliability = "none"
    reliability_out: Reliability = "none"
    reliability_total: Reliability = "none"
