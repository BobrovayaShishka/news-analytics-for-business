"""Фикстурный контур: клиенты, события и карточки для офлайн-прогона без LLM."""

from __future__ import annotations

from datetime import datetime, timezone

from alfa.ranking.models import CardView, ClientRow, RankEvent

AS_OF = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
PUBLISHED = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)


def _client(client_id: str, **kwargs) -> ClientRow:
    payload = dict(
        client_type="CLIENT",
        conflicting_records=False,
        profile_version="v1",
        has_credit=False,
        has_leasing=False,
        has_deposit=False,
        has_guarantees=False,
        has_factoring=False,
        has_payroll=False,
        has_rko=False,
        workers_class=1,
        size_class="S3",
        has_payment_data=False,
        reliability="none",
        region="москва",
        themes=[],
        revenue=50_000_000.0,
    )
    payload.update(kwargs)
    return ClientRow(client_id=client_id, **payload)


def _event(event_id: str, event_type: str, **kwargs) -> RankEvent:
    payload = dict(
        published_at=PUBLISHED,
        extraction_status="ok",
        is_business_relevant=True,
        is_primary=True,
        trust_tier="major_media",
        cluster_id=event_id,
        revision=1,
        fact_signature=event_id,
        magnitude=0.25,
        n_sources=1,
        title=event_id,
        text=f"Публикация {event_id} типа {event_type}. Уникальный маркер qx{event_id}.",
        entities=[],
    )
    payload.update(kwargs)
    return RankEvent(event_id=event_id, event_type=event_type, **payload)


def catalog_rows() -> list[ClientRow]:
    return [
        _client("C_credit", has_credit=True, industry_group="manufacturing_machinery", themes=["manufacturing"], size_class="S4", revenue=400_000_000.0, key_owner_known=True),
        _client("C_plain", has_rko=True, industry_group="retail", themes=["retail_demand"], size_class="S2", revenue=5_000_000.0, key_owner_known=False),
        _client("C_fts_cn", fx_exposure="known", fx_confidence="high", fts_countries=["CN"], industry_group="wholesale", themes=["trade"], size_class="S3"),
        _client("C_fts_az", has_leasing=True, has_payroll=True, workers_class=2, fx_exposure="known", fx_confidence="high", fts_countries=["AZ"], industry_group="land_transport", themes=["transport"], size_class="S4", revenue=200_000_000.0),
        _client("C_no_fts", has_rko=True, industry_group="office_admin", themes=[], size_class="S2", workers_class=0),
        _client("C_group_bare", group_id="G1", group_size_reported=2, size_class="S2", revenue=2_000_000.0, industry_group=None, themes=[]),
        _client("C_group_rich", has_credit=True, group_id="G1", group_size_reported=2, industry_group="manufacturing_machinery", themes=["manufacturing"], size_class="S4", revenue=500_000_000.0),
        _client(
            "C_graph_high",
            industry_group="manufacturing_machinery",
            themes=["manufacturing"],
            has_payment_data=True,
            reliability="high",
            known_counterparty_volume_share=0.8,
            fts_counterparty_volume_share=0.1,
            n_edges=20,
            industry_mix_out={"gold_mining": 0.65, "manufacturing_machinery": 0.35},
            intra_group_payment_share=0.05,
        ),
        _client(
            "C_graph_low",
            industry_group="manufacturing_machinery",
            themes=["manufacturing"],
            has_payment_data=True,
            reliability="low",
            known_counterparty_volume_share=0.2,
            n_edges=3,
            industry_mix_out={"gold_mining": 0.65},
        ),
        _client("C_graph_none", industry_group="manufacturing_machinery", themes=["manufacturing"], has_payment_data=False, reliability="none"),
        _client("C_deposit", has_deposit=True, industry_group="real_estate", themes=["real_estate"], size_class="S3", key_owner_known=True),
        _client("C_gold_mine", has_credit=True, okved_code="07.29.1", industry_group="gold_mining", themes=["gold_mining"], size_class="S4", revenue=300_000_000.0),
        _client("C_metal_trade", has_rko=True, okved_code="46.72.1", industry_group="wholesale_metals", themes=["metal_trade"], size_class="S3"),
        _client("C_payroll", has_payroll=True, workers_class=3, industry_group="manufacturing_machinery", themes=["manufacturing"], size_class="S3"),
        _client("C_leasing", has_leasing=True, industry_group="construction_buildings", themes=["construction"], size_class="S3"),
        ClientRow(client_id="C_lowcov", client_type="CLIENT", profile_version="v1", conflicting_records=False),
        _client("C_guar", has_guarantees=True, fx_exposure="known", fx_confidence="high", fts_countries=["CN"], industry_group="wholesale", themes=["trade"]),
        _client("C_factor", has_credit=True, has_factoring=True, industry_group="retail", themes=["retail_demand"], size_class="S3"),
        _client("C_oil", industry_group="mining_oil_gas", themes=["oil_gas"], okved_code="06.10", size_class="S5", revenue=2_000_000_000.0),
    ]


def catalog_events() -> list[RankEvent]:
    return [
        _event("E_key", "key_rate", trust_tier="official", magnitude=0.6, n_sources=3, title="Ключевая ставка", text="Совет директоров регулятора изменил ключевую ставку. Маркер qxE_key."),
        _event("E_fx_cn", "fx", entities=[{"text": "Китай", "type": "country"}], title="Курс и Китай", text="Официальный курс и расчёты с Китаем. Маркер qxE_fx_cn."),
        _event("E_fx", "fx", title="Курс рубля", text="Рыночный курс рубля без страны. Маркер qxE_fx."),
        _event("E_sanc_cn", "sanctions_trade", entities=[{"text": "Китай", "type": "country"}], title="Внешняя торговля", text="Ограничение расчётов с Китаем. Маркер qxE_sanc_cn."),
        _event("E_gold", "commodities_gold", entities=[{"text": "золотодобыча", "type": "industry"}], title="Золото", text="Цена золота на мировом рынке. Маркер qxE_gold."),
        _event("E_oil", "commodities_gold", entities=[{"text": "нефть", "type": "industry"}], title="Нефть", text="Котировка нефти Brent. Маркер qxE_oil."),
        _event("E_labor", "labor", trust_tier="official", title="Рынок труда", text="Минтруд опубликовал ориентир по зарплатам. Маркер qxE_labor."),
        _event("E_infl", "inflation", trust_tier="official", magnitude=0.2, title="Инфляция", text="Недельная инфляция. Маркер qxE_infl."),
        _event("E_tax", "taxes", trust_tier="official", title="Налоги", text="Минфин обсуждает налоговый режим. Маркер qxE_tax."),
        _event("E_demand", "consumer_demand", entities=[{"text": "розничная торговля", "type": "industry"}], title="Спрос", text="Оборот розницы. Маркер qxE_demand."),
        _event("E_reg_tr", "sector_regulation", entities=[{"text": "грузоперевозки", "type": "industry"}], title="Регулирование перевозок", text="Тарифный коридор перевозок. Маркер qxE_reg_tr."),
        _event("E_ma", "ma", magnitude=0.2, title="Сделка", text="Крупная сделка смены контроля. Маркер qxE_ma."),
    ]


def catalog_cards() -> list[CardView]:
    return [
        CardView("card-key", ["key_rate"], datetime(2026, 10, 18, tzinfo=timezone.utc)),
        CardView("card-infl", ["inflation"], datetime(2026, 10, 1, tzinfo=timezone.utc)),
        CardView("card-fx", ["fx"], datetime(2026, 10, 19, tzinfo=timezone.utc)),
        CardView("card-gold", ["commodities_gold"], datetime(2026, 10, 10, tzinfo=timezone.utc)),
        CardView("card-labor", ["labor"], datetime(2026, 10, 15, tzinfo=timezone.utc)),
        CardView("card-tax", ["taxes"], datetime(2026, 10, 12, tzinfo=timezone.utc)),
        CardView("card-demand", ["consumer_demand"], datetime(2026, 10, 11, tzinfo=timezone.utc)),
        CardView("card-sanc", ["sanctions_trade"], datetime(2026, 10, 16, tzinfo=timezone.utc)),
    ]
