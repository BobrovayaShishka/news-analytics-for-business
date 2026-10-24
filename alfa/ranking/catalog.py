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
        url="",
        source="",
        facts=[],
        echo_sources=[],
    )
    payload.update(kwargs)
    return RankEvent(event_id=event_id, event_type=event_type, **payload)


def catalog_rows() -> list[ClientRow]:
    return [
        _client(
            "C_credit",
            has_credit=True,
            industry_group="manufacturing_machinery",
            themes=["manufacturing"],
            size_class="S4",
            revenue=400_000_000.0,
            key_owner_known=True,
            okved_code="28.11",
            market_rank=3,
            market_peers=120,
            market_okved2="28",
        ),
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
    key_text = (
        "Совет директоров Банка России 20 октября 2026 года изменил ключевую ставку. "
        "Новое значение ключевой ставки составляет 14% годовых. "
        "Решение вступает в силу с 21 октября 2026 года. Маркер qxE_key."
    )
    key_echo_rbc = (
        "Банк России установил ключевую ставку на уровне 14% годовых. "
        "Решение опубликовано 20 октября 2026 года. Маркер qxE_key_rbc."
    )
    key_echo_ved = (
        "ЦБ сохранил паузу: ключевая ставка равна 14%. "
        "Дата решения — 20 октября 2026 года. Маркер qxE_key_ved."
    )
    infl_text = (
        "Росстат оценил недельную инфляцию. "
        "Недельная инфляция составила 0,12%. Маркер qxE_infl."
    )
    return [
        _event(
            "E_key",
            "key_rate",
            trust_tier="official",
            magnitude=0.6,
            n_sources=3,
            echo_count=2,
            cluster_id="C_key",
            title="Ключевая ставка",
            text=key_text,
            url="https://www.cbr.ru/press/keypr/",
            source="cbr",
            echo_sources=["rbc", "vedomosti"],
        ),
        _event(
            "E_key_rbc",
            "key_rate",
            trust_tier="major_media",
            is_primary=False,
            cluster_id="C_key",
            n_sources=1,
            title="ЦБ по ставке",
            text=key_echo_rbc,
            url="https://www.rbc.ru/finances/key-rate",
            source="rbc",
        ),
        _event(
            "E_key_ved",
            "key_rate",
            trust_tier="major_media",
            is_primary=False,
            cluster_id="C_key",
            n_sources=1,
            title="Пауза по ставке",
            text=key_echo_ved,
            url="https://www.vedomosti.ru/finance/key-rate",
            source="vedomosti",
        ),
        _event(
            "E_fx_cn",
            "fx",
            entities=[{"text": "Китай", "type": "country"}],
            title="Курс и Китай",
            text="Официальный курс доллара составил 82,85 рубля. Расчёты с Китаем остаются в юанях. Маркер qxE_fx_cn.",
            url="https://www.cbr.ru/currency_base/",
            source="cbr",
        ),
        _event("E_fx", "fx", title="Курс рубля", text="Рыночный курс рубля без страны. Маркер qxE_fx.", url="https://www.moex.com/", source="moex"),
        _event("E_sanc_cn", "sanctions_trade", entities=[{"text": "Китай", "type": "country"}], title="Внешняя торговля", text="Ограничение расчётов с Китаем. Маркер qxE_sanc_cn."),
        _event("E_gold", "commodities_gold", entities=[{"text": "золотодобыча", "type": "industry"}], title="Золото", text="Цена золота на мировом рынке. Маркер qxE_gold."),
        _event("E_oil", "commodities_gold", entities=[{"text": "нефть", "type": "industry"}], title="Нефть", text="Котировка нефти Brent. Маркер qxE_oil."),
        _event("E_labor", "labor", trust_tier="official", title="Рынок труда", text="Минтруд опубликовал ориентир по зарплатам. Маркер qxE_labor."),
        _event(
            "E_infl",
            "inflation",
            trust_tier="official",
            magnitude=0.2,
            title="Инфляция",
            text=infl_text,
            url="https://rosstat.gov.ru/inflation",
            source="rosstat",
        ),
        _event("E_tax", "taxes", trust_tier="official", title="Налоги", text="Минфин обсуждает налоговый режим. Маркер qxE_tax."),
        _event("E_demand", "consumer_demand", entities=[{"text": "розничная торговля", "type": "industry"}], title="Спрос", text="Оборот розницы. Маркер qxE_demand."),
        _event("E_reg_tr", "sector_regulation", entities=[{"text": "грузоперевозки", "type": "industry"}], title="Регулирование перевозок", text="Тарифный коридор перевозок. Маркер qxE_reg_tr."),
        _event("E_ma", "ma", magnitude=0.2, title="Сделка", text="Крупная сделка смены контроля. Маркер qxE_ma."),
    ]


def heldout_events() -> list[RankEvent]:
    """Контрольная выборка: тексты не из catalog_events и не из примеров taxonomy.yaml."""
    return [
        _event(
            "H_energy",
            "other_econ",
            trust_tier="official",
            magnitude=0.45,
            title="Тариф на электроэнергию",
            text=(
                "Регулятор повысил тариф на электроэнергию для промышленных потребителей на 12% "
                "с 1 ноября 2026 года. Маркер qxH_energy."
            ),
        ),
        _event(
            "H_util_fee",
            "other_econ",
            trust_tier="official",
            magnitude=0.4,
            title="Утилизационный сбор",
            text=(
                "Правительство ввело утилизационный сбор на импортную спецтехнику. "
                "Ставка сбора дифференцирована по мощности. Маркер qxH_util_fee."
            ),
        ),
        _event(
            "H_customs",
            "other_econ",
            trust_tier="major_media",
            magnitude=0.5,
            title="Сбой таможенного оформления",
            text=(
                "Сбой в системе таможенного оформления задержал контейнеры в порту. "
                "Импортные поставки идут с опозданием. Маркер qxH_customs."
            ),
        ),
        _event(
            "H_stores",
            "other_econ",
            trust_tier="major_media",
            magnitude=0.35,
            title="Сеть закрывает магазины",
            text=(
                "Крупная розничная сеть закрыла 200 магазинов из-за падения покупательского трафика "
                "и охлаждения потребления. Маркер qxH_stores."
            ),
        ),
        _event(
            "H_osago",
            "other_econ",
            trust_tier="official",
            magnitude=0.3,
            title="Тарифный коридор ОСАГО",
            text=(
                "Расширен тарифный коридор ОСАГО для коммерческого транспорта. "
                "Страховая нагрузка на перевозчиков вырастет. Маркер qxH_osago."
            ),
        ),
    ]


_CATALOG_TOPICS = (
    "key_rate",
    "inflation",
    "fx",
    "commodities_gold",
    "labor",
    "taxes",
    "consumer_demand",
    "sanctions_trade",
)


def catalog_cards() -> list[CardView]:
    """Свежайшая карточка по теме из фикстур аналитики — тот же текст, что видит агент."""
    from alfa.core.config import load_runtime_config
    from alfa.core.schemas import AnalyticsCard
    from alfa.core.timeutil import attach_utc
    from alfa.data.stubs.analytics_stub import JsonlAnalyticsStore

    store = JsonlAnalyticsStore(load_runtime_config()["paths"]["analytics_fixtures"])
    best: dict[str, AnalyticsCard] = {}
    for card in store.get_cards():
        topics = [str(topic.value) for topic in card.topic]
        for topic in topics:
            if topic not in _CATALOG_TOPICS:
                continue
            previous = best.get(topic)
            if previous is None or attach_utc(card.published_at) > attach_utc(previous.published_at):
                best[topic] = card
    output: list[CardView] = []
    for topic in _CATALOG_TOPICS:
        card = best.get(topic)
        if card is None:
            continue
        output.append(
            CardView(
                card.card_id,
                [topic],
                attach_utc(card.published_at),
                verified=str(card.verified.value),
            )
        )
    return output
