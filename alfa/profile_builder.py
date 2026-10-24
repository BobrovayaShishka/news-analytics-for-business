"""Профиль компании из df_Uid_data. Без LLM и без подстановки отсутствующих полей."""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from alfa.settings import industry_group, load_okved_map, norm_label, norm_place, okved_section, size_band, split_countries

log = logging.getLogger("alfa.profile")

_MARKS = {
    "has_credit": "CL_PRODUCT_LOA_MARK",
    "has_leasing": "CL_PRODUCT_LEA_MARK",
    "has_deposit": "CL_PRODUCT_DEP_MARK",
    "has_guarantees": "CL_PRODUCT_GAR_MARK",
    "has_factoring": "FACTORING_MARK",
    "has_payroll": "CL_PRODUCT_ZPP_MARK",
    "has_rko": "CL_PRODUCT_RKO_MARK",
}


def _blank(value) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        return False
    return not str(value).strip() or str(value).strip().lower() in {"nan", "none", "<na>"}


def _cell(row: dict, column: str) -> str | None:
    value = row.get(column)
    if _blank(value):
        return None
    return str(value).strip()


def _as_bool(value) -> bool | None:
    if _blank(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true"}:
        return True
    if text in {"0", "0.0", "false"}:
        return False
    log.warning("неизвестная кодировка булева поля: %s", value)
    return None


def _as_revenue(value, sentinels: set[float], non_positive_unknown: bool) -> tuple[float | None, bool]:
    if _blank(value):
        return None, True
    number = float(str(value).strip().replace(",", "."))
    if non_positive_unknown and number <= 0:
        return None, True
    if any(number == sentinel for sentinel in sentinels):
        return None, True
    return number, False


def _ordinal(value, mapping: dict, field_name: str) -> int | None:
    label = norm_label(None if _blank(value) else str(value))
    if not label:
        return None
    if label not in mapping:
        log.warning("неизвестная метка %s: %s", field_name, label)
        return None
    return int(mapping[label])


def _fx(flag: bool | None, countries: list[str], raw: str | None) -> dict:
    if flag is None:
        return {"exposure": "unknown", "confidence": "none", "countries": countries, "raw": raw}
    nonempty = bool(countries)
    if flag and nonempty:
        confidence = "high"
        exposure = "known"
    elif flag or nonempty:
        confidence = "low"
        exposure = "known"
    else:
        confidence = "none"
        exposure = "unknown"
    return {"exposure": exposure, "confidence": confidence, "countries": countries, "raw": raw}


def _size_class(revenue: float | None, revenue_unknown: bool, edges: list[float], labels: list[str]) -> str:
    if revenue_unknown or revenue is None:
        return "unknown"
    return size_band(revenue, edges, labels)


def profile_sentence(profile: dict) -> str:
    attributes = profile["attributes"]
    products = profile["products"]
    fx = profile["fx"]
    parts = [f"Компания {profile['client_id']}."]
    if profile["conflicting_records"]:
        fields = ", ".join(profile["conflicting_fields"])
        parts.append(f"В выгрузке {profile['source_row_count']} противоречивых строк. Поля без единого значения: {fields}.")
    if attributes["client_type"]:
        parts.append(f"Тип клиента: {attributes['client_type']}.")
    if attributes["industry_group"] not in {None, "unknown"}:
        parts.append(
            f"Отрасль: {attributes['industry_group']}, ОКВЭД {attributes['okved_code'] or 'не указан'}."
        )
    elif attributes["okved_code"]:
        parts.append(f"ОКВЭД {attributes['okved_code']}, отраслевая группа не сопоставлена.")
    else:
        parts.append("Отрасль неизвестна.")
    if attributes["region"] or attributes["city"]:
        parts.append(f"Регион: {attributes['region'] or 'не указан'}, город: {attributes['city'] or 'не указан'}.")
    if attributes["workers_class"] is not None:
        parts.append(f"Класс численности: {attributes['workers_class']}.")
    if attributes["age_class"] is not None:
        parts.append(f"Класс возраста компании: {attributes['age_class']}.")
    if attributes["revenue_unknown"]:
        parts.append("Выручка неизвестна.")
    else:
        parts.append(f"Выручка: {attributes['revenue']:.0f} руб. Класс размера: {attributes['size_class']}.")
    position = profile.get("market_position") or {}
    if position.get("rank") is not None and position.get("peers") is not None:
        parts.append(
            f"Положение в ОКВЭД {position.get('okved2') or 'не указан'}: "
            f"ранг {position['rank']} из {position['peers']} по выручке среди клиентов банка. "
            "Доля рынка неизвестна."
        )
    present = [name.removeprefix("has_") for name, value in products.items() if value is True]
    absent = [name.removeprefix("has_") for name, value in products.items() if value is False]
    if present:
        parts.append("Продукты банка: " + ", ".join(present) + ".")
    if absent:
        parts.append("Продуктов банка нет: " + ", ".join(absent) + ".")
    if fx["exposure"] == "known":
        countries = ", ".join(fx["countries"]) if fx["countries"] else "страна не указана"
        parts.append(f"Внешнеэкономический контур известен, уверенность {fx['confidence']}. Страны ФТС: {countries}.")
    else:
        parts.append("Внешнеэкономический контур неизвестен.")
    parts.append(
        "Неизвестны и не заполняются: тип ставки, сумма долга, срок долга, доля валютной выручки, доля валютных расходов."
    )
    return " ".join(parts)


def _missing(attributes: dict, products: dict) -> list[str]:
    missing = []
    for key in ("client_type", "okved_code", "okved_name", "region", "city", "age_class", "workers_class", "eco_stage"):
        if attributes.get(key) is None:
            missing.append(key)
    for key, value in products.items():
        if value is None:
            missing.append(key)
    return missing


def known_revenues(profiles: list[dict]) -> list[float]:
    """Известная выручка клиентов без конфликтов. Из неё считаются квантили размера."""
    values = []
    for item in profiles:
        if item["conflicting_records"]:
            continue
        attributes = item["attributes"]
        revenue = attributes["revenue"]
        if attributes["revenue_unknown"] or revenue is None:
            continue
        values.append(float(revenue))
    return values


def apply_market_position(profiles: list[dict]) -> None:
    """Ранг по выручке внутри ОКВЭД-2. Доля рынка не выводится — в данных её нет."""
    buckets: dict[str, list[tuple[int, float, str]]] = {}
    for index, profile in enumerate(profiles):
        attributes = profile["attributes"]
        code = okved_section(attributes.get("okved_code"))
        revenue = attributes.get("revenue")
        eligible = (
            not profile["conflicting_records"]
            and not attributes.get("revenue_unknown")
            and revenue is not None
            and code is not None
        )
        if eligible:
            buckets.setdefault(code, []).append((index, float(revenue), str(profile["client_id"])))
        profile["market_position"] = {
            "method": "revenue_rank_within_okved2",
            "okved2": code,
            "rank": None,
            "peers": None,
            "share_percent": None,
            "note": (
                "Выручка или ОКВЭД неизвестны — ранг не считаем. Доля рынка в выгрузке отсутствует."
                if not eligible
                else "Ранг по выручке среди клиентов банка с тем же ОКВЭД-2. Доля рынка в выгрузке отсутствует."
            ),
        }

    for code, members in buckets.items():
        members.sort(key=lambda item: (-item[1], item[2]))
        peers = len(members)
        rank = 0
        previous: float | None = None
        for offset, (index, revenue, _client_id) in enumerate(members):
            if previous is None or revenue != previous:
                rank = offset + 1
                previous = revenue
            profile = profiles[index]
            profile["market_position"] = {
                "method": "revenue_rank_within_okved2",
                "okved2": code,
                "rank": rank,
                "peers": peers,
                "share_percent": None,
                "note": (
                    f"Ранг {rank} из {peers} по выручке внутри ОКВЭД {code} "
                    "среди клиентов банка. Доля рынка в выгрузке отсутствует."
                ),
            }


def apply_profile_size(profiles: list[dict], config: dict, version: str) -> None:
    size_edges = [float(item) for item in config["computed"]["size_edges"]]
    size_labels = list(config["size_class"]["classes"])
    for profile in profiles:
        attributes = profile["attributes"]
        attributes["size_class"] = _size_class(
            attributes["revenue"],
            bool(attributes["revenue_unknown"]),
            size_edges,
            size_labels,
        )
        profile["profile_version"] = version
        profile["text_description"] = profile_sentence(profile)


def build_profiles(companies: pd.DataFrame, config: dict) -> tuple[list[dict], dict]:
    ignore = set(config["ignored_columns"])
    sentinels = {float(item) for item in config["revenue"]["unknown_sentinels"]}
    age_map = {norm_label(key): int(value) for key, value in config["age_scale"]["map"].items()}
    worker_map = {norm_label(key): int(value) for key, value in config["workers_scale"]["map"].items()}
    okved_map = load_okved_map()
    separators = list(config["fx"]["country_separators"])
    allowed_types = set(config["client_types"])
    non_positive_unknown = bool(config["revenue"]["non_positive_unknown"])

    columns = [column for column in companies.columns if column != "SYNTHETIC_U" and column not in ignore]
    work = companies[["SYNTHETIC_U", *columns]].copy()
    for column in columns:
        blank = work[column].str.strip().eq("") | work[column].str.lower().isin(["nan", "none"])
        work.loc[blank, column] = pd.NA
    counts = work.groupby("SYNTHETIC_U", sort=False).size()
    flagged = work.groupby("SYNTHETIC_U", sort=False)[columns].nunique(dropna=True).gt(1)
    conflict_fields = {
        client_id: [column for column in columns if flagged.at[client_id, column]]
        for client_id in flagged.index[flagged.any(axis=1)]
    }
    agreed = work.groupby("SYNTHETIC_U", sort=False)[columns].first().mask(flagged)
    group_sizes: Counter[str] = Counter()
    resolved: list[tuple[dict, list[str], int]] = []
    for client_id, row in agreed.iterrows():
        conflicts = conflict_fields.get(client_id, [])
        data = {column: None if _blank(value) else value for column, value in row.items()}
        data["SYNTHETIC_U"] = client_id
        resolved.append((data, conflicts, int(counts.at[client_id])))
        if not conflicts and data.get("GROUP_ID"):
            group_sizes[str(data["GROUP_ID"]).strip()] += 1

    profiles = []
    for row, conflicts, row_count in resolved:
        revenue, revenue_unknown = _as_revenue(row.get("REVENUE_COMPANY"), sentinels, non_positive_unknown)
        okved = _cell(row, "MAINOKVED_CODE")
        okved_name = _cell(row, "MAINOKVED_NAME")
        age = _ordinal(row.get("COMPANY_AGE_YEARS_GRADATION"), age_map, "COMPANY_AGE_YEARS_GRADATION")
        workers = _ordinal(row.get("AMOUNT_WORKERS_FNS_GRADATION"), worker_map, "AMOUNT_WORKERS_FNS_GRADATION")
        client_type = _cell(row, "CL_TYPE")
        if client_type not in allowed_types:
            client_type = None
        group_id = _cell(row, "GROUP_ID")
        country_raw = _cell(row, "FTS_COUNTRY_NAME")
        products = {feature: _as_bool(row.get(column)) for feature, column in _MARKS.items()}
        attributes = {
            "client_type": client_type,
            "okved_code": okved,
            "okved_name": okved_name,
            "okved_section": okved_section(okved),
            "industry_group": industry_group(okved, okved_map),
            "region": norm_place(_cell(row, "REGION")),
            "city": norm_place(_cell(row, "CITY")),
            "age_class": age,
            "workers_class": workers,
            "revenue": revenue,
            "revenue_unknown": revenue_unknown,
            "size_class": None,
            "group_id": group_id,
            "group_size": None if group_id is None else int(group_sizes[group_id]),
            "eco_stage": _cell(row, "ECO_MARK_LABEL"),
            "org_type": _cell(row, "ORG_TYPE"),
            "okfs_name": _cell(row, "OKFS_NAME"),
        }
        profile = {
            "client_id": row["SYNTHETIC_U"],
            "profile_version": "",
            "conflicting_records": bool(conflicts),
            "conflicting_fields": conflicts,
            "source_row_count": row_count,
            "attributes": attributes,
            "products": products,
            "fx": _fx(_as_bool(row.get("FTS_FLAG")), split_countries(country_raw, separators), country_raw),
            "missing_fields": _missing(attributes, products),
            "structurally_missing": list(config["structurally_missing"]),
        }
        profiles.append(profile)

    return profiles, {"unique_clients": len(profiles)}
