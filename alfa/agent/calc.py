"""Детерминированные расчёты. Арифметику модель не выполняет."""

from __future__ import annotations

import re
from typing import Any

CALC_VERSION = "calc-v1"
FUNCTIONS = ("rate_sensitivity", "fx_sensitivity", "payroll_cost_change")
MIDPOINT_MARK = "середина выбранного диапазона, банк не проверял"
STEP_MARK = "шаг 1 п.п. задан библиотекой, это не цифра из новости"
# Середины диапазонов ответов. unknown и пусто остаются иллюстрацией на единицу.
RANGE_MIDPOINT = {
    "debt_amount": {"small": 50_000_000.0, "mid": 300_000_000.0, "large": 750_000_000.0},
    "fx_expense_share": {"low": 0.05, "mid": 0.20, "high": 0.40},
    "fx_revenue_share": {"low": 0.05, "mid": 0.20, "high": 0.40},
}
_RATE_MAGNITUDE = re.compile(r"\d+(?:[,.]\d+)?\s*(?:п\.?\s*п\.?|процентн\w*\s+пункт)", re.IGNORECASE)


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _pack(
    fn: str,
    formula: str,
    inputs: dict[str, Any],
    assumptions: list[str],
    result: float | None,
    unit: str,
    illustrative: bool,
    text: str,
) -> dict[str, Any]:
    return {
        "calc_id": f"calc:{fn}",
        "fn": fn,
        "formula": formula,
        "inputs": inputs,
        "assumptions": assumptions,
        "result": result,
        "unit": unit,
        "illustrative": illustrative,
        "text": text,
        "version": CALC_VERSION,
    }


def rate_sensitivity(debt: Any, delta_pp: Any) -> dict[str, Any]:
    delta = _float(delta_pp)
    if delta is None:
        delta = 1.0
    amount = _float(debt)
    if amount is None:
        return _pack(
            "rate_sensitivity",
            "interest = debt * delta_pp / 100",
            {"debt": None, "delta_pp": delta, "illustrative_debt_rub": 100_000_000},
            ["сумма кредита неизвестна", "шаг ставки задан как иллюстрация", "это не оценка по компании"],
            1_000_000.0,
            "RUB/year",
            True,
            "иллюстрация не про компанию: на каждые 100 млн рублей кредита изменение ставки на 1 процентный пункт даёт 1 млн рублей в год",
        )
    result = amount * delta / 100.0
    shown = int(result) if result.is_integer() else round(result, 2)
    return _pack(
        "rate_sensitivity",
        "interest = debt * delta_pp / 100",
        {"debt": amount, "delta_pp": delta},
        ["сумма взята из ответа пользователя и банком не проверена"],
        float(shown),
        "RUB/year",
        False,
        f"изменение ставки на {int(delta) if float(delta).is_integer() else delta} процентный пункт даёт {shown} рублей в год",
    )


def fx_sensitivity(expense_share: Any, revenue: Any, delta_pct: Any) -> dict[str, Any]:
    share = _float(expense_share)
    money = _float(revenue)
    delta = _float(delta_pct)
    if delta is None:
        delta = 1.0
    if share is None or money is None:
        return _pack(
            "fx_sensitivity",
            "effect = revenue * expense_share * delta_pct / 100",
            {"expense_share": share, "revenue": None, "delta_pct": delta},
            ["доля валютных расходов или выручка неизвестны", "это не оценка по компании"],
            1.0,
            "unit",
            True,
            "иллюстрация не про компанию: на 100 единиц выручки при доле 10 процентов сдвиг курса на 1 процент даёт 1 единицу",
        )
    result = money * share * delta / 100.0
    shown = int(result) if result.is_integer() else round(result, 2)
    return _pack(
        "fx_sensitivity",
        "effect = revenue * expense_share * delta_pct / 100",
        {"expense_share": share, "revenue": money, "delta_pct": delta},
        ["доли взяты из ответа пользователя и банком не проверены"],
        float(shown),
        "RUB",
        False,
        f"сдвиг курса на {int(delta) if float(delta).is_integer() else delta} процент даёт {shown} рублей",
    )


def payroll_cost_change(headcount_class: Any, delta: Any) -> dict[str, Any]:
    change = _float(delta)
    if change is None:
        change = 1.0
    klass = _float(headcount_class)
    return _pack(
        "payroll_cost_change",
        "fund_change_pct = delta_pct",
        {"headcount_class": klass, "delta_pct": change},
        ["рублёвая база фонда неизвестна", "это не оценка по компании"],
        change,
        "percent",
        True,
        "иллюстрация не про компанию: на каждый 1 процент изменения оплаты фонд меняется на 1 процент",
    )


def run_function(name: str, *, debt: Any = None, delta_pp: Any = None, expense_share: Any = None, revenue: Any = None, delta_pct: Any = None, headcount_class: Any = None) -> dict[str, Any]:
    if name == "rate_sensitivity":
        return rate_sensitivity(debt, delta_pp)
    if name == "fx_sensitivity":
        return fx_sensitivity(expense_share, revenue, delta_pct)
    if name == "payroll_cost_change":
        return payroll_cost_change(headcount_class, delta_pct)
    raise KeyError(name)


def translate_answer(field: str, value: Any) -> tuple[float | None, bool]:
    """Перевод ответа в число. Второй флаг — взята середина диапазона."""
    if value is None or (isinstance(value, str) and value.strip() in {"", "unknown"}):
        return None, False
    table = RANGE_MIDPOINT.get(field) or {}
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        key = str(value).strip()
        if key in table:
            return float(table[key]), True
    return _float(value), False


def fact_has_rate_magnitude(text: str) -> bool:
    return bool(_RATE_MAGNITUDE.search(text or ""))


def _with_phrase(data: dict[str, Any], phrase: str, *, illustrative: bool | None = None) -> dict[str, Any]:
    output = dict(data)
    assumptions = [item for item in output.get("assumptions") or [] if item != phrase]
    output["assumptions"] = [phrase, *assumptions]
    if illustrative is not None:
        output["illustrative"] = illustrative
    text = str(output.get("text") or "")
    if phrase not in text:
        output["text"] = f"{phrase}. {text}" if text else phrase
    return output


def build_calc(
    fn: str,
    *,
    user: dict[str, Any] | None,
    features: dict[str, Any] | None,
    revenue_allowed: bool,
    fact_text: str = "",
) -> dict[str, Any]:
    """Собрать расчёт из ответа и профиля. Шаг ставки всегда 1 п.п."""
    answers = user or {}
    profile = features or {}
    if fn == "rate_sensitivity":
        debt, midpoint = translate_answer("debt_amount", answers.get("debt_amount"))
        data = rate_sensitivity(debt, 1)
        if midpoint and not data["illustrative"]:
            data = _with_phrase(data, MIDPOINT_MARK, illustrative=False)
        if not fact_has_rate_magnitude(fact_text):
            data = _with_phrase(data, STEP_MARK)
        return data
    if fn == "fx_sensitivity":
        expense = answers.get("fx_expense_share")
        if expense not in (None, "", "unknown"):
            share, midpoint = translate_answer("fx_expense_share", expense)
        else:
            share, midpoint = translate_answer("fx_revenue_share", answers.get("fx_revenue_share"))
        revenue = _float(profile.get("revenue")) if revenue_allowed else None
        data = fx_sensitivity(share, revenue, 1)
        if midpoint and not data["illustrative"]:
            data = _with_phrase(data, MIDPOINT_MARK, illustrative=False)
        return data
    if fn == "payroll_cost_change":
        # Класс численности не переводится в рубли: базы фонда нет.
        return payroll_cost_change(profile.get("workers_class"), 1)
    raise KeyError(fn)
