"""Выбор NewsSource и AnalyticsStore. data.impl: stub либо модуль с фабриками."""

from __future__ import annotations

import importlib
from typing import Any

from alfa.core.config import load_runtime_config
from alfa.data.interfaces import AnalyticsStore, NewsSource
from alfa.data.stubs.analytics_stub import JsonlAnalyticsStore
from alfa.data.stubs.news_stub import JsonlNewsSource


def build_sources(config: dict[str, Any] | None = None) -> tuple[NewsSource, AnalyticsStore]:
    cfg = config or load_runtime_config()
    impl = str(cfg["data"]["impl"])
    if impl == "stub":
        paths = cfg["paths"]
        return (
            JsonlNewsSource(paths["news_fixtures"]),
            JsonlAnalyticsStore(paths["analytics_fixtures"]),
        )
    module = importlib.import_module(impl)
    return module.news_source(cfg), module.analytics_store(cfg)
