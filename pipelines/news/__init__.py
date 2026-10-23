"""Живой источник новостей. Подключение: data.impl = pipelines.news."""

from pipelines.news.source import analytics_store, news_source

__all__ = ["analytics_store", "news_source"]
