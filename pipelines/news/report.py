"""Сводка запуска для make ingest-report."""

from __future__ import annotations

from alfa.data.interfaces import RunReport
from pipelines.news.pipeline import sample_clusters
from pipelines.news.store import NewsStore


def format_report(report: RunReport, *, duplicate_identities: int | None = None) -> str:
    fresh = "нет дат" if report.freshness_hours is None else f"{report.freshness_hours:.1f}"
    lines = [
        f"Запуск {report.run_id}",
        f"Покрытие источников: {report.coverage:.0%}",
        f"Найдено {report.found}, скачано {report.downloaded}, разобрано {report.parsed}",
        f"Пустые или короткие: {report.empty_or_short}, с флагами: {report.flagged}, кластеров с дублями: {report.clustered}",
        f"Новых документов: {report.new_docs}, доля дублей: {report.duplicate_ratio:.1%}",
        f"Свежесть, медианный возраст, ч: {fresh}",
        f"Эмбеддер дедупа: {report.embedder}",
        "Проблемы: " + (", ".join(report.problems) if report.problems else "нет"),
    ]
    if duplicate_identities is not None:
        lines.append(f"Дубли по каноническому URL внутри источника: {duplicate_identities}")
    lines.append("Источники:")
    if not report.sources:
        lines.append("- нет")
    for source in report.sources:
        errors = "; ".join(source.get("errors") or []) or "нет"
        lines.append(
            f"- {source.get('name')}: найдено {source.get('found', 0)}, "
            f"скачано {source.get('downloaded', 0)}, разобрано {source.get('parsed', 0)}, "
            f"короткие {source.get('empty_or_short', 0)}, ошибки: {errors}"
        )
    lines.append("Случайные кластеры:")
    if not report.sample_clusters:
        lines.append("- нет")
    for cluster in report.sample_clusters:
        members = "; ".join(
            f"{member['source']}: {member['title']}" + (" *" if member.get("is_primary") else "")
            for member in cluster["members"]
        )
        lines.append(f"- {cluster['cluster_id']} ({cluster['size']}) {members}")
    return "\n".join(lines)


def format_store_report(store: NewsStore, *, sample_size: int = 8) -> str:
    latest = store.latest_run()
    if latest is None:
        report = RunReport(run_id="-", started_at="", finished_at="", embedder="")
        text = "Запусков ещё не было\n" + format_report(report, duplicate_identities=store.duplicate_identity_count())
        return text
    report = RunReport(
        run_id=str(latest.get("run_id") or ""),
        started_at=str(latest.get("started_at") or ""),
        finished_at=str(latest.get("finished_at") or ""),
        found=int(latest.get("found") or 0),
        downloaded=int(latest.get("downloaded") or 0),
        parsed=int(latest.get("parsed") or 0),
        empty_or_short=int(latest.get("empty_or_short") or 0),
        flagged=int(latest.get("flagged") or 0),
        clustered=int(latest.get("clustered") or 0),
        new_docs=int(latest.get("new_docs") or 0),
        problems=list(latest.get("problems") or []),
        sources=list(latest.get("sources") or []),
        duplicate_ratio=float(latest.get("duplicate_ratio") or 0),
        freshness_hours=latest.get("freshness_hours"),
        coverage=float(latest.get("coverage") or 0),
        sample_clusters=sample_clusters(store, sample_size, str(latest.get("run_id") or "report")),
        embedder=str(latest.get("embedder") or ""),
        elapsed_s=float(latest.get("elapsed_s") or 0),
    )
    return format_report(report, duplicate_identities=store.duplicate_identity_count())
