"""Команды: ingest, report. Секретов в аргументах нет."""

from __future__ import annotations

import argparse
import time

from alfa.core.config import load_runtime_config
from alfa.core.logging import get_logger
from pipelines.news.report import format_store_report
from pipelines.news.source import LiveNewsSource


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pipelines.news")
    parser.add_argument("command", nargs="?", default="ingest", choices=("ingest", "report"))
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--backfill-days", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--offline", action="store_true", help="не ходить в сеть, показать последнюю сводку")
    parser.add_argument("--db", default=None)
    parser.add_argument("--sources-file", default=None)
    args = parser.parse_args(argv)
    get_logger("news")
    config = load_runtime_config()
    if args.db:
        config = {**config, "news": {**(config.get("news") or {}), "db_path": args.db}}
    if args.sources_file:
        config = {**config, "news": {**(config.get("news") or {}), "sources_path": args.sources_file}}
    source = LiveNewsSource.from_config(config)
    if args.command == "report" or args.offline:
        print(format_store_report(source.pipeline.store, sample_size=int(source.pipeline.settings["report_sample_size"])))
        return
    if not args.loop:
        report = source.run_ingest(args.sources, args.backfill_days)
        print(format_store_report(source.pipeline.store, sample_size=int(source.pipeline.settings["report_sample_size"])))
        print(f"новых документов: {report.new_docs}")
        return
    while True:
        due = source.pipeline.due_source_names()
        if due:
            source.run_ingest(due, args.backfill_days)
            print(format_store_report(source.pipeline.store, sample_size=int(source.pipeline.settings["report_sample_size"])))
        time.sleep(max(1.0, min(30.0, source.pipeline.seconds_until_due())))


if __name__ == "__main__":
    main()
