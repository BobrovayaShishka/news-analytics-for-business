"""Печать ленты демо-клиента из clients/index.json.

python -m alfa.ranking.feed --client U_3263
"""

from __future__ import annotations

import argparse
import sys

from alfa.access import session_scope
from alfa.ranking.catalog import AS_OF, catalog_cards, catalog_events
from alfa.ranking.config import load_ranking_config, repo_path
from alfa.ranking.matrix import load_demo_rows
from alfa.ranking.service import RankingService


def render_feed(client_id: str, limit: int | None = None) -> str:
    config = load_ranking_config()
    directory = repo_path(config.scoring["paths"]["clients_dir"])
    rows = load_demo_rows(directory, config)
    known = {row.client_id for row in rows}
    if client_id not in known:
        listed = ", ".join(sorted(known)) or "пусто"
        raise SystemExit(f"клиента {client_id} нет в {directory / 'index.json'}. Доступны: {listed}")
    service = RankingService(rows, catalog_events(), catalog_cards(), config, as_of=AS_OF)
    with session_scope(client_id):
        feed = service.build_feed(client_id, AS_OF, limit=limit, record_impressions=False)
    lines = [f"клиент: {feed.client_id}", f"as_of: {feed.as_of.isoformat()}"]
    if feed.empty_state:
        lines.append(f"пусто: {feed.empty_reason}")
        return "\n".join(lines)
    for number, item in enumerate(feed.items, start=1):
        mechanisms = ", ".join(item.mechanism_ids) or "—"
        lines.append(f"{number}. {item.event_id}  {item.score:.4f}  {item.priority}  {mechanisms}")
        for reason in item.reasons:
            lines.append(f"   {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="alfa.ranking.feed")
    parser.add_argument("--client", default="U_3263", help="company_id из clients/index.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    print(render_feed(args.client, args.limit))


if __name__ == "__main__":
    main()
