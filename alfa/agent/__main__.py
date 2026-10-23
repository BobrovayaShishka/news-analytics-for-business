"""Прогон эксперимента Critic и запись цифр для слайда."""

from __future__ import annotations

import json
from pathlib import Path

from alfa.agent.experiment import run_experiment
from alfa.agent.service import _catalog_ranking


def main() -> None:
    root = Path("reports/agent-experiment")
    root.mkdir(parents=True, exist_ok=True)
    report = run_experiment(_catalog_ranking(), limit=18, work_dir=root)
    path = root / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
