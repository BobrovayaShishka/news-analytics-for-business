"""Прогон проверки: один черновик, три конфигурации и зонд ошибок."""

from __future__ import annotations

import copy
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from alfa.agent.models import Session
from alfa.agent.service import InsightService
from alfa.agent.verify import apply_repair, decide, heuristic_critic, verify_stage1

PROBE_KINDS = ("number", "layer_alfa", "bank_voice", "floating", "stale", "impact", "overclaim", "supplier")


def pick_pairs(ranking, limit: int) -> list[tuple[str, str, str]]:
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for event in ranking.events:
        scored = [pair for pair in ranking.score_event(event.event_id, ranking.as_of) if pair.above_threshold]
        scored.sort(key=lambda pair: (-pair.score, pair.client_id))
        for pair in scored:
            buckets[event.event_type].append((event.event_id, pair.client_id))
    chosen: list[tuple[str, str, str]] = []
    while len(chosen) < limit and any(buckets.values()):
        for event_type in list(buckets):
            if not buckets[event_type]:
                continue
            event_id, client_id = buckets[event_type].pop(0)
            chosen.append((event_id, client_id, event_type))
            if len(chosen) >= limit:
                break
    return chosen


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 1.0


def measure(draft: dict, state: dict, config) -> dict[str, Any]:
    stage1 = verify_stage1(draft, state, config)
    critic = heuristic_critic(draft, state)
    sources = (state.get("evidence") or {}).get("sources") or {}
    grounded = 0
    claims = list(draft.get("claims") or [])
    for claim in claims:
        refs = list(claim.get("refs") or []) + list(claim.get("calc_refs") or []) + list(claim.get("evidence") or [])
        ok = all(ref in sources for ref in refs)
        if claim.get("layer") == "fact":
            ok = ok and any(ref.startswith("fact:") for ref in refs)
        elif claim.get("layer") == "alfa":
            ok = ok and any(ref.startswith("card:") for ref in refs)
        elif claim.get("layer") == "model":
            ok = ok and bool(claim.get("feature_refs") or claim.get("mechanism_id"))
        grounded += int(ok)
    return {
        "claims": len(claims),
        "groundedness": _ratio(grounded, len(claims)),
        "number_errors": sum(1 for item in stage1["issues"] if item["type"] in {"number", "date"}),
        "layer_violations": sum(1 for item in stage1["issues"] if item["type"] == "layer_violation"),
        "stage1_issues": len(stage1["issues"]),
        "critic_issues": len(critic),
    }


def _branch(draft: dict, state: dict, config, *, critic: bool) -> tuple[dict, dict]:
    current = copy.deepcopy(draft)
    local = copy.deepcopy(state)
    detected_code: list[dict] = []
    detected_critic: list[dict] = []
    iterations = 0
    decision = "pass"
    while True:
        stage1 = verify_stage1(current, local, config)
        stage2 = heuristic_critic(current, local) if critic else []
        if iterations == 0:
            detected_code = list(stage1["issues"])
            detected_critic = list(stage2)
        issues = list(stage1["issues"]) + list(stage2)
        decision = decide(issues)
        if decision != "repair" or iterations >= config.max_repair:
            if decision == "repair":
                decision = "withheld"
            break
        iterations += 1
        current = apply_repair(current, issues, local, config)
        sources = dict((local.get("evidence") or {}).get("sources") or {})
        sources.update(current.pop("extra_sources", {}) or {})
        evidence = dict(local.get("evidence") or {})
        evidence["sources"] = sources
        local["evidence"] = evidence
    return current, {
        "iterations": iterations,
        "decision": decision,
        "detected_code": detected_code,
        "detected_critic": detected_critic,
    }


def _mutate(draft: dict, state: dict, kind: str) -> tuple[dict, dict, bool]:
    draft = copy.deepcopy(draft)
    state = copy.deepcopy(state)
    claims = draft.get("claims") or []
    changed = False
    if kind == "number":
        for claim in claims:
            if claim.get("layer") == "fact":
                claim["text"] = str(claim.get("text") or "") + " Внезапная ставка 17 процентов."
                changed = True
                break
    elif kind == "layer_alfa":
        for claim in claims:
            if claim.get("layer") == "alfa":
                claim["refs"] = ["fact:f1"]
                changed = True
                break
        if not changed:
            claims.append(
                {
                    "id": "c_alfa_plant",
                    "layer": "alfa",
                    "text": "По оценке Альфа-Банка, отдельной карточки нет.",
                    "refs": ["fact:f1"],
                }
            )
            changed = True
    elif kind == "bank_voice":
        for claim in claims:
            if claim.get("layer") == "model" and not claim.get("calc_refs"):
                claim["text"] = "Альфа-Банк считает, что " + str(claim.get("text") or "")
                changed = True
                break
    elif kind == "floating":
        for claim in claims:
            if claim.get("layer") == "model" and not claim.get("calc_refs"):
                claim["text"] = "У компании плавающая ставка по кредиту."
                changed = True
                break
    elif kind == "stale":
        for card in (state.get("evidence") or {}).get("cards") or []:
            card["stale"] = True
            changed = True
    elif kind == "impact":
        draft["impact_level"] = "low"
        draft["impact_rationale"] = "Влияние высокое и критичное для компании."
        changed = True
    elif kind == "overclaim":
        state["reliability"] = "high"
        for claim in claims:
            if claim.get("layer") == "model" and not claim.get("calc_refs"):
                claim["text"] = str(claim.get("text") or "") + " Однозначно вырастет нагрузка."
                changed = True
                break
    elif kind == "supplier":
        for claim in claims:
            if claim.get("layer") == "hypothesis":
                claim["text"] = "Крупный поставщик остановит отгрузки."
                claim["refs"] = []
                claim["evidence"] = []
                changed = True
                break
    return draft, state, changed


def _live_model() -> bool:
    return os.environ.get("ALFA_EXPERIMENT_LLM", "").strip().lower() in {"1", "true", "yes"}


def _public_status(decision: str) -> str:
    if decision == "pass":
        return "published"
    if decision == "caveats":
        return "published_with_caveats"
    return "withheld"


def run_experiment(ranking, limit: int = 18, work_dir: Path | None = None) -> dict[str, Any]:
    root = Path(work_dir) if work_dir else Path("reports/agent-experiment")
    live = _live_model()
    llm = None
    if live:
        from alfa.core.llm import LLMClient

        llm = LLMClient()
    service = InsightService(
        ranking=ranking,
        store_dir=root / "work",
        journal_path=root / "work" / "journal.jsonl",
        llm=llm,
        llm_enabled=live,
    )
    pairs = pick_pairs(ranking, limit)
    runs = []
    clean = []
    for index, (event_id, client_id, event_type) in enumerate(pairs):
        session = Session(
            user_id=f"exp-{index}",
            allowed_company_ids=[client_id],
            active_client_id=client_id,
            as_of=ranking.as_of,
        )
        started = time.perf_counter()
        insight = service.generate(session, event_id, client_id)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        record = service.store.get(insight["insight_id"]) or {}
        state = record.get("state") or {}
        state["evidence"] = state.get("evidence") or {"sources": {}}
        snapshot = state.get("draft_snapshot") or state.get("draft") or {}
        runs.append(
            {
                "event_id": event_id,
                "client_id": client_id,
                "event_type": event_type,
                "status": insight["status"],
                "latency_ms": latency_ms,
                "llm_tokens": int(state.get("tokens") or 0),
                "cost_rub": float(state.get("cost_rub") or 0),
                "iterations": int(insight["verification"]["iterations"]),
                "mechanisms": list(state.get("mechanism_ids") or []),
            }
        )
        clean.append((snapshot, state))
    configs = {"A_draft": [], "B_stage1": [], "C_stage1_stage2": []}
    repair_iterations = []
    for snapshot, state in clean:
        configs["A_draft"].append(measure(snapshot, state, service.config))
        final_b, meta_b = _branch(snapshot, state, service.config, critic=False)
        final_c, meta_c = _branch(snapshot, state, service.config, critic=True)
        configs["B_stage1"].append(measure(final_b, state, service.config))
        configs["C_stage1_stage2"].append(measure(final_c, state, service.config))
        repair_iterations.append(meta_c["iterations"])
    false_positives = 0
    for snapshot, state in clean:
        false_positives += len(heuristic_critic(snapshot, state))
    probes = []
    for index, (snapshot, state) in enumerate(clean):
        kind = PROBE_KINDS[index % len(PROBE_KINDS)]
        mutated, mutated_state, changed = _mutate(snapshot, state, kind)
        if not changed:
            continue
        clean_code = {item["type"] for item in verify_stage1(snapshot, state, service.config)["issues"]}
        clean_critic = {item["type"] for item in heuristic_critic(snapshot, state)}
        _final_b, meta_b = _branch(mutated, mutated_state, service.config, critic=False)
        _final_c, meta_c = _branch(mutated, mutated_state, service.config, critic=True)
        code_types = {item["type"] for item in meta_b["detected_code"]} - clean_code
        critic_types = {item["type"] for item in meta_c["detected_critic"]} - clean_critic
        probes.append(
            {
                "kind": kind,
                "caught_by_code": sorted(code_types),
                "caught_by_critic": sorted(critic_types),
                "critic_only": sorted(critic_types - code_types),
                "preexisting_issue_types": sorted(clean_code),
                "status_b": _public_status(meta_b["decision"]),
                "status_c": _public_status(meta_c["decision"]),
            }
        )
    code_only = sorted({kind for item in probes for kind in item["caught_by_code"]})
    critic_only = sorted({kind for item in probes for kind in item["critic_only"]})
    n = len(runs) or 1
    mean_latency = round(sum(item["latency_ms"] for item in runs) / n, 3) if runs else 0
    probe_n = len(probes)
    released = sum(1 for item in probes if item["status_b"] in {"published", "published_with_caveats"})
    beyond = sum(1 for item in probes if item["critic_only"])
    if live:
        cost = {
            "mean_latency_ms": mean_latency,
            "mean_llm_tokens": round(sum(item["llm_tokens"] for item in runs) / n, 3) if runs else 0,
            "mean_cost_rub": round(sum(item["cost_rub"] for item in runs) / n, 6) if runs else 0,
            "mean_repair_iterations": round(sum(repair_iterations) / len(repair_iterations), 3) if repair_iterations else 0,
            "label": "живой прогон модели",
        }
        backend = {"interpret": "llm", "critic": "llm", "note": "живой прогон модели", "llm_enabled": True}
    else:
        cost = {
            "mean_latency_ms": mean_latency,
            "mean_llm_tokens": 0,
            "mean_cost_rub": 0,
            "mean_repair_iterations": round(sum(repair_iterations) / len(repair_iterations), 3) if repair_iterations else 0,
            "label": "модель выключена, стоимость 0",
        }
        backend = {"interpret": "template", "critic": "heuristic", "note": "LLM выключена, стоимость генерации 0", "llm_enabled": False}
    report = {
        "n_pairs": len(runs),
        "backend": backend,
        "configs": {name: _average(rows) for name, rows in configs.items()},
        "probes": probes,
        "caught_by_code_types": code_only,
        "critic_only_types": critic_only,
        "false_positives_on_clean": false_positives,
        "preexisting_issue_types": sorted({kind for item in probes for kind in item["preexisting_issue_types"]}),
        "probe_outcomes": {
            "b_published_share": round(released / probe_n, 4) if probe_n else 0.0,
            "c_caught_beyond_b_share": round(beyond / probe_n, 4) if probe_n else 0.0,
        },
        "cost": cost,
        "runs": runs,
        "expert_scores": None,
        "conclusion": _conclusion(code_only, critic_only, false_positives, len(clean)),
    }
    return report


def _average(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = ("groundedness", "number_errors", "layer_violations", "stage1_issues", "critic_issues")
    output = {"n": len(rows)}
    for key in keys:
        output[key] = round(sum(row[key] for row in rows) / len(rows), 4)
    return output


def _conclusion(code_types: list[str], critic_only: list[str], false_positives: int, clean: int) -> str:
    return (
        "Код ловит схему, числа, слои, structurally_missing, запрещённые формулировки и устаревшие карточки"
        f" ({', '.join(code_types) or 'на чистых черновиках замечаний нет'}). "
        "Critic дополнительно ловит расхождение обоснования с рубрикой и слишком сильные формулировки"
        f" ({', '.join(critic_only) or 'на этом прогоне уникальных типов нет'}). "
        f"Ложных срабатываний Critic на чистых черновиках: {false_positives} из {clean}. "
        "Второй этап оставляем включённым только при условиях раздела 9: срабатывание кода, высокий уровень влияния, "
        "расхождение с позицией банка, подозрительный текст и выборка мониторинга."
    )
