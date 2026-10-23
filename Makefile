# Если make нет: команды из README.md.
# LLM: локальный Ollama, qwen2.5:7b (alfa/config/config.yaml). Уровень логов: ALFA_LOG_LEVEL.

PY ?= python

.PHONY: test profiles feed extract index eval-extraction eval-search ingest ingest-report scale-bench

profiles:
	$(PY) -m alfa

feed:
	$(PY) -m alfa.ranking.feed --client U_3263

test:
	$(PY) -m pytest

extract:
	$(PY) -m alfa.extraction.event_extractor

index:
	$(PY) -m alfa.kb.index

eval-extraction:
	$(PY) -m alfa.eval.run_eval --task extraction --split dev

eval-search:
	$(PY) -m alfa.eval.run_eval --task search --split dev --noise none

ingest:
	$(PY) -m pipelines.news ingest

ingest-report:
	$(PY) -m pipelines.news report

scale-bench:
	$(PY) -m alfa.ranking.bench
