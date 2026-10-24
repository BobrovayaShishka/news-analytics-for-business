# Масштабирование и сопровождение

Документ закрывает операционный контур: как расширять решение без правок ядра и как его сопровождать.

## Масштабирование без правок Python

| Что добавляем | Куда писать | Что не трогать |
| --- | --- | --- |
| Отрасль / ОКВЭД | `alfa/config/okved_to_theme.yaml`, `industry_labels` в `scoring.yaml` | `interpret.py`, `tools.py` |
| Тип клиента / продукт банка | правило в `sensitivity.yaml`, признак в профиле, `presentation.feature_ref` в `mechanisms.yaml` | хардкод заголовков |
| Источник новостей | запись в `pipelines/news/sources.yaml`; при новой БД — модуль с `NewsSource` и `data.impl` в `alfa/config/config.yaml` | контракт `alfa/data/interfaces.py` |
| Аналитика банка | JSONL/фикстуры через `AnalyticsStore`; `data.impl` указывает фабрику | поиск в `alfa/kb` |
| Механизм влияния | блок в `mechanisms.yaml` (tools, templates, **presentation**), правило в `sensitivity.yaml`, строка в `scoring.reasons` | `interpret.py` / `tools.py` |
| Вопрос клиенту | `questions` в `mechanisms.yaml` + поле в `structurally_missing` | — |
| Промпт LLM | новый файл `alfa/agent/prompts/*.vN.md` и версия в `agent.yaml` | старые версии оставлять |

Карточки аналитики не зашиты в код (`_CARD_TEXT` убран): текст и метаданные приходят из `AnalyticsStore` / KB. Новый документ = новые фикстуры или реализация хранилища, пересборка индекса.

### Чеклист нового механизма

1. Правило и `requires` в `alfa/config/sensitivity.yaml`.
2. Запись в `mechanisms` с `risk_template` / `opportunity_template` / `watch_template` и блоком `presentation` (`headline`, `factor`, `effect`, `feature_ref`, при необходимости `headlines`, `factor_parts`, `feature_ref_prefer`, `tags`).
3. Причина в `scoring.yaml` → `reasons`.
4. При необходимости уточнения в `consequences.yaml` (`by_field` / `event_matrix` / `scenario_matrix`).
5. `python -m pytest` и `python -m alfa.eval.run_eval --task extraction --split dev --mock`.

После правок конфигов перезапуск процесса сбрасывает `lru_cache` загрузчиков (`load_agent_config` и др.).

## Техническое сопровождение

### Мониторинг качества

- `monitor_rate` в `agent.yaml` — доля сессий на выборочный мониторинг (critic / experiment).
- `python -m alfa.eval.run_eval` — регрессии extraction и search.
- `python -m alfa.ranking.bench --smoke` — ранжирование.
- Трассы шагов агента (`alfa.core.trace`) и журнал LLM в experiment-слое.

### Источники новостей

- Здоровье лент: `NewsSource.source_health()` (состояние, пустые ответы, интервал опроса).
- Docker: `docker-compose.yml` сервис `news-ingest` с `--loop`; логи и `ALFA_LOG_LEVEL`.
- Реестр и политика сбора: `pipelines/news/sources.yaml`, `DEPENDENCIES.md`.

### Обновление карточек аналитики

- Порог устаревания: `stale_after_days` в `agent.yaml` (и зеркало в scoring). Устаревшие карточки помечаются `stale`; planner может добрать контекст новостями.
- Поставка контента: обновить файлы фикстур / выгрузку в `AnalyticsStore`, пересобрать индекс KB.
- Банк-view (`forecast` / `opinion` / `risk_assessment`) отбирается по `claim_type`, не по хардкоду текста.

### Версионирование и выкладка

- Версии: `version` в YAML (`mechanisms-v2`, `agent-v2`, …) и `prompt_versions`.
- Конфиг без секретов; ключи только из env (`ALFA_LLM_API_KEY`, `ALFA_ORGANIZER_KEY`).
- Образ новостного контура — `Dockerfile` + `requirements-news.txt`; полный контур — `requirements-dev.txt`.

### Роли сопровождения

| Роль | Зона |
| --- | --- |
| Аналитик / product | YAML механизмов, последствия, отрасли, вопросы |
| Data / content | источники RSS, карточки, фикстуры аналитики |
| DevOps | Docker ingest, логи, пороги stale/monitor |
| ML / QA | eval, prompt versions, smoke ranking |
