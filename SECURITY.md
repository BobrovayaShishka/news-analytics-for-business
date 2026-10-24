# Безопасная работа AI-компонентов

Документ описывает механизмы безопасной работы агентного слоя прототипа:
разграничение доступа, защиту чувствительных данных, журналирование обращений
к инструментам и ограничения на потенциально небезопасные действия.
Реализация уже есть в коде; здесь — их место в архитектуре и допущения прототипа.

## Схема слоёв

```text
Клиент (HTTP / CLI)
        │
        ▼
┌───────────────────┐
│  API / Service    │  Session: user_id + allowed_company_ids
│  alfa/agent/      │  _check() на каждом методе → 403 access_denied
│  service.py       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Graph            │  init_context: client_id ∈ allowed_company_ids
│  alfa/agent/      │  BudgetGuard: токены / число инсайтов на сессию
│  graph.py         │  флаги: access_denied, suspicious_instructions, budget_*
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Planner          │  только TOOL_NAMES; схема аргументов _ALLOWED
│  alfa/agent/      │  client_id в плане запрещён; replan только из optional
│  planner.py       │  whitelist (search_news, external_stats, get_group_context)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Toolkit          │  белый список инструментов, max_calls, tool_timeout_s
│  alfa/agent/      │  _deny: client_id только из state, не из args
│  tools.py         │  redact / mask_args → journal agent.tool
└─────┬─────────┬───┘
      │         │
      ▼         ▼
┌──────────┐  ┌──────────────────┐
│ Access   │  │ Security / LLM   │
│ access.py│  │ security.py      │
│ session_ │  │ wrap_data,       │
│ scope +  │  │ detect_injection │
│ журнал   │  │ data_policy →    │
│ профилей │  │ interpret.py     │
└──────────┘  └──────────────────┘
```

Конфигурация лимитов и политики данных: [`alfa/config/agent.yaml`](alfa/config/agent.yaml)
(`max_calls`, `tool_timeout_s`, `max_steps`, `max_seconds`, `max_tokens`,
`session_max_*`, `data_policy`, `prompt_retention_days`).
Секреты LLM — только из env (`ALFA_LLM_API_KEY`, `ALFA_ORGANIZER_KEY`);
см. [`alfa/config/config.yaml`](alfa/config/config.yaml) и [`OPERATIONS.md`](OPERATIONS.md).

## Таблица механизмов по слоям

| Слой | Механизм | Где в коде / конфиге |
| --- | --- | --- |
| API | Компания должна входить в `allowed_company_ids` сессии | [`alfa/agent/service.py`](alfa/agent/service.py) `_check`; модели [`Session`](alfa/agent/models.py) |
| API | Те же проверки на `get` / `answer` / `chat` / `trace` / `feedback` / `check_news` | [`InsightService`](alfa/agent/service.py) |
| Graph | Повторная проверка `client_id` при старте; иначе `withheld` + `access_denied` | [`alfa/agent/graph.py`](alfa/agent/graph.py) `init_context` |
| Graph | Бюджет сессии: токены и число инсайтов | `BudgetGuard`, `session_max_tokens` / `session_max_insights` в `agent.yaml` |
| Graph | Лимиты шагов, repair, replan, wall-clock, токенов на прогон | `max_steps`, `max_repair`, `max_replan`, `max_seconds`, `max_tokens` |
| Planner | Белый список имён инструментов; запрет `client_id` в args плана | [`alfa/agent/planner.py`](alfa/agent/planner.py) `_schema_error`, `_ALLOWED` |
| Planner | После gather можно только *добавить* шаги из optional whitelist | docstring и `optional_tools` / mechanisms в YAML |
| Toolkit | Неизвестный инструмент → `not_whitelisted` | [`Toolkit._dispatch`](alfa/agent/tools.py) |
| Toolkit | `client_id` из args отвергается; доступ только по state | `_deny` → `access_denied` |
| Toolkit | Лимит вызовов на инструмент за прогон | `max_calls` → `call_limit` |
| Toolkit | Таймаут на handler | `tool_timeout_s` → `timeout` |
| Toolkit | Журнал каждого вызова: tool, masked args, status, redactions, latency | `call` → `log_step("agent.tool")`; [`alfa/core/trace.py`](alfa/core/trace.py) |
| Access | Профиль/граф только внутри `session_scope(client_id)` | [`alfa/access.py`](alfa/access.py); тесты в [`test_acceptance.py`](alfa/tests/test_acceptance.py) |
| Access | Аудит чтения профиля/графа (отдельный jsonl) | `_log` в `access.py`, путь `trace` в [`alfa/config.yaml`](alfa/config.yaml) |
| Data | Контрагенты — слоты `counterparty_N`, без сырых id | `get_counterparties` в `tools.py`; демо в acceptance |
| Data | Группа — только агрегаты, без списка компаний | `get_group_context` |
| Security | Маскирование `U_\d+` в результатах и args журнала | [`alfa/agent/security.py`](alfa/agent/security.py) `redact_*`, `mask_args` |
| Security | Внешний текст в промпт как данные, не инструкции | `wrap_data` → [`interpret.py`](alfa/agent/interpret.py) |
| Security | Детектор injection в тексте события → флаг + critic | `detect_injection`; `suspicious_instructions` в graph/verify |
| LLM channel | `data_policy`: local vs openrouter; закрытые признаки режутся | `agent.yaml` `data_policy`; `_CLOSED` / `_closed_ok` в `interpret.py` |
| LLM channel | Тела промптов в журнал только при `log_prompts: true` | [`alfa/core/llm.py`](alfa/core/llm.py), `config.yaml` |
| LLM channel | Срок хранения метаданных узлов с `retain_until` | `prompt_retention_days` |
| Verify | При `suspicious` critic обязателен; иначе понижение до caveats | [`alfa/agent/verify.py`](alfa/agent/verify.py) `should_call_critic`, `critic_mandatory` |
| Surface | Инструменты только read-only (нет записи/отправки) | каталог в `_dispatch` |
| Surface | `check_news`: без fetch URL; текст → `trust_tier=unverified` | `InsightService.check_news` |
| Ingest | Сбор новостей уважает robots.txt | [`pipelines/news/pipeline.py`](pipelines/news/pipeline.py) |

## Как это закрывает требование

1. **Разграничение прав** — сессия с явным списком компаний; двойная проверка на API и в графе; инструменты не принимают чужой `client_id`; слой данных завязан на `session_scope`.
2. **Защита чувствительных данных** — маскирование идентификаторов, обезличенные слоты контрагентов, политика канала LLM, секреты из env, опциональное логирование промптов.
3. **Контроль и журналирование инструментов** — whitelist + схема args + `max_calls` + timeout; jsonl с `trace_id` и замаскированными аргументами; отдельный журнал доступа к профилю.
4. **Ограничения небезопасных действий** — только чтение; нет свободного tool-calling у модели; injection → флаг и усиленная верификация; ad-hoc новость без загрузки ссылки.

Тесты механизмов агентного слоя: [`alfa/tests/test_agent_security.py`](alfa/tests/test_agent_security.py).

## Допущения прототипа

Ниже — известные упрощения. Для хакатона их достаточно явно зафиксировать;
в промышленном контуре их нужно закрывать отдельно.

### 1. Сессия задаётся клиентом, а не IdP

HTTP-обёртка [`create_app`](alfa/agent/service.py) принимает `user_id` и
`allowed_company_ids` из тела запроса или query. Отдельной аутентификации /
подписи JWT нет: любой вызывающий может указать любой список компаний.

**Ожидание продакшена:** список компаний выдаёт IdP (или gateway) после login;
агент только проверяет claim, а не доверяет сырому JSON от UI.

### 2. Маскирование узкое

[`security.py`](alfa/agent/security.py) заменяет только шаблон `U_\d+` и явно
маскирует поле `client_id` в args журнала. ИНН, телефоны, e-mail, ФИО в текстах
новостей и свободный `reason` в `/feedback` не маскируются.

**Ожидание продакшена:** расширенный PII-пайплайн (или redaction на стороне
хранилища) до передачи текста в LLM и в долгоживущие журналы.

### 3. Детектор внедрения — эвристика, не блокер

[`detect_injection`](alfa/agent/security.py) — короткий regex. Совпадение ставит
`suspicious_instructions` / включает critic, но **не останавливает** выдачу
инсайта само по себе. Текст по-прежнему оборачивается в `<<<DATA>>>`.

**Ожидание продакшена:** более сильная фильтрация / классификатор + политика
`withhold` или `refuse` для высокоуверенных атак; мониторинг ложных срабатываний.

### 4. Роли не дают разных прав доступа

Роли `ceo` / `cfo` / `default` влияют на **подачу** текста
(`role_presentation` в `agent.yaml`), а не на набор доступных компаний или
инструментов. Разграничение сейчас только по `allowed_company_ids` и whitelist tools.

**Ожидание продакшена:** RBAC (роль → права на продукты/поля профиля/каналы LLM)
на уровне сессии и `_deny` / `data_policy`.

### Прочие сознательные упрощения

- Журнал `access.py` пишет `client_id` открытым текстом (аудит доступа);
  журнал агента маскирует его — разные цели, одно решение нужно держать в уме.
- `calc` — детерминированные формулы из белого списка функций, без произвольного кода.
- Внешний канал `openrouter` в прототипе завязан на наличие `ALFA_ORGANIZER_KEY`
  для закрытых признаков; по умолчанию `data_policy.channel: local`.
