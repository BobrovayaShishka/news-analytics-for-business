# Персональная лента для клиента банка

Профили компаний из выгрузки, сбор новостей и ранжирование событий. Команды ниже запускаются из корня репозитория.

## Установка

Полный локальный контур, включая CPU-сборку torch и pytest:

```bash
pip install -r requirements-dev.txt
```

Этого файла достаточно для чистой машины: он подключает два списка ниже и заранее указывает индекс, с которого ставится `torch==2.14.1+cpu`.

| Файл | Назначение |
| --- | --- |
| `requirements-dev.txt` | Команда установки целиком: новости, эмбеддинги, профили, pytest. |
| `requirements.txt` | HTTP, RSS, извлечение текста, `sentence-transformers`, `openai`, PDF. Строка `--extra-index-url https://download.pytorch.org/whl/cpu` нужна потому, что колесо `torch==2.14.1+cpu` на обычном PyPI отсутствует. |
| `requirements-alfa.txt` | Профили и лента: `pandas`, `pyarrow`, `pydantic`, BM25, `pymorphy3`. |
| `requirements-news.txt` | Только сбор новостей, без torch. Этот список ставит образ в `Dockerfile`. |
| `pyproject.toml` | Зависимости пакета, без которых не импортируются профили, новости и лента. Версии совпадают с файлами выше, включая `pandas` и `pyarrow`. Torch в метаданные пакета не входит: индекс CPU-колёс читается из requirements, а не из `pyproject.toml`. `pip install -e .` поднимает ядро; плотный поиск без `sentence-transformers` считает хешами. |

## Два конфига

Оба пути считаются от корня репозитория. Каталог данных один: `data/`.

| Файл | Кто читает | Что лежит |
| --- | --- | --- |
| `alfa/config.yaml` | `alfa.settings` | CSV компаний и платежей, квантили размера, пути parquet и журнала доступа. |
| `alfa/config/config.yaml` | `alfa.core.config` | Ollama, лимиты текста, поиск, sqlite новостей. Кэш модели: `data/models`. Кэш ответов LLM: `data/cache/llm_cache.sqlite`. |

Секретов в этих файлах нет. Если задан `ALFA_LLM_API_KEY`, он перекрывает `llm.api_key`. Уровень логов: `ALFA_LOG_LEVEL` (по умолчанию INFO).

Сборка профилей дописывает в `alfa/config.yaml` блок `computed`: границы классов выручки и 99-й перцентиль суммы платежа.

## Команды

Профили. Нужны `data/df_Uid_data_hack.csv` и `data/df_Uid_payment_hack.csv`.

```bash
python -m alfa
```

Результат: `data/features/profiles.parquet` и `data/features/graph_features.parquet`.

Новости. Реестр лент — `pipelines/news/sources.yaml`.

```bash
python -m pipelines.news ingest
python -m pipelines.news report
```

Пока в `alfa/config/config.yaml` стоит `data.impl: stub`, извлечение и поиск читают фикстуры в `alfa/data/fixtures/`. Чтобы читать sqlite после `ingest`, поставьте `data.impl: pipelines.news` и `data.origin: real`.

Лента демо-клиента. Список и пути к файлам — `clients/index.json`, каталог для ранжирования — `paths.clients_dir` в `alfa/config/scoring.yaml` (`clients`). События на этом запуске — фикстуры ранжирования, не живая лента новостей.

```bash
python -m alfa.ranking.feed --client U_3263
```

Другие id: `U_26030`, `U_9250`, `U_177581`, `U_99802`. Пароль демо-пользователей в `clients/index.json` — `demo`.

Проверка и оценка:

```bash
python -m pytest
python -m alfa.eval.run_eval --task extraction --split dev --mock
python -m alfa.eval.run_eval --task search --split dev --noise none
python -m alfa.ranking.bench --smoke
```

Схема новости: `schemas/news_item.schema.json`.

Масштабирование на отрасли, типы клиентов, источники данных и сопровождение (мониторинг, карточки, новый механизм без правок Python) — в [OPERATIONS.md](OPERATIONS.md).

Если установлен make: `make profiles`, `make ingest`, `make feed`, `make test`.
