"""Извлечение события из новости. Модель роли extract берётся из config, не из константы в коде."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alfa.core.config import PACKAGE_ROOT, load_runtime_config
from alfa.core.llm import LLMClient
from alfa.core.logging import get_logger
from alfa.core.schemas import (
    Entity,
    EventExtraction,
    ExtractionStatus,
    ExtractionValidation,
    Fact,
    NewsDoc,
    SpanNote,
)
from alfa.core.taxonomy import get_taxonomy
from alfa.core.textcheck import apply_textcheck
from alfa.core.trace import set_trace_id

logger = get_logger("alfa.extraction")

PROMPT_FILE = PACKAGE_ROOT / "extraction" / "prompts" / "extract_event.v1.md"


class _Draft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_id: str | None = None
    event_type: str | None = None
    secondary_types: list[str] = Field(default_factory=list)
    is_business_relevant: bool | None = None
    entities: list[Entity] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    risks: list[SpanNote] = Field(default_factory=list)
    opportunities: list[SpanNote] = Field(default_factory=list)
    extraction_status: str | None = None


def prompt_version(config: dict[str, Any] | None = None) -> str:
    cfg = config or load_runtime_config()
    return str(cfg["prompts"]["extract_event"])


def load_prompt_template() -> str:
    path = PROMPT_FILE
    version = prompt_version()
    expected = PACKAGE_ROOT / "extraction" / "prompts" / f"{version}.md"
    if expected.exists():
        path = expected
    return path.read_text(encoding="utf-8")


def truncate_text(text: str, token_limit: int, chars_per_token: int) -> tuple[str, bool]:
    max_chars = max(1, token_limit * chars_per_token)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _failed(doc: NewsDoc, *, model: str, version: str, trace_id: str) -> EventExtraction:
    return EventExtraction(
        doc_id=doc.id,
        event_type=None,
        secondary_types=[],
        is_business_relevant=None,
        entities=[],
        facts=[],
        risks=[],
        opportunities=[],
        extraction_status=ExtractionStatus("failed"),
        validation=ExtractionValidation(),
        model=model,
        prompt_version=version,
        trace_id=trace_id,
        truncated=False,
    )


def _render_prompt(doc: NewsDoc, text: str, truncated: bool) -> str:
    template = load_prompt_template()
    published = doc.published_at.isoformat() if doc.published_at else ""
    return (
        template.replace("{{TAXONOMY}}", get_taxonomy().event_type_prompt())
        .replace("{{DOC_ID}}", doc.id)
        .replace("{{TITLE}}", doc.title)
        .replace("{{PUBLISHED_AT}}", published)
        .replace("{{TRUNCATED}}", "true" if truncated else "false")
        .replace("{{TEXT}}", text)
    )


def extract_event(
    doc: NewsDoc,
    *,
    trace_id: str,
    client: LLMClient | None = None,
    config: dict[str, Any] | None = None,
) -> EventExtraction:
    cfg = config or load_runtime_config()
    set_trace_id(trace_id)
    llm = client or LLMClient(cfg)
    model = llm.model_for_role("extract")
    version = prompt_version(cfg)
    limits = cfg["limits"]
    text, truncated = truncate_text(
        doc.text,
        int(limits["news_tokens"]),
        int(limits["chars_per_token"]),
    )
    messages = [
        {"role": "system", "content": "Отвечай только JSON-объектом по схеме извлечения события."},
        {"role": "user", "content": _render_prompt(doc, text, truncated)},
    ]
    result = llm.complete(
        role="extract",
        model=model,
        messages=messages,
        response_schema=_Draft,
        temperature=0.0,
        trace_id=trace_id,
        prompt_version=version,
    )
    if not result.ok or not isinstance(result.parsed, _Draft):
        logger.info("extraction failed %s: %s", doc.id, result.error)
        failed = _failed(doc, model=model, version=version, trace_id=trace_id)
        return failed.model_copy(update={"truncated": truncated})

    draft = result.parsed
    taxonomy = get_taxonomy()
    if draft.event_type:
        try:
            event_type = taxonomy.event_type(draft.event_type)
        except KeyError:
            event_type = taxonomy.event_type("other_econ")
    else:
        event_type = None
    secondary = []
    for code in draft.secondary_types:
        try:
            secondary.append(taxonomy.event_type(code))
        except KeyError:
            continue

    status_code = draft.extraction_status or "ok"
    if status_code not in get_taxonomy().codes("extraction_status"):
        status_code = "ok"
    extraction = EventExtraction(
        doc_id=doc.id,
        event_type=event_type,  # type: ignore[arg-type]
        secondary_types=secondary,  # type: ignore[arg-type]
        is_business_relevant=draft.is_business_relevant,
        entities=draft.entities,
        facts=draft.facts,
        risks=draft.risks,
        opportunities=draft.opportunities,
        extraction_status=ExtractionStatus(status_code),
        validation=ExtractionValidation(),
        model=model,
        prompt_version=version,
        trace_id=trace_id,
        truncated=truncated,
    )
    checked = apply_textcheck(extraction, doc.text)
    if str(draft.extraction_status) == "failed":
        return checked.model_copy(update={"extraction_status": ExtractionStatus("failed")})
    return checked


@dataclass
class BatchExtraction:
    items: list[EventExtraction]
    status_counts: dict[str, int]


def extract_many(
    docs: list[NewsDoc],
    max_workers: int = 4,
    client: LLMClient | None = None,
    config: dict[str, Any] | None = None,
) -> BatchExtraction:
    cfg = config or load_runtime_config()
    llm = client or LLMClient(cfg)
    ordered: list[EventExtraction | None] = [None] * len(docs)

    def _one(index: int, doc: NewsDoc) -> tuple[int, EventExtraction]:
        try:
            return index, extract_event(doc, trace_id=doc.id, client=llm, config=cfg)
        except Exception as exc:
            logger.exception("extract_many dropped error for %s", doc.id)
            failed = _failed(doc, model=llm.model_for_role("extract"), version=prompt_version(cfg), trace_id=doc.id)
            failed = failed.model_copy(update={"validation": ExtractionValidation(missing_spans=[str(exc)])})
            return index, failed

    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, index, doc) for index, doc in enumerate(docs)]
        for future in as_completed(futures):
            index, item = future.result()
            ordered[index] = item
    items = [item for item in ordered if item is not None]
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.extraction_status.value)
        counts[key] = counts.get(key, 0) + 1
    return BatchExtraction(items=items, status_counts=counts)


def main() -> None:
    from alfa.data.factory import build_sources
    from alfa.data.interfaces import eligible_for_extraction

    news, _store = build_sources()
    if hasattr(news, "pending_extraction"):
        docs = news.pending_extraction()
    else:
        docs = [doc for doc in news.list_news(None, None) if eligible_for_extraction(doc)]
    batch = extract_many(docs[:3], max_workers=2)
    print(batch.status_counts)


if __name__ == "__main__":
    main()
