"""Таксономия из YAML. Коды в Python не дублируются списком-источником истины."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from alfa.core.config import TAXONOMY_PATH


class Taxonomy:
    def __init__(self, data: dict[str, Any], path: str):
        self.data = data
        self.path = path

    def codes(self, group: str) -> list[str]:
        block = self.data[group]
        if not isinstance(block, dict):
            raise KeyError(group)
        return list(block)

    def require(self, group: str, code: str) -> str:
        if code not in self.data[group]:
            known = ", ".join(self.codes(group))
            raise KeyError(f"{group}.{code} is not in taxonomy ({known})")
        return code

    def event_type(self, code: str) -> str:
        return self.require("event_type", code)

    def topic(self, code: str) -> str:
        return self.require("topic", code)

    def profile_features(self, topic_code: str) -> list[str]:
        mapping = self.data["topic_to_profile_features"]
        if topic_code not in mapping:
            raise KeyError(topic_code)
        return list(mapping[topic_code])

    def describe(self, group: str, code: str) -> dict[str, Any]:
        return dict(self.data[group][code])

    def event_type_prompt(self) -> str:
        lines: list[str] = []
        for code, meta in self.data["event_type"].items():
            examples = meta.get("examples") or []
            example_text = " | ".join(str(item) for item in examples)
            lines.append(
                f"- {code} ({meta.get('title', code)}): {meta.get('description', '')} "
                f"Примеры: {example_text}"
            )
        stance = ", ".join(self.codes("stance"))
        claims = ", ".join(self.codes("claim_type"))
        entities = ", ".join(self.codes("entity_type"))
        lines.append(f"Допустимые stance не извлекаются из новости целиком; сущности: {entities}.")
        lines.append(f"Типы утверждений в карточках (для ориентира): {claims}. Тональности: {stance}.")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    with TAXONOMY_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"taxonomy must be a mapping: {TAXONOMY_PATH}")
    return Taxonomy(data, str(TAXONOMY_PATH))
