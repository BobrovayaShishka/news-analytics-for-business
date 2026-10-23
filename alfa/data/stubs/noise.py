"""Детерминированный OCR-шум. Один и тот же seed даёт тот же текст.

Уровни none/low/medium задаются в config.yaml как rate. На карточки шум не накладывается.
"""

from __future__ import annotations

import random

_DIGIT_SWAP = {"0": "6", "6": "8", "8": "0", "1": "7", "7": "1"}


def add_ocr_noise(text: str, *, rate: float, seed: int) -> str:
    if rate <= 0 or not text:
        return text
    rng = random.Random(seed)
    out: list[str] = []
    for ch in text:
        if ch in _DIGIT_SWAP and rng.random() < rate:
            out.append(_DIGIT_SWAP[ch])
            continue
        if ch in ",." and rng.random() < rate:
            out.append("." if ch == "," else ",")
            continue
        if ch in "-−–—" and rng.random() < rate:
            out.append("—" if ch != "—" else "-")
            continue
        if ch == "%" and rng.random() < rate:
            continue
        if ch == " " and rng.random() < rate * 0.5:
            continue
        out.append(ch)
        if ch.isalpha() and rng.random() < rate * 0.25:
            out.append(" ")
        elif ch == " " and rng.random() < rate * 0.35:
            out.append("\n")
    result = "".join(out)
    lines = result.split("\n")
    if len(lines) >= 4 and rng.random() < min(1.0, rate * 4):
        for index in range(0, len(lines) - 1, 2):
            if rng.random() < max(rate, 0.5):
                lines[index], lines[index + 1] = lines[index + 1], lines[index]
        result = "\n".join(lines)
    return result
