"""
O*NET SOC code vocabulary loader.

Maps SOC codes to titles and provides keyword-based lookup.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "onet_soc_codes.json"


@lru_cache(maxsize=1)
def load_soc_codes() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_soc_by_code(code: str) -> dict | None:
    for entry in load_soc_codes():
        if entry["soc_code"] == code:
            return entry
    return None


def get_soc_title(code: str) -> str:
    entry = get_soc_by_code(code)
    return entry["title"] if entry else "Unknown"


def all_soc_codes() -> list[str]:
    return [e["soc_code"] for e in load_soc_codes()]


def all_titles() -> list[str]:
    return [e["title"] for e in load_soc_codes()]


def keyword_lookup(text: str, top_k: int = 5) -> list[dict]:
    """
    Naive keyword-overlap lookup for fallback / test use.
    Returns list of {soc_code, title, score} sorted by overlap count descending.
    """
    text_lower = text.lower()
    scores: list[tuple[float, dict]] = []
    for entry in load_soc_codes():
        hits = sum(1 for kw in entry.get("keywords", []) if kw in text_lower)
        if hits > 0:
            scores.append((hits, entry))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [
        {"soc_code": e["soc_code"], "title": e["title"], "confidence": float(hits) / 5.0}
        for hits, e in scores[:top_k]
    ]
