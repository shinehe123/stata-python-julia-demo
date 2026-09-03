"""Normalization helpers for Chinese patent applicant names."""

from __future__ import annotations

import re
import unicodedata

_CORPORATE_SUFFIXES = (
    "有限责任公司",
    "股份有限公司",
    "集团有限公司",
    "有限公司",
    "集团公司",
    "股份公司",
)


def normalize_name(value: str, *, strip_suffix: bool = False) -> str:
    """Return a conservative matching form without changing the source value."""
    name = unicodedata.normalize("NFKC", str(value or "")).strip()
    name = re.sub(r"[\s\u3000]+", "", name)
    name = name.replace("（", "(").replace("）", ")")
    if strip_suffix:
        for suffix in _CORPORATE_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
                break
    return name.casefold()


def split_applicants(value: str) -> list[str]:
    """Split the common delimiters used by PSS while preserving order."""
    parts = re.split(r"[;；|\n]+", str(value or ""))
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = part.strip().strip(",，、")
        key = normalize_name(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result

