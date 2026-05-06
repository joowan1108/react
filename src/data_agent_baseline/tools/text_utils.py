from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def split_text_chunks(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if chunks:
        return chunks
    stripped = text.strip()
    return [stripped] if stripped else []
