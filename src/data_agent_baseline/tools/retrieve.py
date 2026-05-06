from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import load_document_text
from data_agent_baseline.tools.scan import iter_source_paths
from data_agent_baseline.tools.text_utils import split_text_chunks, tokenize


def keyword_score(query: str, content: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    content_tokens = tokenize(content)
    if not content_tokens:
        return 0.0

    content_counts: dict[str, int] = {}
    for token in content_tokens:
        content_counts[token] = content_counts.get(token, 0) + 1

    overlap = 0.0
    unique_matches = 0
    for token in query_tokens:
        if token in content_counts:
            unique_matches += 1
            overlap += 1.0 + math.log1p(content_counts[token])

    coverage_bonus = unique_matches / len(query_tokens)
    return (overlap / math.sqrt(len(content_tokens))) + coverage_bonus


def build_markdown_database(
    task: PublicTask,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    indexed_paths: list[str] = []

    for path in iter_source_paths(task, sources):
        if path.suffix.lower() != ".md":
            continue
        relative_path = path.relative_to(task.context_dir).as_posix()
        indexed_paths.append(relative_path)
        document_name = Path(relative_path).name
        text = load_document_text(task, relative_path)
        for index, chunk in enumerate(split_text_chunks(text), start=1):
            records.append(
                {
                    "path": relative_path,
                    "record_type": "markdown_chunk",
                    "record_id": f"{relative_path}#chunk:{index}",
                    "text": f"document: {document_name} | chunk: {index} | {chunk}",
                    "metadata": {"chunk_index": index, "document": document_name},
                }
            )

    return {
        "database_type": "markdown",
        "record_count": len(records),
        "indexed_paths": indexed_paths,
        "records": records,
    }


def search_keyword_database(
    *,
    query: str,
    database: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    raw_records = database.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("database.records must be a list.")

    scored_matches: list[dict[str, Any]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        text = str(raw_record.get("text", ""))
        score = keyword_score(query, text)
        if score <= 0:
            continue

        match = dict(raw_record)
        match["score"] = round(score, 6)
        match["content"] = text
        scored_matches.append(match)

    scored_matches.sort(key=lambda item: item["score"], reverse=True)
    matches = scored_matches[:top_k]
    return {
        "query": query,
        "matches": matches,
        "match_count": len(matches),
        "record_count": len(raw_records),
        "truncated": len(scored_matches) > top_k,
    }


def retrieve_by_keyword(
    task: PublicTask,
    *,
    query: str,
    sources: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    database = build_markdown_database(task, sources=sources)
    return search_keyword_database(query=query, database=database, top_k=top_k)
