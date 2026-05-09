from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import load_document_text
from data_agent_baseline.tools.scan import iter_source_paths
from data_agent_baseline.tools.text_utils import split_text_chunks, tokenize

NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
RANGE_PATTERN = re.compile(
    r"\b(?:between\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?)\b",
    flags=re.IGNORECASE,
)
COMPARISON_PATTERN = re.compile(
    r"(?:>=|<=|>|<|at least|at most|less than|greater than|more than|under|over)\s*\d+(?:\.\d+)?",
    flags=re.IGNORECASE,
)
RULE_KEYWORDS = {
    "normal",
    "abnormal",
    "range",
    "between",
    "greater",
    "less",
    "over",
    "under",
    "threshold",
    "minimum",
    "maximum",
}


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


def _extract_matched_terms(query: str, content: str) -> list[str]:
    query_tokens = list(dict.fromkeys(tokenize(query)))
    content_token_set = set(tokenize(content))
    return [token for token in query_tokens if token in content_token_set]


def _extract_numeric_strings(text: str) -> list[str]:
    return list(dict.fromkeys(NUMBER_PATTERN.findall(text)))


def _extract_rule_snippets(text: str) -> list[str]:
    snippets = RANGE_PATTERN.findall(text) + COMPARISON_PATTERN.findall(text)
    return list(dict.fromkeys(snippet.strip() for snippet in snippets if snippet.strip()))


def _contains_exact_query_phrase(query: str, content: str) -> bool:
    normalized_query = query.strip().casefold()
    return bool(normalized_query) and normalized_query in content.casefold()


def _contains_rule_keyword_match(query: str, content: str) -> bool:
    query_tokens = set(tokenize(query))
    content_tokens = set(tokenize(content))
    return bool((query_tokens & RULE_KEYWORDS) and (content_tokens & RULE_KEYWORDS))


def _filename_overlap_score(query: str, relative_path: str) -> float:
    query_tokens = set(tokenize(query))
    filename_tokens = set(tokenize(Path(relative_path).stem))
    if not query_tokens or not filename_tokens:
        return 0.0
    return len(query_tokens & filename_tokens) / max(1, len(filename_tokens))


def _score_record(query: str, content: str, relative_path: str, *, mode: str) -> float:
    score = keyword_score(query, content)
    score += 0.5 * _filename_overlap_score(query, relative_path)

    if _contains_exact_query_phrase(query, content):
        score += 1.0

    if mode == "rule":
        if RANGE_PATTERN.search(content):
            score += 1.0
        if COMPARISON_PATTERN.search(content):
            score += 1.0
        if _contains_rule_keyword_match(query, content):
            score += 0.75
    elif mode == "entity":
        query_tokens = tokenize(query)
        if any(len(token) >= 6 for token in query_tokens) and _extract_matched_terms(query, content):
            score += 0.4

    return score


def _signal_level(*, score: float, exact_match: bool, range_match: bool, comparison_match: bool, matched_terms: list[str]) -> str:
    if exact_match or range_match or comparison_match:
        return "high"
    if score >= 1.5 and len(matched_terms) >= 2:
        return "medium"
    return "low"


def build_markdown_database(
    task: PublicTask,
    *,
    sources: list[str] | None = None,
    query: str | None = None,
    max_documents: int | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    indexed_paths: list[str] = []

    markdown_paths = [path for path in iter_source_paths(task, sources) if path.suffix.lower() == ".md"]
    if query and sources is None:
        markdown_paths.sort(
            key=lambda path: _filename_overlap_score(query, path.relative_to(task.context_dir).as_posix()),
            reverse=True,
        )
    if max_documents is not None:
        markdown_paths = markdown_paths[:max_documents]

    for path in markdown_paths:
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
    mode: str = "entity",
) -> dict[str, Any]:
    raw_records = database.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("database.records must be a list.")

    scored_matches: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        text = str(raw_record.get("text", ""))
        if text in seen_texts:
            continue
        relative_path = str(raw_record.get("path", ""))
        score = _score_record(query, text, relative_path, mode=mode)
        if score <= 0:
            continue
        seen_texts.add(text)

        matched_terms = _extract_matched_terms(query, text)
        exact_match = _contains_exact_query_phrase(query, text)
        range_match = bool(RANGE_PATTERN.search(text))
        comparison_match = bool(COMPARISON_PATTERN.search(text))
        extracted_rules = _extract_rule_snippets(text)
        extracted_numbers = _extract_numeric_strings(text)
        signal_level = _signal_level(
            score=score,
            exact_match=exact_match,
            range_match=range_match,
            comparison_match=comparison_match,
            matched_terms=matched_terms,
        )

        match = dict(raw_record)
        match["score"] = round(score, 6)
        match["content"] = text
        match["matched_terms"] = matched_terms
        match["contains_exact_query"] = exact_match
        match["contains_range_pattern"] = range_match
        match["contains_comparison_pattern"] = comparison_match
        match["extracted_rules"] = extracted_rules
        match["extracted_numbers"] = extracted_numbers
        match["signal_level"] = signal_level
        scored_matches.append(match)

    scored_matches.sort(key=lambda item: item["score"], reverse=True)
    matches = scored_matches[:top_k]
    low_signal = not matches or all(str(match.get("signal_level", "low")) == "low" for match in matches)
    return {
        "query": query,
        "mode": mode,
        "matches": matches,
        "match_count": len(matches),
        "record_count": len(raw_records),
        "truncated": len(scored_matches) > top_k,
        "low_signal": low_signal,
    }


def retrieve_by_keyword(
    task: PublicTask,
    *,
    query: str,
    sources: list[str] | None = None,
    top_k: int = 5,
    mode: str = "entity",
) -> dict[str, Any]:
    database = build_markdown_database(task, sources=sources, query=query, max_documents=8)
    return search_keyword_database(query=query, database=database, top_k=top_k, mode=mode)
