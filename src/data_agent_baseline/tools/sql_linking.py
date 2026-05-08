from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
QUOTED_TEXT_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")

TEXT_VALUE_COLUMN_SUFFIXES = (
    "name",
    "title",
    "description",
    "type",
    "status",
    "label",
    "sex",
    "gender",
    "diagnosis",
    "publisher",
    "category",
)

NUMERIC_VALUE_COLUMN_TOKENS = (
    "count",
    "num",
    "amount",
    "cost",
    "price",
    "year",
    "age",
    "height",
    "weight",
    "score",
    "date",
    "time",
    "length",
    "duration",
    "total",
    "avg",
    "average",
    "sum",
)


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _normalize_name_tokens(name: str) -> list[str]:
    return [part for part in re.split(r"[_\W]+", name.casefold()) if part]


def _extract_table_columns(create_sql: str) -> list[str]:
    columns: list[str] = []
    for line in create_sql.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        if stripped.upper().startswith(("CREATE TABLE", "PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CONSTRAINT", ")")):
            continue

        if stripped.startswith(("`", '"')):
            quote = stripped[0]
            closing = stripped.find(quote, 1)
            if closing > 1:
                columns.append(stripped[1:closing])
                continue

        match = re.match(r"([A-Za-z_][A-Za-z0-9_\-]*)\s+", stripped)
        if match:
            columns.append(match.group(1))
    return columns


def _score_name_overlap(question_tokens: set[str], name: str) -> int:
    name_tokens = set(_normalize_name_tokens(name))
    return len(question_tokens & name_tokens)


def _context_tokens_around_span(question: str, start: int, end: int, *, window_chars: int = 48) -> set[str]:
    snippet_start = max(0, start - window_chars)
    snippet_end = min(len(question), end + window_chars)
    return set(_tokenize(question[snippet_start:snippet_end]))


def _score_text_value_column(column_ref: dict[str, str], context_tokens: set[str]) -> int:
    table_tokens = set(_normalize_name_tokens(column_ref["table"]))
    column_tokens = set(_normalize_name_tokens(column_ref["column"]))
    column_name = column_ref["column"].casefold()

    score = 0
    score += 2 * len(context_tokens & column_tokens)
    score += len(context_tokens & table_tokens)

    if column_name.endswith(TEXT_VALUE_COLUMN_SUFFIXES):
        score += 2
    if any(token in column_tokens for token in ("name", "title", "label", "type", "status", "diagnosis", "category")):
        score += 1
    if column_name == "id" or column_name.endswith("_id") or column_name.startswith("link_to_"):
        score -= 3
    return score


def _score_numeric_value_column(column_ref: dict[str, str], context_tokens: set[str]) -> int:
    table_tokens = set(_normalize_name_tokens(column_ref["table"]))
    column_tokens = set(_normalize_name_tokens(column_ref["column"]))
    column_name = column_ref["column"].casefold()

    score = 0
    score += 2 * len(context_tokens & column_tokens)
    score += len(context_tokens & table_tokens)

    if any(token in column_name for token in NUMERIC_VALUE_COLUMN_TOKENS):
        score += 2
    if any(token in column_tokens for token in NUMERIC_VALUE_COLUMN_TOKENS):
        score += 1
    if column_name == "id" or column_name.endswith("_id") or column_name.startswith("link_to_"):
        score -= 3
    return score


def _guess_join_hints(
    table_columns: dict[str, list[str]],
    schema_tables: list[dict[str, object]],
) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for table_entry in schema_tables:
        if not isinstance(table_entry, dict):
            continue
        left_table = str(table_entry.get("name") or "")
        foreign_keys = table_entry.get("foreign_keys")
        if not left_table or not isinstance(foreign_keys, list):
            continue
        for foreign_key in foreign_keys:
            if not isinstance(foreign_key, dict):
                continue
            left_column = str(foreign_key.get("from") or "")
            right_table = str(foreign_key.get("to_table") or "")
            right_column = str(foreign_key.get("to_column") or "")
            if not left_column or not right_table or not right_column:
                continue
            key = (left_table, left_column, right_table, right_column)
            if key in seen:
                continue
            seen.add(key)
            joins.append(
                {
                    "left_table": left_table,
                    "left_column": left_column,
                    "right_table": right_table,
                    "right_column": right_column,
                    "confidence": 1.0,
                    "reason": "explicit foreign key metadata from sqlite schema",
                }
            )

    table_names = list(table_columns)

    for left_table in table_names:
        for left_column in table_columns[left_table]:
            left_lower = left_column.casefold()
            for right_table in table_names:
                if left_table == right_table:
                    continue
                for right_column in table_columns[right_table]:
                    right_lower = right_column.casefold()

                    confidence = 0.0
                    reason = None

                    if left_lower.endswith("_id") and right_lower == "id":
                        stem = left_lower[:-3]
                        if stem == right_table.casefold() or stem.rstrip("s") == right_table.casefold().rstrip("s"):
                            confidence = 0.95
                            reason = "foreign-key style column matches target table id"
                    elif left_lower.startswith("link_to_") and right_lower in {"id", f"{right_table}_id"}:
                        stem = left_lower.removeprefix("link_to_")
                        if stem == right_table.casefold() or stem.rstrip("s") == right_table.casefold().rstrip("s"):
                            confidence = 0.93
                            reason = "link_to_* column matches target table id"
                    elif left_lower == right_lower and left_lower.endswith("_id"):
                        confidence = 0.78
                        reason = "shared *_id column name across tables"

                    if reason is None:
                        continue

                    key = (left_table, left_column, right_table, right_column)
                    if key in seen:
                        continue
                    seen.add(key)
                    joins.append(
                        {
                            "left_table": left_table,
                            "left_column": left_column,
                            "right_table": right_table,
                            "right_column": right_column,
                            "confidence": round(confidence, 2),
                            "reason": reason,
                        }
                    )
    return joins


def _build_fk_adjacency(schema_tables: list[dict[str, object]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for table_entry in schema_tables:
        if not isinstance(table_entry, dict):
            continue
        left_table = str(table_entry.get("name") or "")
        foreign_keys = table_entry.get("foreign_keys")
        if not left_table or not isinstance(foreign_keys, list):
            continue
        for foreign_key in foreign_keys:
            if not isinstance(foreign_key, dict):
                continue
            right_table = str(foreign_key.get("to_table") or "")
            if not right_table or right_table == left_table:
                continue
            adjacency[left_table].add(right_table)
            adjacency[right_table].add(left_table)
    return adjacency


def _shortest_table_path(
    adjacency: dict[str, set[str]],
    start: str,
    goal: str,
) -> list[str]:
    if start == goal:
        return [start]

    queue: list[list[str]] = [[start]]
    visited = {start}

    while queue:
        path = queue.pop(0)
        current = path[-1]
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append(next_path)
    return []


def _apply_relational_closure(
    relevant_tables: list[str],
    schema_tables: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    unique_tables = list(dict.fromkeys(relevant_tables))
    if len(unique_tables) < 2:
        return unique_tables, []

    adjacency = _build_fk_adjacency(schema_tables)
    if not adjacency:
        return unique_tables, []

    expanded_tables = list(unique_tables)
    warnings: list[str] = []

    for index, left_table in enumerate(unique_tables):
        for right_table in unique_tables[index + 1 :]:
            path = _shortest_table_path(adjacency, left_table, right_table)
            if len(path) <= 2:
                continue
            for table_name in path[1:-1]:
                if table_name not in expanded_tables:
                    expanded_tables.append(table_name)
                    warnings.append(
                        f"Added intermediate relation '{table_name}' via FK path between '{left_table}' and '{right_table}'."
                    )

    return expanded_tables, warnings


def _extract_value_hints(question: str, relevant_columns: list[dict[str, str]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    question_lower = question.casefold()
    for match in QUOTED_TEXT_RE.finditer(question):
        value = match.group(1) or match.group(2)
        if not value:
            continue
        context_tokens = _context_tokens_around_span(question, match.start(), match.end())
        scored_columns = sorted(
            (
                (
                    _score_text_value_column(column_ref, context_tokens),
                    column_ref,
                )
                for column_ref in relevant_columns
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, column_ref in scored_columns:
            if score <= 0:
                continue
            key = (column_ref["table"], column_ref["column"], value)
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                {
                    "table": column_ref["table"],
                    "column": column_ref["column"],
                    "operator": "=",
                    "value": value,
                    "reason": "quoted value from question matched to likely text column",
                    "confidence": round(min(1.0, 0.45 + 0.08 * score), 2),
                }
            )
            if len([hint for hint in hints if hint["value"] == value]) >= 3:
                break

    for number_match in NUMBER_RE.finditer(question):
        number = number_match.group(0)
        operator = "="
        prefix_patterns = [
            (rf"(?:over|greater than|more than)\s+{re.escape(number)}", ">"),
            (rf"(?:under|less than|fewer than|below)\s+{re.escape(number)}", "<"),
            (rf"(?:at least|no less than)\s+{re.escape(number)}", ">="),
            (rf"(?:at most|no more than)\s+{re.escape(number)}", "<="),
        ]
        for pattern, detected_operator in prefix_patterns:
            if re.search(pattern, question_lower):
                operator = detected_operator
                break

        context_tokens = _context_tokens_around_span(question, number_match.start(), number_match.end())
        scored_columns = sorted(
            (
                (
                    _score_numeric_value_column(column_ref, context_tokens),
                    column_ref,
                )
                for column_ref in relevant_columns
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, column_ref in scored_columns:
            if score <= 0:
                continue
            key = (column_ref["table"], column_ref["column"], number)
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                {
                    "table": column_ref["table"],
                    "column": column_ref["column"],
                    "operator": operator,
                    "value": number,
                    "reason": "numeric literal from question matched to likely numeric column",
                    "confidence": round(min(1.0, 0.45 + 0.08 * score), 2),
                }
            )
            if len([hint for hint in hints if hint["value"] == number]) >= 3:
                break
    return hints


def build_schema_link_context(
    *,
    question: str,
    database_path: str,
    schema_info: dict[str, object],
) -> dict[str, object]:
    question_tokens = set(_tokenize(question))
    tables = schema_info.get("tables")
    if not isinstance(tables, list):
        raise ValueError("schema_info.tables must be a list.")

    table_columns: dict[str, list[str]] = {}
    relevant_tables: list[str] = []
    relevant_columns: list[dict[str, str]] = []
    warnings: list[str] = []

    table_scores: dict[str, int] = defaultdict(int)

    for table_entry in tables:
        if not isinstance(table_entry, dict):
            continue
        table_name = str(table_entry.get("name") or "")
        create_sql = str(table_entry.get("create_sql") or "")
        if not table_name:
            continue
        columns = _extract_table_columns(create_sql)
        table_columns[table_name] = columns

        table_score = _score_name_overlap(question_tokens, table_name)
        column_matches = [
            {
                "table": table_name,
                "column": column,
                "reason": "column name overlaps with question terms",
                "score": _score_name_overlap(question_tokens, column),
            }
            for column in columns
        ]
        matched_columns = [item for item in column_matches if item["score"] > 0]
        table_scores[table_name] = table_score + sum(item["score"] for item in matched_columns)

        if table_score > 0 or matched_columns:
            relevant_tables.append(table_name)
            for item in matched_columns:
                relevant_columns.append(
                    {
                        "table": item["table"],
                        "column": item["column"],
                        "reason": item["reason"],
                    }
                )

    if not relevant_tables and table_columns:
        ranked_tables = sorted(table_columns, key=lambda name: len(table_columns[name]), reverse=True)
        relevant_tables = ranked_tables[: min(2, len(ranked_tables))]
        warnings.append(
            "No strong lexical schema match found; returning fallback tables with the richest schemas."
        )

    if not relevant_columns and relevant_tables:
        for table_name in relevant_tables:
            for column in table_columns.get(table_name, [])[:4]:
                relevant_columns.append(
                    {
                        "table": table_name,
                        "column": column,
                        "reason": "fallback column from relevant table",
                    }
                )

    relevant_tables, closure_warnings = _apply_relational_closure(relevant_tables, tables)
    warnings.extend(closure_warnings)

    existing_column_keys = {(item["table"], item["column"]) for item in relevant_columns}
    for table_name in relevant_tables:
        if any(item["table"] == table_name for item in relevant_columns):
            continue
        for column in table_columns.get(table_name, [])[:4]:
            key = (table_name, column)
            if key in existing_column_keys:
                continue
            existing_column_keys.add(key)
            relevant_columns.append(
                {
                    "table": table_name,
                    "column": column,
                    "reason": "column from FK-connected intermediate relation",
                }
            )

    join_hints = _guess_join_hints(table_columns, tables)
    if relevant_tables:
        filtered_join_hints = [
            hint
            for hint in join_hints
            if hint["left_table"] in relevant_tables or hint["right_table"] in relevant_tables
        ]
    else:
        filtered_join_hints = join_hints

    value_hints = _extract_value_hints(question, relevant_columns)

    ranked_relevant_tables = sorted(
        dict.fromkeys(relevant_tables),
        key=lambda name: table_scores.get(name, 0),
        reverse=True,
    )

    return {
        "database_path": database_path,
        "question": question,
        "relevant_tables": ranked_relevant_tables,
        "relevant_columns": relevant_columns,
        "join_hints": filtered_join_hints,
        "value_hints": value_hints,
        "warnings": warnings,
    }
