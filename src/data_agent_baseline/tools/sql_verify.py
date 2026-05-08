from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from data_agent_baseline.tools.sqlite import execute_read_only_sql


AGGREGATE_HINTS = (
    "how many",
    "count",
    "number of",
    "average",
    "avg",
    "sum",
    "total",
    "maximum",
    "minimum",
    "highest",
    "lowest",
    "max",
    "min",
)

LIST_HINTS = (
    "list",
    "show",
    "which",
    "what are",
    "who are",
    "give me",
)

TABLE_REF_RE = re.compile(
    r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:as\s+)?[A-Za-z_][A-Za-z0-9_]*)?",
    re.IGNORECASE,
)
STRING_LITERAL_RE = re.compile(r"'([^']*)'")
NUMERIC_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _extract_candidate_sqls(candidates: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            sql = candidate.strip()
            rationale = ""
        elif isinstance(candidate, dict):
            sql = str(candidate.get("sql") or "").strip()
            rationale = str(candidate.get("rationale") or "").strip()
        else:
            raise ValueError(
                f"verify_sql_candidates candidates[{index}] must be a SQL string or an object with a `sql` field."
            )

        if not sql:
            raise ValueError(f"verify_sql_candidates candidates[{index}] is missing SQL text.")

        normalized.append({"sql": sql, "rationale": rationale})
    return normalized


def _question_expects_single_value(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in AGGREGATE_HINTS)


def _question_expects_listing(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in LIST_HINTS)


def _question_expects_grouping(question: str) -> bool:
    lowered = question.casefold()
    grouping_markers = (" for each ", " by ", " per ", " grouped by ", " each ")
    return any(marker in f" {lowered} " for marker in grouping_markers)


def _extract_sql_table_refs(sql: str) -> list[str]:
    return [match.group(1) for match in TABLE_REF_RE.finditer(sql)]


def _extract_sql_string_literals(sql: str) -> list[str]:
    return [match.group(1) for match in STRING_LITERAL_RE.finditer(sql)]


def _extract_sql_numeric_literals(sql: str) -> list[str]:
    return NUMERIC_LITERAL_RE.findall(sql)


def _has_group_by(sql: str) -> bool:
    return " group by " in f" {sql.casefold()} "


def _check_schema_link_consistency(
    *,
    sql: str,
    question: str,
    schema_link_context: dict[str, object] | None,
) -> tuple[dict[str, Any], list[str]]:
    logic_checks: dict[str, Any] = {
        "uses_any_relevant_tables": None,
        "uses_only_relevant_tables": None,
        "uses_value_hints": None,
        "grouping_matches_question": None,
        "unexpected_tables": [],
        "missing_value_hints": [],
    }
    warnings: list[str] = []

    used_tables = _extract_sql_table_refs(sql)
    used_table_set = {table.casefold() for table in used_tables}
    sql_lower = sql.casefold()

    expects_grouping = _question_expects_grouping(question)
    has_group_by = _has_group_by(sql)
    logic_checks["grouping_matches_question"] = (expects_grouping == has_group_by) or (not expects_grouping and not has_group_by)
    if expects_grouping and not has_group_by:
        warnings.append("missing_group_by_for_grouping_question")
    elif not expects_grouping and has_group_by and not _question_expects_single_value(question):
        warnings.append("unexpected_group_by_for_non_grouping_question")

    if not schema_link_context:
        return logic_checks, warnings

    raw_relevant_tables = schema_link_context.get("relevant_tables")
    relevant_tables = (
        {str(table).casefold() for table in raw_relevant_tables}
        if isinstance(raw_relevant_tables, list)
        else set()
    )
    if relevant_tables:
        matched_tables = sorted(table for table in used_table_set if table in relevant_tables)
        unexpected_tables = sorted(table for table in used_table_set if table not in relevant_tables)
        logic_checks["uses_any_relevant_tables"] = bool(matched_tables)
        logic_checks["uses_only_relevant_tables"] = not unexpected_tables
        logic_checks["unexpected_tables"] = unexpected_tables
        if not matched_tables:
            warnings.append("sql_uses_no_relevant_tables")
        if unexpected_tables:
            warnings.append("sql_uses_tables_outside_schema_link_context")

    raw_value_hints = schema_link_context.get("value_hints")
    if isinstance(raw_value_hints, list) and raw_value_hints:
        all_values: list[str] = []
        missing_values: list[str] = []
        for hint in raw_value_hints:
            if not isinstance(hint, dict):
                continue
            value = str(hint.get("value") or "").strip()
            if not value:
                continue
            all_values.append(value)
            escaped_value = value.casefold()
            if escaped_value not in sql_lower:
                missing_values.append(value)
        if all_values:
            logic_checks["uses_value_hints"] = not missing_values
            logic_checks["missing_value_hints"] = missing_values
            if missing_values:
                warnings.append("sql_missing_expected_value_hints")

    return logic_checks, warnings


def _classify_acceptability(*, valid: bool, warnings: list[str], logic_checks: dict[str, Any]) -> str:
    if not valid:
        return "reject"
    if logic_checks.get("uses_any_relevant_tables") is False:
        return "reject"
    if "missing_group_by_for_grouping_question" in warnings:
        return "reject"
    severe_warning_markers = {
        "sql_missing_expected_value_hints",
        "sql_uses_tables_outside_schema_link_context",
        "result_too_tall_for_single_value_question",
        "result_too_wide_for_single_value_question",
    }
    if any(item in severe_warning_markers for item in warnings):
        return "weak_accept"
    if warnings:
        return "weak_accept"
    return "accept"


def _score_candidate(
    *,
    question: str,
    valid: bool,
    column_count: int,
    row_count: int | None,
    truncated: bool,
    warnings: list[str],
    logic_checks: dict[str, Any],
) -> float:
    if not valid:
        return 0.0

    score = 0.55
    expects_single_value = _question_expects_single_value(question)
    expects_listing = _question_expects_listing(question)

    if row_count == 0:
        score -= 0.25
    elif row_count is not None:
        score += 0.1

    if expects_single_value:
        if row_count == 1:
            score += 0.18
        if column_count == 1:
            score += 0.18
        if row_count is not None and row_count > 5:
            score -= 0.18
        if column_count > 2:
            score -= 0.15
    elif expects_listing:
        if column_count <= 3:
            score += 0.12
        elif column_count > 5:
            score -= 0.10
    else:
        if column_count <= 4:
            score += 0.08
        elif column_count > 6:
            score -= 0.10

    if truncated:
        score -= 0.05

    if logic_checks.get("uses_any_relevant_tables") is False:
        score -= 0.25
    if logic_checks.get("uses_only_relevant_tables") is False:
        score -= 0.12
    if logic_checks.get("uses_value_hints") is False:
        score -= 0.12
    if logic_checks.get("grouping_matches_question") is False:
        score -= 0.15

    score -= 0.04 * len(warnings)
    return round(max(0.0, min(1.0, score)), 2)


def verify_sql_candidates(
    *,
    path: Path,
    candidates: list[Any],
    question: str,
    schema_link_context: dict[str, object] | None = None,
    limit: int = 50,
) -> dict[str, object]:
    normalized_candidates = _extract_candidate_sqls(candidates)
    results: list[dict[str, Any]] = []

    for index, candidate in enumerate(normalized_candidates):
        sql = candidate["sql"]
        rationale = candidate["rationale"]
        warnings: list[str] = []
        errors: list[str] = []

        try:
            execution = execute_read_only_sql(path, sql, limit=limit)
            columns = [str(name) for name in execution.get("columns", [])]
            rows = execution.get("rows", [])
            row_count = int(execution.get("row_count", 0))
            truncated = bool(execution.get("truncated", False))
            logic_checks, logic_warnings = _check_schema_link_consistency(
                sql=sql,
                question=question,
                schema_link_context=schema_link_context,
            )

            if row_count == 0:
                warnings.append("empty_result")
            if truncated:
                warnings.append("result_truncated")

            if _question_expects_single_value(question):
                if len(columns) > 1:
                    warnings.append("result_too_wide_for_single_value_question")
                if row_count > 5:
                    warnings.append("result_too_tall_for_single_value_question")
            elif _question_expects_listing(question):
                if len(columns) > 5:
                    warnings.append("result_has_many_columns_for_listing_question")

            warnings.extend(item for item in logic_warnings if item not in warnings)
            score = _score_candidate(
                question=question,
                valid=True,
                column_count=len(columns),
                row_count=row_count,
                truncated=truncated,
                warnings=warnings,
                logic_checks=logic_checks,
            )
            acceptability = _classify_acceptability(
                valid=True,
                warnings=warnings,
                logic_checks=logic_checks,
            )
            results.append(
                {
                    "candidate_index": index,
                    "sql": sql,
                    "rationale": rationale,
                    "valid": True,
                    "columns": columns,
                    "column_count": len(columns),
                    "row_count": row_count,
                    "sample_rows": rows[: min(3, len(rows))],
                    "errors": errors,
                    "warnings": warnings,
                    "logic_checks": logic_checks,
                    "acceptability": acceptability,
                    "truncated": truncated,
                    "score": score,
                }
            )
        except Exception as exc:
            errors.append(str(exc))
            logic_checks, logic_warnings = _check_schema_link_consistency(
                sql=sql,
                question=question,
                schema_link_context=schema_link_context,
            )
            warnings.extend(item for item in logic_warnings if item not in warnings)
            results.append(
                {
                    "candidate_index": index,
                    "sql": sql,
                    "rationale": rationale,
                    "valid": False,
                    "columns": [],
                    "column_count": 0,
                    "row_count": None,
                    "sample_rows": [],
                    "errors": errors,
                    "warnings": warnings,
                    "logic_checks": logic_checks,
                    "acceptability": "reject",
                    "truncated": False,
                    "score": 0.0,
                }
            )

    best_candidate_index = None
    if results:
        best_result = max(results, key=lambda item: (float(item["score"]), bool(item["valid"])))
        best_candidate_index = int(best_result["candidate_index"])

    return {
        "path": str(path),
        "question": question,
        "candidate_count": len(results),
        "best_candidate_index": best_candidate_index,
        "results": results,
    }
