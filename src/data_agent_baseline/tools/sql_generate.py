from __future__ import annotations

import json
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage


def _render_schema(schema_info: dict[str, object]) -> str:
    rendered_tables: list[str] = []
    raw_tables = schema_info.get("tables")
    if not isinstance(raw_tables, list):
        raise ValueError("schema_info.tables must be a list.")

    for table_entry in raw_tables:
        if not isinstance(table_entry, dict):
            continue
        table_name = str(table_entry.get("name") or "")
        if not table_name:
            continue
        raw_columns = table_entry.get("columns")
        column_names: list[str] = []
        if isinstance(raw_columns, list):
            for column in raw_columns:
                if isinstance(column, dict):
                    name = column.get("name")
                    if isinstance(name, str) and name:
                        column_names.append(name)
        if not column_names:
            create_sql = str(table_entry.get("create_sql") or "")
            column_names = [create_sql] if create_sql else []

        primary_keys = table_entry.get("primary_keys")
        foreign_keys = table_entry.get("foreign_keys")

        lines = [f"table: {table_name}"]
        if column_names:
            lines.append(f"  columns: {', '.join(column_names)}")
        if isinstance(primary_keys, list) and primary_keys:
            lines.append(f"  primary_keys: {', '.join(str(item) for item in primary_keys)}")
        if isinstance(foreign_keys, list) and foreign_keys:
            fk_parts: list[str] = []
            for foreign_key in foreign_keys:
                if not isinstance(foreign_key, dict):
                    continue
                left = str(foreign_key.get('from') or "")
                right_table = str(foreign_key.get('to_table') or "")
                right_column = str(foreign_key.get('to_column') or "")
                if left and right_table and right_column:
                    fk_parts.append(f"{left} -> {right_table}.{right_column}")
            if fk_parts:
                lines.append(f"  foreign_keys: {', '.join(fk_parts)}")
        rendered_tables.append("\n".join(lines))
    return "\n\n".join(rendered_tables)


def build_generate_sql_candidate_messages(
    *,
    question: str,
    schema_info: dict[str, object],
    schema_link_context: dict[str, object] | None = None,
    num_candidates: int = 3,
) -> list[ModelMessage]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    schema_text = _render_schema(schema_info)
    linking_text = ""
    if schema_link_context:
        linking_text = json.dumps(schema_link_context, ensure_ascii=False, indent=2)

    instructions = [
        f"Generate up to {num_candidates} grounded read-only SQL candidates for the question.",
        "Use only the observed tables and columns from the provided schema.",
        "Each candidate must target exactly one sqlite database.",
        "Prefer candidates that differ in join, filtering, or aggregation strategy while staying grounded in the schema.",
        "Return only a JSON array.",
        'Each array item must be an object with keys "sql" and "rationale".',
        "Do not include markdown fences or extra commentary.",
    ]

    user_parts = [
        "\n".join(instructions),
        f"Question:\n{question}",
        f"Observed schema:\n{schema_text}",
    ]
    if linking_text:
        user_parts.append(f"Schema linking context:\n{linking_text}")

    return [
        ModelMessage(
            role="system",
            content=(
                "You are a SQL candidate generation helper for a ReAct-style data agent. "
                "Generate grounded candidate SQL queries and return only JSON."
            ),
        ),
        ModelMessage(role="user", content="\n\n".join(user_parts)),
    ]


def _extract_json_payload(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.casefold().startswith("json"):
            text = text[4:].lstrip()
    return text


def generate_sql_candidates_with_model(
    model: ModelAdapter,
    *,
    question: str,
    schema_info: dict[str, object],
    schema_link_context: dict[str, object] | None = None,
    num_candidates: int = 3,
) -> dict[str, object]:
    messages = build_generate_sql_candidate_messages(
        question=question,
        schema_info=schema_info,
        schema_link_context=schema_link_context,
        num_candidates=num_candidates,
    )
    raw_response = model.complete(messages)
    payload = _extract_json_payload(raw_response)

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"generate_sql_candidates returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("generate_sql_candidates must return a non-empty JSON array.")

    candidates: list[dict[str, str]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise RuntimeError(f"generate_sql_candidates item {index} is not an object.")
        sql = str(item.get("sql") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        if not sql:
            raise RuntimeError(f"generate_sql_candidates item {index} is missing SQL text.")
        candidates.append({"sql": sql, "rationale": rationale})

    return {
        "question": question,
        "candidate_count": len(candidates),
        "candidates": candidates[:num_candidates],
        "raw_response": raw_response,
    }
