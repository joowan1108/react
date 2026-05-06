from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import (
    load_csv_rows,
    load_json_value,
    resolve_context_path,
)

TABLE_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]+")


def iter_source_paths(task: PublicTask, sources: list[str] | None) -> list[Path]:
    # Resolve input sources into concrete file paths.
    if not sources:
        return sorted(path for path in task.context_dir.rglob("*") if path.is_file())

    resolved_paths: list[Path] = []
    for source in sources:
        resolved = resolve_context_path(task, source)
        if resolved.is_file():
            resolved_paths.append(resolved)
            continue
        resolved_paths.extend(sorted(path for path in resolved.rglob("*") if path.is_file()))
    return resolved_paths


def to_table_name(name: str) -> str:
    # Normalize a file or table label into a SQLite-safe table name.
    sanitized = TABLE_NAME_SANITIZER.sub("_", name).strip("_").lower()
    return sanitized or "table"


def normalize_sql_value(value: Any) -> Any:
    # Serialize nested values so they can live in a single SQLite cell.
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def normalize_row_dict(row: dict[str, Any], columns: list[str]) -> list[Any]:
    # Reorder row values to match the final column order.
    return [normalize_sql_value(row.get(column)) for column in columns]


def derive_json_tables(relative_path: str, payload: Any) -> list[dict[str, Any]]:
    # Convert one JSON asset into one or more SQLite-ready table specs.
    base_table_name = to_table_name(Path(relative_path).stem)

    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        raw_rows = payload["records"]
        table_name = to_table_name(str(payload.get("table") or base_table_name))
        rows = [row if isinstance(row, dict) else {"value": row} for row in raw_rows]
    elif isinstance(payload, list):
        table_name = base_table_name
        if payload and all(isinstance(item, dict) for item in payload):
            rows = [dict(item) for item in payload]
        else:
            rows = [{"value": item} for item in payload]
    elif isinstance(payload, dict):
        table_name = base_table_name
        rows = [dict(payload)]
    else:
        table_name = base_table_name
        rows = [{"value": payload}]

    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    if not columns:
        columns = ["value"]

    normalized_rows = [normalize_row_dict(row, columns) for row in rows]
    return [
        {
            "source_path": relative_path,
            "source_type": "json",
            "table_name": table_name,
            "columns": columns,
            "rows": normalized_rows,
        }
    ]


def derive_csv_table(task: PublicTask, relative_path: str) -> dict[str, Any]:
    # Convert one CSV asset into a SQLite-ready table spec.
    columns, rows = load_csv_rows(task, relative_path)
    if not columns:
        columns = ["value"]
    normalized_rows = [
        list(row[: len(columns)]) + [""] * max(0, len(columns) - len(row))
        for row in rows
    ]
    return {
        "source_path": relative_path,
        "source_type": "csv",
        "table_name": to_table_name(Path(relative_path).stem),
        "columns": columns,
        "rows": normalized_rows,
    }


def derive_structured_tables(task: PublicTask, sources: list[str] | None = None) -> list[dict[str, Any]]:
    # Collect table specs from structured CSV/JSON context files.
    tables: list[dict[str, Any]] = []
    for path in iter_source_paths(task, sources):
        relative_path = path.relative_to(task.context_dir).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".csv":
            tables.append(derive_csv_table(task, relative_path))
        elif suffix == ".json":
            tables.extend(derive_json_tables(relative_path, load_json_value(task, relative_path)))
    return tables


def build_structured_sqlite_database(
    task: PublicTask,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    # Materialize structured sources into a temporary SQLite database.
    table_specs = derive_structured_tables(task, sources=sources)
    if not table_specs:
        raise ValueError("No structured CSV/JSON files found for sqlite conversion.")

    handle = tempfile.NamedTemporaryFile(prefix="data_agent_structured_", suffix=".sqlite", delete=False)
    handle.close()
    db_path = Path(handle.name)

    table_summaries: list[dict[str, Any]] = []
    table_names = []
    with sqlite3.connect(db_path) as conn:
        for spec in table_specs:
            table_name = str(spec["table_name"])
            table_names.append(table_name)
            columns = [str(column) for column in spec["columns"]]
            quoted_columns = [f'"{column}" TEXT' for column in columns]
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(quoted_columns)})')

            rows = [list(row) for row in spec["rows"]]
            if rows:
                placeholders = ", ".join("?" for _ in columns)
                conn.executemany(
                    f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                    rows,
                )

            table_summaries.append(
                {
                    "source_path": spec["source_path"],
                    "source_type": spec["source_type"],
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": len(rows),
                }
            )
        conn.commit()

    return {
        "database_type": "sqlite",
        "path": str(db_path),
        "table_count": len(table_summaries),
        "tables": table_summaries,
        "table_names": table_names,
    }


def scan_sources(
    task: PublicTask,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    # Public scan entrypoint.
    return build_structured_sqlite_database(task, sources=sources)
