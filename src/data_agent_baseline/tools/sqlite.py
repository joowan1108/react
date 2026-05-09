from __future__ import annotations

import sqlite3
from pathlib import Path


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _select_sample_columns(columns: list[dict[str, object]], *, max_columns: int = 4) -> list[str]:
    prioritized_tokens = ("date", "time", "name", "status", "type", "category")
    selected: list[str] = []

    for column in columns:
        name = str(column["name"])
        lowered = name.casefold()
        if any(token in lowered for token in prioritized_tokens):
            selected.append(name)

    for column in columns:
        name = str(column["name"])
        if name not in selected:
            selected.append(name)

    return selected[:max_columns]


def _fetch_column_sample_values(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    limit: int = 5,
) -> list[object]:
    quoted_table = _quote_identifier(table_name)
    quoted_column = _quote_identifier(column_name)
    rows = conn.execute(
        f"""
        SELECT DISTINCT {quoted_column}
        FROM {quoted_table}
        WHERE {quoted_column} IS NOT NULL
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [row[0] for row in rows]

def list_sqlite_table_names(path: Path) -> list[str]:
    with _connect_read_only(path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return [str(name) for (name,) in rows]


def inspect_sqlite_schema(path: Path) -> dict[str, object]:
    with _connect_read_only(path) as conn:
        rows = conn.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables: list[dict[str, object]] = []
        for name, create_sql in rows:
            table_info_rows = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            foreign_key_rows = conn.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()

            columns = [
                {
                    "cid": info_row[0],
                    "name": info_row[1],
                    "type": info_row[2],
                    "notnull": bool(info_row[3]),
                    "default_value": info_row[4],
                    "primary_key_position": int(info_row[5]),
                }
                for info_row in table_info_rows
            ]

            primary_keys = [
                column["name"]
                for column in sorted(
                    columns,
                    key=lambda item: int(item["primary_key_position"]),
                )
                if int(column["primary_key_position"]) > 0
            ]

            foreign_keys = [
                {
                    "id": fk_row[0],
                    "seq": fk_row[1],
                    "to_table": fk_row[2],
                    "from": fk_row[3],
                    "to_column": fk_row[4],
                    "on_update": fk_row[5],
                    "on_delete": fk_row[6],
                    "match": fk_row[7],
                }
                for fk_row in foreign_key_rows
            ]

            sample_values: dict[str, list[object]] = {}
            for column_name in _select_sample_columns(columns):
                try:
                    values = _fetch_column_sample_values(
                        conn,
                        table_name=str(name),
                        column_name=column_name,
                    )
                except sqlite3.Error:
                    values = []
                sample_values[column_name] = values

            tables.append(
                {
                    "name": name,
                    "create_sql": create_sql,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                    "sample_values": sample_values,
                }
            )
    return {
        "path": str(path),
        "tables": tables,
    }


def execute_read_only_sql(path: Path, sql: str, *, limit: int = 200) -> dict[str, object]:
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    with _connect_read_only(path) as conn:
        cursor = conn.execute(sql)
        column_names = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "path": str(path),
        "columns": column_names,
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }
