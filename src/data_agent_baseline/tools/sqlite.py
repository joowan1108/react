from __future__ import annotations

import sqlite3
from pathlib import Path


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'

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

            tables.append(
                {
                    "name": name,
                    "create_sql": create_sql,
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                }
            )
    return {
        "path": str(path),
        "tables": tables,
    }


def search_distinct_text_column_values(
    path: Path,
    *,
    table: str,
    column: str,
    search_terms: list[str],
    limit: int = 5,
) -> list[str]:
    normalized_terms = [term.strip().casefold() for term in search_terms if term and term.strip()]
    if not normalized_terms:
        return []

    quoted_table = _quote_identifier(table)
    quoted_column = _quote_identifier(column)
    where_clauses = [
        f"LOWER(CAST({quoted_column} AS TEXT)) LIKE ?"
        for _ in normalized_terms
    ]
    sql = (
        f"SELECT DISTINCT CAST({quoted_column} AS TEXT) AS value "
        f"FROM {quoted_table} "
        f"WHERE {quoted_column} IS NOT NULL "
        f"AND TRIM(CAST({quoted_column} AS TEXT)) <> '' "
        f"AND {' AND '.join(where_clauses)} "
        f"ORDER BY LENGTH(CAST({quoted_column} AS TEXT)) ASC, value ASC "
        f"LIMIT ?"
    )
    parameters = [f"%{term}%" for term in normalized_terms]
    parameters.append(max(1, limit))

    with _connect_read_only(path) as conn:
        rows = conn.execute(sql, parameters).fetchall()
    return [str(value) for (value,) in rows if value is not None]


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
