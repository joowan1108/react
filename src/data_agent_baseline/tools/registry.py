from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    load_document_text,
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.link import link_sources
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.retrieve import (
    build_markdown_database,
    retrieve_by_keyword,
    search_keyword_database,
)
from data_agent_baseline.tools.scan import scan_sources
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema, list_sqlite_table_names
from data_agent_baseline.tools.summarize import SummarizeRequest, summarize_text_with_model

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any], ModelAdapter | None], ToolExecutionResult]


def _resolve_sqlite_tool_path(task: PublicTask, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Missing sqlite asset: {raw_path}")
        return candidate
    return resolve_context_path(task, raw_path)


def _list_context(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _build_retrieval_database(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    raw_sources = action_input.get("sources")
    sources = [str(item) for item in raw_sources] if isinstance(raw_sources, list) else None
    return ToolExecutionResult(ok=True, content=build_markdown_database(task, sources=sources))


def _retrieve(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    query = str(action_input["query"])
    top_k = int(action_input.get("top_k", 5))
    raw_database = action_input.get("database")
    if isinstance(raw_database, dict):
        content = search_keyword_database(query=query, database=raw_database, top_k=top_k)
        return ToolExecutionResult(ok=True, content=content)

    raw_sources = action_input.get("sources")
    sources = [str(item) for item in raw_sources] if isinstance(raw_sources, list) else None
    return ToolExecutionResult(ok=True, content=retrieve_by_keyword(task, query=query, sources=sources, top_k=top_k))


def _scan(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    raw_sources = action_input.get("sources")
    sources = [str(item) for item in raw_sources] if isinstance(raw_sources, list) else None
    return ToolExecutionResult(ok=True, content=scan_sources(task, sources=sources))


def _link(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    left_db_path_raw = action_input.get("left_db_path")
    left_table = action_input.get("left_table")
    left_field = action_input.get("left_field")
    left_source = action_input.get("left_source")

    right_db_path_raw = action_input.get("right_db_path")
    right_table = action_input.get("right_table")
    right_field = action_input.get("right_field")
    right_source = action_input.get("right_source")

    contains = action_input.get("contains")
    top_k = int(action_input.get("top_k", 20))

    left_db_path = None
    if left_db_path_raw is not None:
        left_db_path = _resolve_sqlite_tool_path(task, str(left_db_path_raw))
        if not isinstance(left_table, str) or not left_table:
            raise ValueError("link.left_table must be a non-empty string when left_db_path is provided.")
        if not isinstance(left_field, str) or not left_field:
            raise ValueError("link.left_field must be a non-empty string when left_db_path is provided.")

    right_db_path = None
    if right_db_path_raw is not None:
        right_db_path = _resolve_sqlite_tool_path(task, str(right_db_path_raw))
        if not isinstance(right_table, str) or not right_table:
            raise ValueError("link.right_table must be a non-empty string when right_db_path is provided.")
        if not isinstance(right_field, str) or not right_field:
            raise ValueError("link.right_field must be a non-empty string when right_db_path is provided.")

    normalized_left_source = str(left_source) if left_source is not None else None
    normalized_right_source = str(right_source) if right_source is not None else None

    content = link_sources(
        task,
        left_db_path=left_db_path,
        left_table=str(left_table) if isinstance(left_table, str) else None,
        left_field=str(left_field) if isinstance(left_field, str) else None,
        left_source=normalized_left_source,
        right_db_path=right_db_path,
        right_table=str(right_table) if isinstance(right_table, str) else None,
        right_field=str(right_field) if isinstance(right_field, str) else None,
        right_source=normalized_right_source,
        contains=str(contains) if contains is not None else None,
        top_k=top_k,
    )
    return ToolExecutionResult(ok=True, content=content)


def _summarize(task: PublicTask, action_input: dict[str, Any], model: ModelAdapter | None) -> ToolExecutionResult:
    if model is None:
        raise RuntimeError("summarize requires an available model adapter.")

    raw_text = action_input.get("text")
    raw_path = action_input.get("path")
    if raw_text is None and raw_path is None:
        raise ValueError("summarize requires either `text` or `path`.")
    if raw_text is not None and raw_path is not None:
        raise ValueError("summarize accepts either `text` or `path`, not both.")

    if raw_text is not None:
        text = str(raw_text)
        source_path = None
    else:
        source_path = str(raw_path)
        text = load_document_text(task, source_path)

    focus = action_input.get("focus")
    max_input_chars = int(action_input.get("max_input_chars", 12000))
    content = summarize_text_with_model(
        model,
        SummarizeRequest(
            text=text,
            focus=str(focus) if focus is not None else None,
            max_input_chars=max_input_chars,
        ),
    )
    if source_path is not None:
        content["source_path"] = source_path
    return ToolExecutionResult(ok=True, content=content)


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    path = _resolve_sqlite_tool_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    path = _resolve_sqlite_tool_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    try:
        content = execute_read_only_sql(path, sql, limit=limit)
    except Exception as exc:
        error_payload: dict[str, Any] = {
            "path": str(path),
            "sql": sql,
            "error": str(exc),
        }
        lowered = str(exc).casefold()
        if "no such table" in lowered or "no such column" in lowered:
            try:
                error_payload["available_tables"] = list_sqlite_table_names(path)
            except Exception:
                pass
        raise RuntimeError(json.dumps(error_payload, ensure_ascii=False)) from exc
    return ToolExecutionResult(ok=True, content=content)


def _execute_python(task: PublicTask, action_input: dict[str, Any], _: ModelAdapter | None) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: PublicTask, action_input: dict[str, Any], __: ModelAdapter | None) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(
        self,
        task: PublicTask,
        action: str,
        action_input: dict[str, Any],
        *,
        model: ModelAdapter | None = None,
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input, model)


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "build_retrieval_database": ToolSpec(
            name="build_retrieval_database",
            description="Build a markdown retrieval database from .md files in context.",
            input_schema={"sources": ["doc/knowledge.md"]},
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context or a scanned temp sqlite database path.",
            input_schema={"path": "relative/or/absolute/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context or a scanned temp sqlite database path.",
            input_schema={"path": "relative/or/absolute/path/to/file.sqlite"},
        ),
        "link": ToolSpec(
            name="link",
            description="Link db rows to text/markdown chunks, or link text-like sources to each other. Use scan first for CSV/JSON sources.",
            input_schema={
                "left_db_path": "/tmp/scanned.sqlite",
                "left_table": "member",
                "left_field": "link_to_major",
                "right_source": "doc/major.md",
                "contains": "Angela Sanders",
                "top_k": 5,
            },
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
        "retrieve": ToolSpec(
            name="retrieve",
            description="Keyword retrieval over markdown files only. Search .md chunks with a query or a prebuilt markdown database.",
            input_schema={"query": "APS thrombosis", "sources": ["doc/knowledge.md"], "top_k": 5},
        ),
        "scan": ToolSpec(
            name="scan",
            description="Run this first when a task depends on structured CSV/JSON data. Convert structured files into a temporary SQLite database and return its path and table metadata.",
            input_schema={"sources": ["csv/member.csv", "json/patient.json"]},
        ),
        "summarize": ToolSpec(
            name="summarize",
            description="Summarize long context into a shorter fact-preserving summary for the next reasoning step. Accepts either raw `text` or a document `path`.",
            input_schema={"path": "doc/knowledge.md", "focus": "key diagnoses and abnormal lab findings", "max_input_chars": 12000},
        ),
    }
    handlers = {
        "answer": _answer,
        "build_retrieval_database": _build_retrieval_database,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "link": _link,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
        "retrieve": _retrieve,
        "scan": _scan,
        "summarize": _summarize,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
