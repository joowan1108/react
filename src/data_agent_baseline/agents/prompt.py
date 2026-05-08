from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
Tool hints:
- If the task depends on CSV or JSON tables, run `scan` first to create a temporary SQLite database.
- After `scan`, inspect the returned schema before writing SQL.
- Use `inspect_sqlite_schema` on any sqlite file or on the sqlite path returned by `scan`.
- Only query table names and column names that you have actually observed in schema output.
- Do not guess table names, column names, or sqlite file paths.
- If a SQL query fails with `no such table` or `no such column`, inspect schema again instead of repeating the same guess.
- If a path-related tool call fails, reuse an observed valid path instead of inventing a new one.
- Use `execute_context_sql` only after you know which database path and table names are valid.
- Each `execute_context_sql` call works on exactly one database path. Do not write one SQL query that assumes tables from different database paths exist together.
- If data comes from multiple databases, query one database first to collect keys, then query the other database with those observed keys.
- Use `retrieve` for markdown `.md` search only.
- Use `link` to connect scanned db rows with text/markdown sources, or to compare text-like sources.
- Use `summarize` to compress long documents or intermediate text before the next reasoning step when context is too long.
- Before calling `answer`, reduce the result to the smallest table that directly answers the question.
- Keep only the columns explicitly required by the question. Do not submit extra identifier, date, or descriptive columns unless the question asks for them.
- If the question asks for one value, prefer one column and one row in the final answer.
- If the question asks for counts, averages, sums, minima, or maxima, avoid submitting intermediate tables or extra grouping columns unless the question explicitly asks for them.
- If your SQL result is wider than the question requires, use another SQL query or `execute_python` to trim columns, drop extra rows, and prepare the final answer table.
- Treat `execute_python` as an allowed final formatting step when you already found the right data but need to reshape it for submission.
- If a previous step returned a formatting or parsing error, respond with one valid JSON object only and make `action_input` a JSON object.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response for structured CSV/JSON tasks:
```json
{"thought":"This looks like a structured-data task, so I should build a sqlite database first.","action":"scan","action_input":{"sources":["csv","json"]}}
```

Example response after scanning structured data:
```json
{"thought":"I should inspect the schema before writing SQL so I use valid table names.","action":"inspect_sqlite_schema","action_input":{"path":"/tmp/scanned.sqlite"}}
```

Example response when you need to reshape an intermediate result before answering:
```json
{"thought":"I found the relevant records, but I should trim the result down to only the required final columns before submitting.","action":"execute_python","action_input":{"code":"print('final formatting step placeholder')"}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool. "
        "If the task uses structured CSV or JSON data, prefer `scan` first, then `inspect_sqlite_schema`, then SQL. "
        "Before `answer`, trim the result to only the minimal rows and columns needed to answer the question."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
