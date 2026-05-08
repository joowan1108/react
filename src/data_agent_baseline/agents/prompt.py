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
Preferred workflow:
1. If the task depends on structured CSV or JSON data, prefer `scan` first.
2. After `scan`, inspect the schema before writing SQL.
3. If the task depends on text or markdown evidence, use `read_doc`, `retrieve`, or `summarize` as needed.
4. Before `answer`, reduce the result to the smallest table that directly answers the question.

Structured-data rules:
- Use `inspect_sqlite_schema` on any sqlite file or on the sqlite path returned by `scan`.
- Before writing a complex join or multi-condition SQL query, you may use `schema_link_sql_context` to identify likely relevant tables, columns, join keys, and value hints.
- For a difficult SQL task, you may use `generate_sql_candidates` to draft a few grounded SQL options before comparing them.
- For a difficult SQL task, you may compare a few candidate queries with `verify_sql_candidates` before choosing the final SQL to execute.
- If a candidate query fails or gets weak verification feedback, you may use `revise_sql_candidates` to produce improved candidates before trying again.
- Only query table names and column names that you have actually observed.
- Do not guess table names, column names, or sqlite file paths.
- Use `execute_context_sql` only after you know which database path and table names are valid.
- Each `execute_context_sql` call works on exactly one database path.
- If data comes from multiple databases, query one database first to collect keys, then query the other database with those observed keys.

Recovery rules:
- Do not repeat the same failed action. Use the latest tool observation to choose a different next step.
- If a SQL query fails, inspect schema or switch database path instead of repeating the same SQL guess.
- If a path-related tool call fails, reuse an observed valid path instead of inventing a new one.
- If a previous step returned a formatting or parsing error, respond with one valid JSON object only and make `action_input` a JSON object.

Text-tool rules:
- Use `retrieve` for markdown `.md` search only.
- Use `link` to connect scanned db rows with text or markdown sources, or to compare text-like sources.
- Use `summarize` only for long text documents or long text observations when compression will help the next reasoning step.
- Do not use `summarize` for structured CSV or JSON tables when SQL, `scan`, or direct reading is more precise.
- Use `knowledge.md` to understand business meaning, but trust observed schema and file contents over descriptive text when they conflict.

Final answer rules:
- Before calling `answer`, prefer a compact final table that directly answers the question.
- Avoid submitting intermediate tables when a simpler final result is available.
- Avoid extra identifier, date, or descriptive columns unless they help answer the question.
- If the question asks for one value, usually prefer one column and one row in the final answer.
- You may use `execute_python` when it helps reshape the final result after you have already found the relevant data.
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

Example response for difficult SQL generation:
```json
{"thought":"This join looks non-trivial, so I should draft a few grounded SQL candidates first.","action":"generate_sql_candidates","action_input":{"path":"/tmp/scanned.sqlite","question":"How many superheroes with Super Strength have height over 200cm?","num_candidates":3}}
```

Example response for revising weak SQL candidates:
```json
{"thought":"The previous candidates were weak, so I should revise them using the verification feedback.","action":"revise_sql_candidates","action_input":{"path":"/tmp/scanned.sqlite","question":"How many superheroes with Super Strength have height over 200cm?","verification_result":{"candidate_index":1,"warnings":["result_too_wide_for_single_value_question"],"errors":[]}}}
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
        "For difficult SQL, you may compare candidate queries with `verify_sql_candidates` before executing the final one. "
        "Use `knowledge.md` as semantic guidance, but trust observed schema and file contents more. "
        "Before `answer`, prefer a simple final result that directly answers the question."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
