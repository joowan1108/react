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
3. If a structured-data task needs a complex join, value constraint, or multiple conditions, prefer `run_sql_pipeline` first. It runs a lightweight DeepEye-SQL-style flow: schema linking, value grounding, SQL candidate generation, SQL verification, and optional SQL revision, then returns a selected SQL candidate.
4. If the task depends on text or markdown evidence, use `read_doc`, `retrieve`, or `summarize` as needed.
5. Before `answer`, reduce the result to the smallest table that directly answers the question.

Structured-data rules:
- Do not repeatedly call `list_context` after the relevant files are already visible. Reuse observed paths and move on.
- Use `inspect_sqlite_schema` on any sqlite file or on the sqlite path returned by `scan`.
- Before writing a complex join or multi-condition SQL query, prefer `run_sql_pipeline` instead of issuing repeated direct SQL guesses.
- If repeated SQL guesses fail, stop guessing and switch into `run_sql_pipeline` instead of issuing more direct `execute_context_sql` calls.
- If `run_sql_pipeline` returns a selected SQL candidate, prefer executing that SQL next.
- Avoid writing a fresh SQL query immediately after `run_sql_pipeline` returns `selected_sql` unless executing that selected SQL clearly fails first.
- Prefer not to run many SQL pipeline cycles in a row unless the selected SQL execution clearly fails.
- After the selected SQL executes successfully and returns grounded non-empty rows, prefer moving to `answer` unless one tiny final formatting step is still needed.
- Only query table names and column names that you have actually observed.
- Do not guess table names, column names, or sqlite file paths.
- Use `execute_context_sql` only after you know which database path and table names are valid.
- Each `execute_context_sql` call works on exactly one database path.
- If data comes from multiple databases, query one database first to collect keys, then query the other database with those observed keys.
- Use `execute_python` mainly as a final formatting step after grounded rows are already found. Avoid using it for open-ended exploration, SQL planning, or schema discovery.

Recovery rules:
- Do not repeat the same failed action. Use the latest tool observation to choose a different next step.
- If a SQL query fails, inspect schema or switch database path instead of repeating the same SQL guess.
- After repeated SQL failures, use `run_sql_pipeline` instead of another direct SQL guess.
- If `run_sql_pipeline` marks a candidate as weak or rejected, prefer revising your plan, inspecting schema, or rerunning the pipeline instead of immediately writing another fresh direct SQL guess.
- If a path-related tool call fails, reuse an observed valid path instead of inventing a new one.
- If a previous step returned a formatting or parsing error, respond with one valid JSON object only and make `action_input` a JSON object.

Text-tool rules:
- Use `retrieve` for markdown `.md` search only.
- Prefer `retrieve` with mode `entity` for ID/name/entity lookup and mode `rule` for thresholds, ranges, or rule extraction.
- Avoid repeating the same low-signal `retrieve` query. If retrieval is weak, switch to `read_doc`, SQL, or another tool.
- Use `link` to connect scanned db rows with text or markdown sources, or to compare text-like sources.
- Use `summarize` only for long text documents or long text observations when compression will help the next reasoning step.
- Do not use `summarize` for structured CSV or JSON tables when SQL, `scan`, or direct reading is more precise.
- Use `knowledge.md` to understand business meaning, but trust observed schema and file contents over descriptive text when they conflict.

Final answer rules:
- Before calling `answer`, prefer a compact final table that directly answers the question.
- Avoid submitting intermediate tables when a simpler final result is available.
- Avoid extra identifier, date, or descriptive columns unless they help answer the question.
- If the question asks for one value, usually prefer one column and one row in the final answer.
- You may use `execute_python` only when it helps reshape the final result after you have already found the relevant data and only a final formatting step remains.
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

Example response for the higher-level SQL pipeline:
```json
{"thought":"This SQL looks complex, so I should use the SQL pipeline to get a verified candidate before executing final SQL.","action":"run_sql_pipeline","action_input":{"path":"/tmp/scanned.sqlite","question":"How many superheroes with Super Strength have height over 200cm?","num_candidates":2,"revision_rounds":1}}
```

Example response immediately after the SQL pipeline returns `selected_sql`:
```json
{"thought":"The SQL pipeline already selected a candidate, so I should execute that exact SQL next.","action":"execute_context_sql","action_input":{"path":"/tmp/scanned.sqlite","sql":"SELECT COUNT(*) FROM hero_power hp JOIN superhero s ON hp.hero_id = s.id JOIN superpower p ON hp.power_id = p.id WHERE p.power_name = 'Super Strength' AND s.height_cm > 200","limit":200}}
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


def _difficulty_strategy_text(task: PublicTask) -> str:
    difficulty = task.difficulty.casefold().strip()
    if difficulty == "easy":
        return (
            "Difficulty is easy. Prefer the lightest grounded path first: inspect context, scan structured files, "
            "inspect schema, and use direct SQL when the query is simple. Use `run_sql_pipeline` only when the SQL "
            "clearly involves tricky joins, ambiguous entity matching, or repeated SQL failures."
        )
    if difficulty in {"medium", "hard", "extreme"}:
        return (
            f"Difficulty is {difficulty}. For structured-data questions, prefer `run_sql_pipeline` earlier whenever "
            "the SQL needs joins, value grounding, multiple conditions, or non-trivial aggregation. Still skip the "
            "pipeline when the task is mostly document reasoning or when a tiny direct SQL query is obviously enough. "
            "For these harder structured tasks, `run_sql_pipeline` can internally absorb `knowledge.md` when present."
        )
    return (
        "Use the observed question and context complexity to decide between light direct SQL and `run_sql_pipeline`."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Task ID: {task.task_id}\n"
        f"Difficulty: {task.difficulty}\n"
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        f"{_difficulty_strategy_text(task)} "
        "When you have the final table, call the `answer` tool. "
        "If the task uses structured CSV or JSON data, prefer `scan` first, then `inspect_sqlite_schema`, then SQL. "
        "If the SQL looks difficult, prefer `run_sql_pipeline` before executing final SQL. "
        "If direct SQL guesses fail more than once, stop guessing and switch to `run_sql_pipeline` before writing more SQL. "
        "If the pipeline returns a selected SQL candidate, prefer executing that SQL next instead of writing a fresh SQL guess. "
        "Avoid starting another SQL planning step immediately after the pipeline returns selected_sql unless executing that selected SQL fails first. "
        "Prefer not to run many SQL pipeline cycles unless the selected SQL execution clearly fails. "
        "If the selected SQL executes successfully and returns grounded non-empty rows, prefer converging to answer unless one tiny final formatting step is still needed. "
        "Use `knowledge.md` as semantic guidance, but trust observed schema and file contents more. "
        "Before `answer`, prefer a simple final result that directly answers the question."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
