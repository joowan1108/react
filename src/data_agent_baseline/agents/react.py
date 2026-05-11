from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


LOW_SIGNAL_ACTIONS = frozenset({"list_context", "read_doc", "retrieve", "summarize"})
SQL_ACTIONS = frozenset({"execute_context_sql", "run_sql_pipeline", "inspect_sqlite_schema", "scan"})
LISTING_HINTS = ("which", "who", "list", "show", "what are", "give me")
SINGLE_VALUE_HINTS = ("how many", "count", "number of", "average", "avg", "sum", "maximum", "minimum", "max", "min")
HELPER_COLUMN_TOKENS = frozenset(
    {
        "id",
        "date",
        "time",
        "timestamp",
        "created",
        "updated",
        "rank",
        "index",
        "idx",
        "row",
        "num",
        "number",
    }
)
AGGREGATE_COLUMN_TOKENS = frozenset(
    {
        "count",
        "total",
        "sum",
        "avg",
        "average",
        "min",
        "minimum",
        "max",
        "maximum",
    }
)
ENTITY_COLUMN_TOKENS = frozenset(
    {
        "name",
        "names",
        "gender",
        "sex",
        "type",
        "types",
        "status",
        "race",
        "event",
        "school",
        "district",
        "comment",
        "user",
        "driver",
        "disease",
        "funding",
        "country",
        "location",
        "size",
        "time",
        "number",
    }
)
MULTI_FIELD_REQUEST_TOKENS = frozenset(
    {
        "name",
        "names",
        "id",
        "ids",
        "sex",
        "gender",
        "disease",
        "type",
        "types",
        "value",
        "values",
        "age",
        "ages",
        "time",
        "times",
        "latitude",
        "longitude",
        "status",
        "comment",
        "score",
        "cost",
        "funding",
        "views",
        "race",
        "number",
    }
)
SINGLE_OUTPUT_HINTS = (
    "which ",
    "who ",
    "identify ",
    "what is ",
    "what was ",
    "what's ",
    "whats ",
    "which race",
    "which event",
    "which driver",
)
RULE_GROUNDING_TOKENS = frozenset(
    {
        "normal",
        "abnormal",
        "range",
        "between",
        "greater",
        "less",
        "more",
        "under",
        "over",
        "above",
        "below",
        "legal",
        "illegal",
        "banned",
        "restricted",
        "status",
        "threshold",
        "budget",
        "percentage",
        "ratio",
        "toxicology",
        "carcinogenic",
        "commander",
        "format",
    }
)
RULE_SOURCE_TOOL_ACTIONS = frozenset({"read_doc", "retrieve"})
RULE_PATTERN = re.compile(
    r"(?i)(between\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?)|"
    r"((?:>=|<=|>|<)\s*\d+(?:\.\d+)?)|"
    r"((?:above|below|over|under|more than|less than)\s+\d+(?:\.\d+)?)|"
    r"((?:legal|banned|restricted|normal|abnormal)\b)"
)


@dataclass(frozen=True, slots=True)
class AgentPolicyState:
    next_step_index: int
    remaining_steps: int
    near_limit: bool
    has_selected_sql_waiting: bool
    has_ready_sql_result: bool
    has_grounded_sql_result: bool
    mixed_source_mode: bool
    doc_rule_mode: bool
    recent_rule_hints: tuple[str, ...]
    repeated_same_action_count: int
    recent_sql_error_count: int
    list_context_count: int
    low_signal_count: int


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


def _build_parse_error_observation(exc: Exception) -> dict[str, object]:
    error_text = str(exc)
    observation: dict[str, object] = {
        "ok": False,
        "error_type": "parse_error",
        "error": error_text,
        "retry_hint": (
            "Return exactly one JSON object inside one ```json fenced block. "
            "Use keys `thought`, `action`, and `action_input`. "
            "`action_input` must be a JSON object, even when empty."
        ),
    }
    if "one JSON object" in error_text:
        observation["format_hint"] = "Do not include multiple JSON objects or extra trailing text."
    if "action_input must be a JSON object" in error_text:
        observation["format_hint"] = "Set `action_input` to an object like {} instead of a list or string."
    return observation


def _build_tool_error_observation(exc: Exception, action: str) -> dict[str, object]:
    error_text = str(exc)
    observation: dict[str, object] = {
        "ok": False,
        "tool": action,
        "error_type": "tool_error",
        "error": error_text,
    }

    try:
        payload = json.loads(error_text)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        observation["error_details"] = payload
        if "path" in payload:
            observation["path"] = payload.get("path")
        if "sql" in payload:
            observation["sql"] = payload.get("sql")
        if "available_tables" in payload:
            observation["available_tables"] = payload.get("available_tables")
        nested_error = payload.get("error")
        if isinstance(nested_error, str) and nested_error:
            observation["error"] = nested_error

    lowered = error_text.casefold()
    if isinstance(payload, dict):
        nested_error_text = str(payload.get("error") or "")
        if nested_error_text:
            lowered = nested_error_text.casefold()

    if "no such table" in lowered or "no such column" in lowered:
        observation["retry_hint"] = (
            "Inspect the database schema again and only use observed table and column names."
        )
        if isinstance(payload, dict) and payload.get("available_tables"):
            observation["diagnosis"] = "The SQL references a table or column that does not exist in the selected database."
    elif "missing sqlite asset" in lowered or "path escapes context dir" in lowered:
        observation["retry_hint"] = (
            "Reuse a valid observed database path or context-relative path instead of inventing one."
        )
        match = re.search(r"Missing sqlite asset:\s*(.+)$", error_text)
        if match:
            observation["missing_path"] = match.group(1).strip()
    elif "missing context asset" in lowered:
        observation["retry_hint"] = (
            "Use an exact path returned by list_context instead of a glob or guessed path."
        )
        match = re.search(r"Missing context asset:\s*(.+)$", error_text)
        if match:
            observation["missing_path"] = match.group(1).strip()

    if action == "run_sql_pipeline" and isinstance(payload, dict):
        observation["retry_hint"] = (
            "The SQL pipeline failed internally. Inspect the structured error details, then retry with the correct path/schema or switch sources."
        )
    return observation


def _augment_pipeline_observation_hint(action: str, observation: dict[str, object]) -> dict[str, object]:
    if not observation.get("ok"):
        return observation

    if action == "run_sql_pipeline":
        content = observation.get("content")
        if not isinstance(content, dict):
            return observation

        selected_sql = content.get("selected_sql")
        selected_acceptability = str(content.get("selected_acceptability") or "")
        recommended_next_action = str(content.get("recommended_next_action") or "")

        updated = dict(observation)
        if isinstance(selected_sql, str) and selected_sql.strip():
            updated["retry_hint"] = (
                "The SQL pipeline already selected a candidate. Prefer executing that SQL next instead of starting a new SQL planning loop."
            )
            updated["suggested_next_actions"] = ["execute_context_sql"]
            return updated

        if selected_acceptability == "reject" or recommended_next_action == "run_sql_pipeline":
            updated["retry_hint"] = (
                "The SQL pipeline did not produce an acceptable candidate. Inspect the schema or change strategy before trying another direct SQL guess."
            )
            updated["suggested_next_actions"] = ["inspect_sqlite_schema", "run_sql_pipeline"]
            return updated

        return observation

    if action == "verify_sql_candidates":
        content = observation.get("content")
        if not isinstance(content, dict):
            return observation
        results = content.get("results")
        if not isinstance(results, list) or not results:
            return observation
        best_index = content.get("best_candidate_index")
        best_result = None
        if isinstance(best_index, int):
            for result in results:
                if isinstance(result, dict) and int(result.get("candidate_index", -1)) == best_index:
                    best_result = result
                    break
        if best_result is None and isinstance(results[0], dict):
            best_result = results[0]
        if not isinstance(best_result, dict):
            return observation

        acceptability = str(best_result.get("acceptability") or "")
        warnings = best_result.get("warnings")
        warning_list = [str(item) for item in warnings] if isinstance(warnings, list) else []

        updated = dict(observation)
        if acceptability == "reject":
            updated["retry_hint"] = (
                "The best SQL candidate is still rejected. Prefer rerunning the SQL pipeline instead of executing more direct SQL guesses."
            )
            updated["suggested_next_actions"] = ["run_sql_pipeline"]
        elif acceptability == "weak_accept" or warning_list:
            updated["retry_hint"] = (
                "The best SQL candidate is only weakly acceptable. Prefer rerunning the SQL pipeline before falling back to another fresh SQL guess."
            )
            updated["suggested_next_actions"] = ["run_sql_pipeline", "execute_context_sql"]
        else:
            updated["retry_hint"] = (
                "The best verified candidate looks acceptable. Prefer executing that candidate "
                "instead of writing a fresh SQL guess."
            )
            updated["suggested_next_actions"] = ["execute_context_sql"]
        return updated

    return observation


def _summarize_run_sql_pipeline_content(content: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": content.get("path"),
        "selected_sql": content.get("selected_sql"),
        "selected_acceptability": content.get("selected_acceptability"),
        "recommended_next_action": content.get("recommended_next_action"),
    }

    link_context = content.get("schema_link_context")
    if isinstance(link_context, dict):
        summary["schema_link_summary"] = {
            "relevant_tables": link_context.get("relevant_tables", []),
            "join_hint_count": len(link_context.get("join_hints", []))
            if isinstance(link_context.get("join_hints"), list)
            else 0,
            "value_hint_count": len(link_context.get("value_hints", []))
            if isinstance(link_context.get("value_hints"), list)
            else 0,
            "knowledge_hint_count": len(link_context.get("knowledge_hints", []))
            if isinstance(link_context.get("knowledge_hints"), list)
            else 0,
            "warnings": link_context.get("warnings", []),
        }

    final_verification = content.get("final_verification")
    if isinstance(final_verification, dict):
        verification_summary: dict[str, object] = {
            "best_candidate_index": final_verification.get("best_candidate_index"),
            "best_candidate_acceptability": final_verification.get("best_candidate_acceptability"),
            "best_candidate_sql": final_verification.get("best_candidate_sql"),
        }
        results = final_verification.get("results")
        if isinstance(results, list):
            verification_summary["candidate_count"] = len(results)
            best_index = final_verification.get("best_candidate_index")
            if isinstance(best_index, int):
                for result in results:
                    if isinstance(result, dict) and int(result.get("candidate_index", -1)) == best_index:
                        verification_summary["best_candidate_warnings"] = result.get("warnings", [])
                        verification_summary["best_candidate_errors"] = result.get("errors", [])
                        verification_summary["best_candidate_score"] = result.get("score")
                        break
        summary["verification_summary"] = verification_summary

    revisions = content.get("revisions")
    if isinstance(revisions, list):
        summary["revision_count"] = len(revisions)

    return summary


def _slim_success_observation(action: str, observation: dict[str, object]) -> dict[str, object]:
    if not observation.get("ok"):
        return observation

    content = observation.get("content")
    if not isinstance(content, dict):
        return observation

    updated = dict(observation)
    if action == "run_sql_pipeline":
        updated["content"] = _summarize_run_sql_pipeline_content(content)
    return updated


def _augment_selected_sql_execution_observation(
    previous_observation: dict[str, object] | None,
    observation: dict[str, object],
) -> dict[str, object]:
    if not previous_observation:
        return observation

    expected_path, expected_sql = _extract_selected_sql(previous_observation)
    if expected_sql is None:
        return observation

    content = observation.get("content")
    if not isinstance(content, dict):
        return observation

    row_count = content.get("row_count")
    truncated = bool(content.get("truncated"))
    updated = dict(observation)

    if isinstance(row_count, int) and row_count > 0 and not truncated:
        updated["retry_hint"] = (
            "The selected SQL executed successfully and returned non-empty results. "
            "Answer next unless you only need one very small final reshaping step."
        )
        updated["suggested_next_actions"] = ["answer", "execute_python"]
        updated["selected_sql_executed"] = True
        updated["ready_to_answer_if_grounded"] = True
        if expected_path is not None:
            updated["selected_path"] = expected_path
        updated["selected_sql"] = expected_sql
        return updated

    if isinstance(row_count, int) and row_count == 0:
        updated["retry_hint"] = (
            "The selected SQL executed but returned no rows. Prefer inspecting schema or rerunning the SQL pipeline "
            "instead of drifting into repeated direct SQL guesses."
        )
        updated["suggested_next_actions"] = ["inspect_sqlite_schema", "run_sql_pipeline"]
        updated["selected_sql_executed"] = True
        return updated

    if truncated:
        updated["retry_hint"] = (
            "The selected SQL executed but returned a truncated result. Prefer answering only after narrowing to the final shape."
        )
        updated["suggested_next_actions"] = ["execute_python", "answer"]
        updated["selected_sql_executed"] = True
        return updated

    return observation


def _extract_selected_sql(observation: dict[str, object] | None) -> tuple[str | None, str | None]:
    if not isinstance(observation, dict) or not observation.get("ok"):
        return None, None
    if observation.get("tool") != "run_sql_pipeline":
        return None, None
    content = observation.get("content")
    if not isinstance(content, dict):
        return None, None
    selected_sql = content.get("selected_sql")
    path = content.get("path")
    if not isinstance(selected_sql, str) or not selected_sql.strip():
        return None, None
    return str(path) if path is not None else None, selected_sql.strip()


def _normalize_action_signature(action: str, action_input: dict[str, object]) -> str:
    return f"{action}::{json.dumps(action_input, ensure_ascii=False, sort_keys=True)}"


def _count_recent_same_action(
    steps: list[StepRecord],
    action: str,
    action_input: dict[str, object],
    *,
    window: int = 4,
) -> int:
    target = _normalize_action_signature(action, action_input)
    count = 0
    for step in reversed(steps[-window:]):
        if _normalize_action_signature(step.action, step.action_input) != target:
            continue
        count += 1
    return count


def _count_recent_sql_errors(steps: list[StepRecord], *, window: int = 6) -> int:
    count = 0
    for step in steps[-window:]:
        if step.ok:
            continue
        observation = step.observation
        if observation.get("error_type") != "tool_error":
            continue
        if step.action == "execute_context_sql":
            count += 1
    return count


def _find_last_ready_sql_observation(steps: list[StepRecord]) -> dict[str, object] | None:
    for step in reversed(steps):
        observation = step.observation
        if not isinstance(observation, dict) or not observation.get("ok"):
            continue
        if observation.get("tool") != "execute_context_sql":
            continue
        if observation.get("ready_to_answer_if_grounded"):
            return observation
        content = observation.get("content")
        if not isinstance(content, dict):
            continue
        row_count = content.get("row_count")
        truncated = bool(content.get("truncated"))
        columns = content.get("columns")
        if (
            isinstance(row_count, int)
            and row_count > 0
            and not truncated
            and isinstance(columns, list)
            and len(columns) == 1
        ):
            return observation
    return None


def _find_last_grounded_sql_observation(steps: list[StepRecord]) -> dict[str, object] | None:
    for step in reversed(steps):
        observation = step.observation
        if not isinstance(observation, dict) or not observation.get("ok"):
            continue
        if observation.get("tool") != "execute_context_sql":
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            continue
        row_count = content.get("row_count")
        truncated = bool(content.get("truncated"))
        if isinstance(row_count, int) and row_count > 0 and not truncated:
            return observation
    return None


def _find_last_pending_selected_sql_observation(steps: list[StepRecord]) -> dict[str, object] | None:
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        observation = step.observation
        if not isinstance(observation, dict) or not observation.get("ok"):
            continue
        if observation.get("tool") != "run_sql_pipeline":
            continue
        _, selected_sql = _extract_selected_sql(observation)
        if not selected_sql:
            return None
        for later_step in steps[index + 1 :]:
            if later_step.action != "execute_context_sql":
                continue
            later_observation = later_step.observation
            if not isinstance(later_observation, dict) or not later_observation.get("ok"):
                continue
            if later_observation.get("selected_sql") == selected_sql:
                return None
        return observation
    return None


def _build_policy_state(
    state: AgentRuntimeState,
    *,
    task: PublicTask,
    next_step_index: int,
    max_steps: int,
) -> AgentPolicyState:
    remaining_steps = max(0, max_steps - next_step_index + 1)
    pending_selected_sql = _find_last_pending_selected_sql_observation(state.steps)
    ready_sql_observation = _find_last_ready_sql_observation(state.steps)
    grounded_sql_observation = _find_last_grounded_sql_observation(state.steps)
    recent_rule_hints = _find_recent_rule_hints(task, state.steps)
    return AgentPolicyState(
        next_step_index=next_step_index,
        remaining_steps=remaining_steps,
        near_limit=remaining_steps <= 3,
        has_selected_sql_waiting=pending_selected_sql is not None,
        has_ready_sql_result=ready_sql_observation is not None,
        has_grounded_sql_result=grounded_sql_observation is not None,
        mixed_source_mode=_task_has_mixed_structured_sources(task),
        doc_rule_mode=_task_needs_doc_rule_grounding(task),
        recent_rule_hints=recent_rule_hints,
        repeated_same_action_count=0,
        recent_sql_error_count=_count_recent_sql_errors(state.steps),
        list_context_count=sum(1 for step in state.steps if step.action == "list_context"),
        low_signal_count=sum(1 for step in state.steps if step.action in LOW_SIGNAL_ACTIONS),
    )


def _task_has_mixed_structured_sources(task: PublicTask) -> bool:
    context_dir = Path(task.context_dir)
    has_db = any(context_dir.rglob("*.db"))
    has_csv = any(context_dir.rglob("*.csv"))
    has_json = any(context_dir.rglob("*.json"))
    return sum([has_db, has_csv, has_json]) >= 2


def _task_needs_doc_rule_grounding(task: PublicTask) -> bool:
    context_dir = Path(task.context_dir)
    question_tokens = set(_tokenize_text(task.question))
    has_doc = any(context_dir.rglob("*.md"))
    if not has_doc:
        return False
    if question_tokens & RULE_GROUNDING_TOKENS:
        return True
    lowered = task.question.casefold()
    return any(
        phrase in lowered
        for phrase in (
            "how much faster",
            "what percentage",
            "normal level",
            "abnormal level",
            "legal status",
            "budget",
        )
    )


def _normalize_rule_hint(text: str) -> str:
    return " ".join(text.split())


def _extract_rule_hints_from_text(task: PublicTask, text: str) -> list[str]:
    question_tokens = set(_tokenize_text(task.question))
    hints: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip(" -\t")
        if len(line) < 12:
            continue
        lowered = line.casefold()
        tokens = set(_tokenize_text(line))
        if not RULE_PATTERN.search(line):
            if not (question_tokens & tokens and {"format", "status", "legal"} & tokens):
                continue
        if question_tokens and len(question_tokens & tokens) == 0 and not any(
            token in lowered for token in ("legal", "banned", "restricted", "normal", "abnormal")
        ):
            continue
        normalized = _normalize_rule_hint(line)
        if normalized in seen:
            continue
        seen.add(normalized)
        hints.append(normalized[:180])
        if len(hints) >= 3:
            break
    return hints


def _collect_rule_hints_from_observation(task: PublicTask, observation: dict[str, object]) -> list[str]:
    if not observation.get("ok"):
        return []
    tool = str(observation.get("tool") or "")
    if tool not in RULE_SOURCE_TOOL_ACTIONS:
        return []

    content = observation.get("content")
    if not isinstance(content, dict):
        return []

    candidate_texts: list[str] = []
    preview = content.get("preview")
    if isinstance(preview, str) and preview.strip():
        candidate_texts.append(preview)

    matches = content.get("matches")
    if isinstance(matches, list):
        for match in matches[:6]:
            if not isinstance(match, dict):
                continue
            for key in ("content", "text"):
                value = match.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_texts.append(value)
                    break

    hints: list[str] = []
    seen: set[str] = set()
    for text in candidate_texts:
        for hint in _extract_rule_hints_from_text(task, text):
            if hint in seen:
                continue
            seen.add(hint)
            hints.append(hint)
            if len(hints) >= 3:
                return hints
    return hints


def _find_recent_rule_hints(task: PublicTask, steps: list[StepRecord]) -> tuple[str, ...]:
    hints: list[str] = []
    seen: set[str] = set()
    for step in reversed(steps):
        for hint in _collect_rule_hints_from_observation(task, step.observation):
            if hint in seen:
                continue
            seen.add(hint)
            hints.append(hint)
            if len(hints) >= 3:
                return tuple(hints)
    return tuple(hints)


def _build_control_message(task: PublicTask, policy: AgentPolicyState) -> str | None:
    hints: list[str] = []
    difficulty = task.difficulty.casefold().strip()
    if difficulty == "easy":
        hints.append("For easy tasks, prefer the shortest grounded path and avoid extra exploration.")
    if policy.mixed_source_mode:
        hints.append(
            "This is a mixed-source task. Prefer extracting qualifying IDs or key values from one structured source first, then query the downstream source with those observed keys."
        )
        hints.append(
            "Do not assume one SQL query can span multiple databases or files. Bridge the sources in stages."
        )
    if policy.doc_rule_mode and policy.recent_rule_hints:
        hints.append(
            "You already observed rule-like guidance from docs or knowledge. Apply those grounded rules to SQL or Python now instead of doing more document search."
        )
        for hint in policy.recent_rule_hints:
            hints.append(f"Observed rule hint: {hint}")
    elif policy.doc_rule_mode:
        hints.append(
            "This task likely needs rule grounding from docs or knowledge. Extract thresholds or legal/status mappings first, then apply them to structured data."
        )
    if policy.has_selected_sql_waiting:
        hints.append("A selected SQL candidate already exists. Execute it next before starting new exploration.")
    if policy.recent_sql_error_count >= 2:
        hints.append("Recent direct SQL attempts failed. Prefer schema inspection or run_sql_pipeline instead of more fresh SQL guesses.")
    if policy.near_limit and policy.has_ready_sql_result:
        hints.append("You are near the step limit and already have grounded rows. Prefer final shaping or answer now.")
    elif policy.near_limit and policy.has_grounded_sql_result:
        hints.append("You are near the step limit and already have a non-empty SQL result. Prefer answer now unless one tiny reshape is clearly required.")
    elif policy.near_limit:
        hints.append("You are near the step limit. Avoid low-signal exploration and move toward the smallest grounded answer.")
    elif policy.has_ready_sql_result:
        hints.append("A grounded SQL result is already answer-ready. Prefer answer or one tiny final reshape over more exploration.")
    elif policy.has_grounded_sql_result:
        hints.append("A grounded SQL result already exists. Avoid more exploration unless the current columns still clearly need reshaping.")
    if policy.list_context_count >= 2:
        hints.append("Do not call list_context again unless a missing-path error forces it.")
    if not hints:
        return None
    return "Control hints:\n- " + "\n- ".join(hints)


def _augment_text_rule_observation(task: PublicTask, action: str, observation: dict[str, object]) -> dict[str, object]:
    if action not in RULE_SOURCE_TOOL_ACTIONS or not observation.get("ok"):
        return observation
    if not _task_needs_doc_rule_grounding(task):
        return observation

    rule_hints = _collect_rule_hints_from_observation(task, observation)
    if not rule_hints:
        return observation

    updated = dict(observation)
    updated["retry_hint"] = (
        "You already observed rule-like guidance from docs or knowledge. Apply those rules to SQL or Python now instead of continuing broad document search."
    )
    updated["suggested_next_actions"] = ["execute_context_sql", "execute_python"]
    updated["ready_to_apply_rule_grounding"] = True
    updated["derived_rule_hints"] = rule_hints
    return updated


def _question_expects_single_value(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in SINGLE_VALUE_HINTS)


def _question_expects_listing(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in LISTING_HINTS)


def _tokenize_text(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.casefold()) if token]


def _is_helper_column_name(column_name: str) -> bool:
    tokens = _tokenize_text(column_name)
    if not tokens:
        return False
    if tokens[-1] == "id":
        return True
    return all(token in HELPER_COLUMN_TOKENS for token in tokens)


def _is_aggregate_column_name(column_name: str) -> bool:
    tokens = _tokenize_text(column_name)
    return any(token in AGGREGATE_COLUMN_TOKENS for token in tokens)


def _is_entity_like_column_name(column_name: str) -> bool:
    tokens = _tokenize_text(column_name)
    return any(token in ENTITY_COLUMN_TOKENS for token in tokens)


def _question_mentions_column(question: str, column_name: str) -> bool:
    question_lowered = question.casefold()
    column_lowered = column_name.casefold()
    if column_lowered in question_lowered:
        return True

    column_tokens = [token for token in _tokenize_text(column_name) if token not in HELPER_COLUMN_TOKENS]
    if not column_tokens:
        column_tokens = _tokenize_text(column_name)
    if not column_tokens:
        return False

    question_tokens = set(_tokenize_text(question))
    return all(token in question_tokens for token in column_tokens)


def _project_answer_table(
    columns: list[str],
    rows: list[list[Any]],
    keep_indexes: list[int],
) -> tuple[list[str], list[list[Any]]]:
    projected_columns = [columns[index] for index in keep_indexes]
    projected_rows = [[row[index] for index in keep_indexes] for row in rows]
    return projected_columns, projected_rows


def _is_numericish_value(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%)?", text))


def _column_is_mostly_numeric(rows: list[list[Any]], index: int) -> bool:
    observed = [row[index] for row in rows if len(row) > index and str(row[index]).strip()]
    if not observed:
        return False
    numeric_count = sum(1 for value in observed if _is_numericish_value(value))
    return numeric_count >= max(1, len(observed) * 0.7)


def _question_requests_multiple_output_fields(question: str) -> bool:
    lowered = question.casefold()
    if "," in lowered:
        return True
    if " and " not in lowered:
        return False
    tokens = _tokenize_text(question)
    marker_count = sum(1 for token in tokens if token in MULTI_FIELD_REQUEST_TOKENS)
    return marker_count >= 2


def _question_prefers_single_output_column(question: str) -> bool:
    lowered = question.casefold().strip()
    if _question_expects_single_value(question):
        return True
    if _question_requests_multiple_output_fields(question):
        return False
    return any(lowered.startswith(prefix) for prefix in SINGLE_OUTPUT_HINTS)


def _score_output_column(
    *,
    task: PublicTask,
    column: str,
    rows: list[list[Any]],
    index: int,
) -> int:
    score = 0
    question = task.question
    if _question_mentions_column(question, column):
        score += 8
    if not _is_helper_column_name(column):
        score += 3
    else:
        score -= 6
    if _is_entity_like_column_name(column):
        score += 4
    if _is_aggregate_column_name(column):
        score += 5 if _question_expects_single_value(question) else -2

    is_numeric = _column_is_mostly_numeric(rows, index)
    if _question_expects_single_value(question):
        score += 4 if is_numeric else -1
    elif _question_prefers_single_output_column(question):
        score += 2 if not is_numeric else -1
        if _is_entity_like_column_name(column):
            score += 2

    return score


def _sanitize_answer_payload(
    task: PublicTask,
    columns: list[str],
    rows: list[list[Any]],
) -> tuple[list[str], list[list[Any]]]:
    if len(columns) <= 1:
        return columns, rows

    question = task.question
    row_count = len(rows)
    mentioned_indexes = [
        index for index, column in enumerate(columns) if _question_mentions_column(question, column)
    ]
    mentioned_non_helper_indexes = [
        index for index in mentioned_indexes if not _is_helper_column_name(columns[index])
    ]
    non_helper_indexes = [
        index for index, column in enumerate(columns) if not _is_helper_column_name(column)
    ]
    entity_like_indexes = [
        index for index, column in enumerate(columns) if _is_entity_like_column_name(column)
    ]
    prefers_single_output_column = _question_prefers_single_output_column(question)
    requests_multiple_output_fields = _question_requests_multiple_output_fields(question)

    if _question_expects_single_value(question) and row_count == 1:
        preferred_indexes = [
            index
            for index, column in enumerate(columns)
            if _question_mentions_column(question, column) or _is_aggregate_column_name(column)
        ]
        preferred_non_helper = [
            index for index in preferred_indexes if not _is_helper_column_name(columns[index])
        ]
        if len(preferred_non_helper) == 1:
            return _project_answer_table(columns, rows, preferred_non_helper)
        if len(non_helper_indexes) == 1:
            return _project_answer_table(columns, rows, non_helper_indexes)
        return columns, rows

    if prefers_single_output_column:
        ranked_indexes = sorted(
            range(len(columns)),
            key=lambda index: (
                _score_output_column(task=task, column=columns[index], rows=rows, index=index),
                -index,
            ),
            reverse=True,
        )
        if ranked_indexes:
            best_index = ranked_indexes[0]
            best_score = _score_output_column(task=task, column=columns[best_index], rows=rows, index=best_index)
            if best_score >= 4:
                return _project_answer_table(columns, rows, [best_index])

    keep_indexes: list[int] = []
    for index, column in enumerate(columns):
        if _question_mentions_column(question, column) or not _is_helper_column_name(column):
            keep_indexes.append(index)

    if requests_multiple_output_fields and mentioned_non_helper_indexes:
        deduped_indexes: list[int] = []
        seen_indexes: set[int] = set()
        for index in mentioned_non_helper_indexes:
            if index in seen_indexes:
                continue
            deduped_indexes.append(index)
            seen_indexes.add(index)
        if 1 < len(deduped_indexes) < len(columns):
            return _project_answer_table(columns, rows, deduped_indexes)

    if keep_indexes and len(keep_indexes) < len(columns):
        return _project_answer_table(columns, rows, keep_indexes)

    if _question_expects_listing(question) and mentioned_non_helper_indexes and len(mentioned_non_helper_indexes) < len(columns):
        return _project_answer_table(columns, rows, mentioned_non_helper_indexes)

    if _question_expects_listing(question) and entity_like_indexes and len(entity_like_indexes) == 1 and len(columns) > 1:
        return _project_answer_table(columns, rows, entity_like_indexes)

    return columns, rows


def _sanitize_answer_action_input(
    task: PublicTask,
    action_input: dict[str, object],
) -> dict[str, object]:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return action_input
    if not columns or not all(isinstance(item, str) for item in columns):
        return action_input
    if not all(isinstance(row, list) and len(row) == len(columns) for row in rows):
        return action_input

    normalized_columns = list(columns)
    normalized_rows = [list(row) for row in rows]
    trimmed_columns, trimmed_rows = _sanitize_answer_payload(task, normalized_columns, normalized_rows)
    return {
        "columns": trimmed_columns,
        "rows": trimmed_rows,
    }


def _augment_general_sql_execution_observation(
    task: PublicTask,
    observation: dict[str, object],
) -> dict[str, object]:
    content = observation.get("content")
    if not isinstance(content, dict):
        return observation

    row_count = content.get("row_count")
    truncated = bool(content.get("truncated"))
    columns = content.get("columns")
    rows = content.get("rows")
    if not isinstance(row_count, int) or row_count <= 0 or truncated:
        return observation
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        return observation
    if not isinstance(rows, list):
        return observation

    updated = dict(observation)
    sanitized_columns, sanitized_rows = _sanitize_answer_payload(
        task,
        list(columns),
        [list(row) for row in rows if isinstance(row, list)],
    )

    if _question_expects_single_value(task.question):
        if len(sanitized_columns) == 1 and len(sanitized_rows) == 1:
            updated["retry_hint"] = (
                "You already have a grounded SQL result that looks like a single final value. "
                "Prefer answering now instead of exploring more."
            )
            updated["suggested_next_actions"] = ["answer"]
            updated["ready_to_answer_if_grounded"] = True
        return updated

    if len(sanitized_columns) == 1 and len(sanitized_rows) <= 20:
        updated["retry_hint"] = (
            "You already have a grounded SQL result with a compact final shape. "
            "Prefer answering now or doing one tiny final reshape."
        )
        updated["suggested_next_actions"] = ["answer", "execute_python"]
        updated["ready_to_answer_if_grounded"] = True
        return updated

    if _question_expects_listing(task.question) and len(sanitized_columns) <= 2 and len(sanitized_rows) <= 10:
        updated["retry_hint"] = (
            "You already have grounded rows in a compact result. Prefer answering now unless one tiny reshape is still needed."
        )
        updated["suggested_next_actions"] = ["answer", "execute_python"]
        updated["ready_to_answer_if_grounded"] = True
        return updated

    return updated


def _build_fallback_answer(task: PublicTask, state: AgentRuntimeState) -> AnswerTable | None:
    observation = _find_last_ready_sql_observation(state.steps)
    if not observation:
        return None
    content = observation.get("content")
    if not isinstance(content, dict):
        return None
    columns = content.get("columns")
    rows = content.get("rows")
    truncated = bool(content.get("truncated"))
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        return None
    if not isinstance(rows, list) or truncated or not rows:
        return None
    if not all(isinstance(row, list) for row in rows):
        return None

    if _question_expects_single_value(task.question):
        if len(columns) == 1 and len(rows) == 1:
            return AnswerTable(columns=list(columns), rows=[list(rows[0])])
        return None

    if len(columns) == 1 and len(rows) <= 10 and _question_expects_listing(task.question):
        return AnswerTable(columns=list(columns), rows=[list(row) for row in rows])

    return None


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_initial_context_steps(self, task: PublicTask) -> list[StepRecord]:
        del task
        return []

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        next_step_index: int,
    ) -> list[ModelMessage]:
        policy = _build_policy_state(
            state,
            task=task,
            next_step_index=next_step_index,
            max_steps=self.config.max_steps,
        )
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            if step.raw_response:
                messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        control_message = _build_control_message(task, policy)
        if control_message:
            messages.append(ModelMessage(role="user", content=control_message))
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        state.steps.extend(self._build_initial_context_steps(task))
        for step_index in range(1, self.config.max_steps + 1):
            fallback_answer = None
            if self.config.max_steps - step_index < 1:
                fallback_answer = _build_fallback_answer(task, state)
            if fallback_answer is not None:
                state.answer = fallback_answer
                break

            raw_response = self.model.complete(
                self._build_messages(task, state, next_step_index=step_index)
            )
            previous_observation = state.steps[-1].observation if state.steps else None
            try:
                model_step = parse_model_step(raw_response)
            except Exception as exc:
                observation = _build_parse_error_observation(exc)
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                continue

            try:
                if model_step.action == "answer":
                    model_step = ModelStep(
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=_sanitize_answer_action_input(task, model_step.action_input),
                        raw_response=model_step.raw_response,
                    )
                tool_result = self.tools.execute(
                    task,
                    model_step.action,
                    model_step.action_input,
                    model=self.model,
                )
            except Exception as exc:
                observation = _build_tool_error_observation(exc, model_step.action)
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                continue

            observation = {
                "ok": tool_result.ok,
                "tool": model_step.action,
                "content": tool_result.content,
            }
            observation = _slim_success_observation(model_step.action, observation)
            observation = _augment_pipeline_observation_hint(model_step.action, observation)
            observation = _augment_text_rule_observation(task, model_step.action, observation)
            if model_step.action == "execute_context_sql":
                observation = _augment_selected_sql_execution_observation(previous_observation, observation)
                observation = _augment_general_sql_execution_observation(task, observation)
            step_record = StepRecord(
                step_index=step_index,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=raw_response,
                observation=observation,
                ok=tool_result.ok,
            )
            state.steps.append(step_record)
            if tool_result.is_terminal:
                state.answer = tool_result.answer
                break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
