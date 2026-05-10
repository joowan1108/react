from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class AgentPolicyState:
    next_step_index: int
    remaining_steps: int
    near_limit: bool
    has_selected_sql_waiting: bool
    has_ready_sql_result: bool
    repeated_same_action_count: int
    recent_sql_error_count: int
    list_context_count: int
    low_signal_count: int


@dataclass(frozen=True, slots=True)
class ActionBlock:
    retry_hint: str
    suggested_next_actions: list[str]


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
    next_step_index: int,
    max_steps: int,
) -> AgentPolicyState:
    remaining_steps = max(0, max_steps - next_step_index + 1)
    pending_selected_sql = _find_last_pending_selected_sql_observation(state.steps)
    ready_sql_observation = _find_last_ready_sql_observation(state.steps)
    return AgentPolicyState(
        next_step_index=next_step_index,
        remaining_steps=remaining_steps,
        near_limit=remaining_steps <= 3,
        has_selected_sql_waiting=pending_selected_sql is not None,
        has_ready_sql_result=ready_sql_observation is not None,
        repeated_same_action_count=0,
        recent_sql_error_count=_count_recent_sql_errors(state.steps),
        list_context_count=sum(1 for step in state.steps if step.action == "list_context"),
        low_signal_count=sum(1 for step in state.steps if step.action in LOW_SIGNAL_ACTIONS),
    )


def _build_control_message(task: PublicTask, policy: AgentPolicyState) -> str | None:
    hints: list[str] = []
    difficulty = task.difficulty.casefold().strip()
    if difficulty == "easy":
        hints.append("For easy tasks, prefer the shortest grounded path and avoid extra exploration.")
    if policy.has_selected_sql_waiting:
        hints.append("A selected SQL candidate already exists. Execute it next before starting new exploration.")
    if policy.recent_sql_error_count >= 2:
        hints.append("Recent direct SQL attempts failed. Prefer schema inspection or run_sql_pipeline instead of more fresh SQL guesses.")
    if policy.near_limit and policy.has_ready_sql_result:
        hints.append("You are near the step limit and already have grounded rows. Prefer final shaping or answer now.")
    elif policy.near_limit:
        hints.append("You are near the step limit. Avoid low-signal exploration and move toward the smallest grounded answer.")
    if policy.list_context_count >= 2:
        hints.append("Do not call list_context again unless a missing-path error forces it.")
    if not hints:
        return None
    return "Control hints:\n- " + "\n- ".join(hints)


def _build_action_block_observation(action: str, block: ActionBlock) -> dict[str, object]:
    return {
        "ok": False,
        "tool": action,
        "error_type": "policy_block",
        "error": block.retry_hint,
        "retry_hint": block.retry_hint,
        "suggested_next_actions": block.suggested_next_actions,
    }


def _maybe_block_action(
    task: PublicTask,
    state: AgentRuntimeState,
    policy: AgentPolicyState,
    model_step: ModelStep,
) -> ActionBlock | None:
    repeated_count = _count_recent_same_action(state.steps, model_step.action, model_step.action_input)
    policy = AgentPolicyState(
        next_step_index=policy.next_step_index,
        remaining_steps=policy.remaining_steps,
        near_limit=policy.near_limit,
        has_selected_sql_waiting=policy.has_selected_sql_waiting,
        has_ready_sql_result=policy.has_ready_sql_result,
        repeated_same_action_count=repeated_count,
        recent_sql_error_count=policy.recent_sql_error_count,
        list_context_count=policy.list_context_count,
        low_signal_count=policy.low_signal_count,
    )

    if repeated_count >= 2:
        return ActionBlock(
            retry_hint="Do not repeat the same action with the same input again. Use a different grounded next step.",
            suggested_next_actions=["inspect_sqlite_schema", "run_sql_pipeline", "answer"],
        )

    if model_step.action == "list_context" and policy.list_context_count >= 2:
        return ActionBlock(
            retry_hint="The context has already been listed enough times. Reuse observed paths and continue with grounded tools.",
            suggested_next_actions=["scan", "inspect_sqlite_schema", "read_doc"],
        )

    if policy.has_selected_sql_waiting and model_step.action not in {"execute_context_sql", "answer"}:
        return ActionBlock(
            retry_hint="A selected SQL candidate is waiting. Execute that SQL next instead of starting a new planning or exploration step.",
            suggested_next_actions=["execute_context_sql"],
        )

    if policy.recent_sql_error_count >= 2 and model_step.action == "execute_context_sql":
        return ActionBlock(
            retry_hint="Recent direct SQL attempts already failed multiple times. Prefer run_sql_pipeline or schema inspection instead of another fresh SQL guess.",
            suggested_next_actions=["run_sql_pipeline", "inspect_sqlite_schema"],
        )

    if policy.near_limit and model_step.action in LOW_SIGNAL_ACTIONS:
        next_actions = ["answer", "execute_context_sql", "execute_python"] if policy.has_ready_sql_result else ["execute_context_sql", "answer"]
        return ActionBlock(
            retry_hint="You are near the step limit. Avoid low-signal exploration now and move directly toward a grounded answer.",
            suggested_next_actions=next_actions,
        )

    if (
        task.difficulty.casefold().strip() == "easy"
        and model_step.action == "run_sql_pipeline"
        and policy.recent_sql_error_count == 0
        and policy.next_step_index <= 4
    ):
        return ActionBlock(
            retry_hint="For easy tasks, try the lighter grounded path first: inspect schema and use direct SQL before escalating to run_sql_pipeline.",
            suggested_next_actions=["inspect_sqlite_schema", "execute_context_sql"],
        )

    return None


def _question_expects_single_value(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in SINGLE_VALUE_HINTS)


def _question_expects_listing(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in LISTING_HINTS)


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

    if len(columns) == 1 and len(rows) <= 20:
        return AnswerTable(columns=list(columns), rows=[list(row) for row in rows])

    if _question_expects_listing(task.question) and len(columns) <= 2 and len(rows) <= 10:
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

            policy = _build_policy_state(
                state,
                next_step_index=step_index,
                max_steps=self.config.max_steps,
            )
            blocked = _maybe_block_action(task, state, policy, model_step)
            if blocked is not None:
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=_build_action_block_observation(model_step.action, blocked),
                        ok=False,
                    )
                )
                continue

            try:
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
            if model_step.action == "execute_context_sql":
                observation = _augment_selected_sql_execution_observation(previous_observation, observation)
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
