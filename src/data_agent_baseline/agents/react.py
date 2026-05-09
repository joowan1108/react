from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import read_doc_preview
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


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


def _question_expects_single_value(question: str) -> bool:
    lowered = question.casefold()
    return any(
        marker in lowered
        for marker in (
            "how many",
            "count",
            "number of",
            "average",
            "avg",
            "sum",
            "total",
            "maximum",
            "minimum",
            "highest",
            "lowest",
            "max",
            "min",
        )
    )


def _question_expects_grouping(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in f" {lowered} " for marker in (" for each ", " by ", " per ", " grouped by ", " each "))


def _question_expects_listing(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in ("list", "show", "which", "what are", "who are", "give me"))


def _build_final_shape_hint(question: str, content: dict[str, object]) -> dict[str, object] | None:
    columns = content.get("columns")
    row_count = content.get("row_count")
    if not isinstance(columns, list):
        return None

    column_count = len(columns)
    hints: list[str] = []
    expected_shape = None

    if _question_expects_single_value(question):
        expected_shape = "prefer 1 column and usually 1 row"
        if column_count > 1:
            hints.append("This looks wider than a single-value answer.")
    elif _question_expects_grouping(question):
        expected_shape = "prefer group key column(s) plus one metric column"
        if column_count > 3:
            hints.append("This may include extra columns beyond the grouping key and metric.")
    elif not _question_expects_listing(question):
        expected_shape = "prefer only the column(s) explicitly requested by the question"
        if column_count > 2:
            hints.append("This may include extra identifier or descriptive columns.")

    if expected_shape is None:
        return None

    result: dict[str, object] = {"expected_final_shape": expected_shape}
    if hints:
        result["final_shape_hint"] = " ".join(hints)
    if isinstance(row_count, int):
        result["observed_shape"] = {"column_count": column_count, "row_count": row_count}
    return result


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
        candidate_paths = ["knowledge.md"]
        initial_steps: list[StepRecord] = []

        for relative_path in candidate_paths:
            candidate = task.context_dir / relative_path
            if not candidate.exists() or not candidate.is_file():
                continue

            preview = read_doc_preview(task, relative_path, max_chars=4000)
            initial_steps.append(
                StepRecord(
                    step_index=0,
                    thought="",
                    action="__bootstrap_knowledge__",
                    action_input={"path": relative_path, "max_chars": 4000},
                    raw_response="",
                    observation={
                        "ok": True,
                        "tool": "read_doc",
                        "content": preview,
                        "auto_context": True,
                        "reason": "bootstrap_knowledge",
                    },
                    ok=True,
                )
            )
            break

        return initial_steps

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
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
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        state.steps.extend(self._build_initial_context_steps(task))
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state))
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
                content = observation.get("content")
                if isinstance(content, dict):
                    final_shape_hint = _build_final_shape_hint(task.question, content)
                    if isinstance(final_shape_hint, dict):
                        observation = {**observation, **final_shape_hint}
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
