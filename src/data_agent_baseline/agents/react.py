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


def _extract_recovery_signature(observation: dict[str, object]) -> tuple[str | None, str | None, str | None]:
    if observation.get("error_type") != "tool_error":
        return None, None, None

    tool = str(observation.get("tool") or "")
    error_text = str(observation.get("error") or "")
    normalized_error = error_text.casefold()

    error_kind = None
    if "no such table" in normalized_error:
        error_kind = "no_such_table"
    elif "no such column" in normalized_error:
        error_kind = "no_such_column"
    elif "missing sqlite asset" in normalized_error:
        error_kind = "missing_sqlite_asset"
    elif "path escapes context dir" in normalized_error:
        error_kind = "path_escape"
    elif "missing context asset" in normalized_error:
        error_kind = "missing_context_asset"

    db_path = None
    try:
        payload = json.loads(error_text)
        if isinstance(payload, dict):
            raw_path = payload.get("path")
            if raw_path is not None:
                db_path = str(raw_path)
            if error_kind is None:
                payload_error = str(payload.get("error") or "").casefold()
                if "no such table" in payload_error:
                    error_kind = "no_such_table"
                elif "no such column" in payload_error:
                    error_kind = "no_such_column"
    except Exception:
        pass

    return tool or None, db_path, error_kind


def _augment_repeated_tool_error_hint(state: AgentRuntimeState, observation: dict[str, object]) -> dict[str, object]:
    tool, db_path, error_kind = _extract_recovery_signature(observation)
    if tool is None or error_kind is None:
        return observation

    repeat_count = 1
    for previous_step in reversed(state.steps[-4:]):
        prev_obs = previous_step.observation
        if not isinstance(prev_obs, dict):
            continue
        prev_tool, prev_db_path, prev_error_kind = _extract_recovery_signature(prev_obs)
        if prev_tool == tool and prev_error_kind == error_kind and prev_db_path == db_path:
            repeat_count += 1

    if repeat_count < 2:
        return observation

    updated = dict(observation)
    updated["repeat_error"] = True
    updated["repeat_count"] = repeat_count

    if tool == "execute_context_sql" and error_kind in {"no_such_table", "no_such_column"}:
        updated["retry_hint"] = (
            "Stop repeating the same SQL guess. Prefer the higher-level SQL pipeline before making another direct SQL attempt."
        )
        updated["suggested_next_actions"] = [
            "inspect_sqlite_schema",
            "run_sql_pipeline",
        ]
    elif tool == "execute_context_sql" and error_kind in {"missing_sqlite_asset", "path_escape"}:
        updated["retry_hint"] = (
            "Stop reusing the invalid database path. Choose a path that was previously observed "
            "from scan output, inspect_sqlite_schema output, or list_context."
        )
        updated["suggested_next_actions"] = ["list_context", "inspect_sqlite_schema"]
    elif error_kind == "missing_context_asset":
        updated["retry_hint"] = (
            "Stop guessing file paths. Use list_context and then copy an exact observed path."
        )
        updated["suggested_next_actions"] = ["list_context"]

    return updated


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


def _extract_selected_sql_from_pipeline_observation(
    observation: dict[str, object],
) -> tuple[str | None, str | None]:
    if not observation.get("ok"):
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


def _build_pipeline_execution_guard_observation(
    *,
    expected_path: str | None,
    expected_sql: str,
) -> dict[str, object]:
    content: dict[str, object] = {
        "ok": False,
        "error_type": "pipeline_execution_guard",
        "retry_hint": (
            "The previous SQL pipeline already selected a SQL candidate. "
            "Execute that exact SQL next instead of starting a new planning step or writing a fresh SQL guess."
        ),
        "suggested_next_actions": ["execute_context_sql"],
        "selected_sql": expected_sql,
    }
    if expected_path is not None:
        content["selected_path"] = expected_path
    return content


def _build_repeated_list_context_guard_observation() -> dict[str, object]:
    return {
        "ok": False,
        "error_type": "repeated_list_context_guard",
        "error": "Repeated list_context calls are blocked because the available paths were already shown.",
        "retry_hint": (
            "Do not call list_context again right now. Reuse the previously observed paths and move to a more informative next step "
            "such as scan, inspect_sqlite_schema, read_doc, read_csv, read_json, retrieve, or run_sql_pipeline."
        ),
        "suggested_next_actions": [
            "scan",
            "inspect_sqlite_schema",
            "read_doc",
            "read_csv",
            "read_json",
            "retrieve",
            "run_sql_pipeline",
        ],
    }


def _build_execute_python_formatting_guard_observation() -> dict[str, object]:
    return {
        "ok": False,
        "error_type": "execute_python_formatting_guard",
        "error": "execute_python is reserved for final result formatting after the relevant data has already been found.",
        "retry_hint": (
            "Do not use execute_python for open-ended exploration or SQL planning. "
            "Use scan, inspect_sqlite_schema, read_doc, retrieve, run_sql_pipeline, or execute_context_sql first. "
            "Only use execute_python after you already have grounded result rows and only final reshaping remains."
        ),
        "suggested_next_actions": [
            "scan",
            "inspect_sqlite_schema",
            "read_doc",
            "retrieve",
            "run_sql_pipeline",
            "execute_context_sql",
        ],
    }


def _build_repeated_retrieve_guard_observation(*, low_signal: bool) -> dict[str, object]:
    retry_hint = (
        "Repeated retrieve calls are blocked because they are no longer adding useful evidence. "
        "Reuse the best retrieved snippets you already have and move to read_doc, run_sql_pipeline, execute_context_sql, or answer."
    )
    if low_signal:
        retry_hint = (
            "Repeated low-signal retrieve calls are blocked. The current retrieval results are weak, so change strategy instead of searching markdown again."
        )
    return {
        "ok": False,
        "error_type": "repeated_retrieve_guard",
        "error": "Repeated retrieve calls are blocked.",
        "retry_hint": retry_hint,
        "suggested_next_actions": [
            "read_doc",
            "run_sql_pipeline",
            "execute_context_sql",
            "answer",
        ],
    }


def _build_answer_convergence_guard_observation() -> dict[str, object]:
    return {
        "ok": False,
        "error_type": "answer_convergence_guard",
        "error": "A grounded final result is already available and the agent must converge to answer.",
        "retry_hint": (
            "Do not restart exploration. You already have a grounded result from the selected SQL. "
            "Call answer next, or use execute_python only for one final compact reshaping step immediately before answer."
        ),
        "suggested_next_actions": ["answer", "execute_python"],
    }


def _is_final_formatting_ready(state: AgentRuntimeState) -> bool:
    recent_steps = [
        step
        for step in state.steps[-6:]
        if step.action and not step.action.startswith("__")
    ]
    if not recent_steps:
        return False

    successful_answer_like_source = False
    successful_python_count = 0

    for step in recent_steps:
        if step.action == "execute_python" and step.ok:
            successful_python_count += 1

        observation = step.observation if isinstance(step.observation, dict) else {}
        content = observation.get("content")
        if not step.ok or not isinstance(content, dict):
            continue

        if step.action == "execute_context_sql":
            row_count = content.get("row_count")
            if isinstance(row_count, int) and row_count >= 0:
                successful_answer_like_source = True
        elif step.action in {"read_csv", "read_json", "read_doc", "retrieve", "run_sql_pipeline"}:
            successful_answer_like_source = True

    if successful_python_count >= 1:
        return False

    return successful_answer_like_source


def _must_converge_to_answer(state: AgentRuntimeState) -> bool:
    recent_steps = state.steps[-3:]
    if not recent_steps:
        return False

    selected_sql_ready = False
    successful_python_after_ready = 0

    for step in recent_steps:
        obs = step.observation if isinstance(step.observation, dict) else {}
        if bool(obs.get("ready_to_answer_if_grounded")):
            selected_sql_ready = True
        if selected_sql_ready and step.action == "execute_python" and step.ok:
            successful_python_after_ready += 1

    if not selected_sql_ready:
        return False

    return successful_python_after_ready <= 1


def _should_block_retrieve(state: AgentRuntimeState, action_input: dict[str, object]) -> tuple[bool, bool]:
    recent_steps = [
        step
        for step in state.steps[-5:]
        if step.action and not step.action.startswith("__")
    ]
    if not recent_steps:
        return False, False

    current_query = str(action_input.get("query") or "").strip().casefold()
    current_mode = str(action_input.get("mode") or "entity").strip().casefold()
    current_sources = action_input.get("sources")
    normalized_sources = (
        tuple(str(item) for item in current_sources)
        if isinstance(current_sources, list)
        else None
    )

    repeated_same_retrieve = 0
    repeated_low_signal = 0
    for step in recent_steps:
        if step.action != "retrieve":
            continue
        previous_query = str(step.action_input.get("query") or "").strip().casefold()
        previous_mode = str(step.action_input.get("mode") or "entity").strip().casefold()
        previous_sources = step.action_input.get("sources")
        normalized_previous_sources = (
            tuple(str(item) for item in previous_sources)
            if isinstance(previous_sources, list)
            else None
        )

        if (
            previous_query == current_query
            and previous_mode == current_mode
            and normalized_previous_sources == normalized_sources
        ):
            repeated_same_retrieve += 1
            obs = step.observation if isinstance(step.observation, dict) else {}
            content = obs.get("content")
            if isinstance(content, dict) and bool(content.get("low_signal")):
                repeated_low_signal += 1

    if repeated_same_retrieve >= 2:
        return True, repeated_low_signal >= 1
    return False, False


def _augment_selected_sql_execution_observation(
    previous_observation: dict[str, object] | None,
    observation: dict[str, object],
) -> dict[str, object]:
    if not previous_observation:
        return observation

    expected_path, expected_sql = _extract_selected_sql_from_pipeline_observation(previous_observation)
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

            if model_step.action == "list_context":
                recent_actions = [
                    step.action
                    for step in state.steps[-3:]
                    if step.action and not step.action.startswith("__")
                ]
                list_context_count = sum(1 for action in recent_actions if action == "list_context")
                if list_context_count >= 2:
                    observation = _build_repeated_list_context_guard_observation()
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

            if model_step.action == "execute_python" and not _is_final_formatting_ready(state):
                observation = _build_execute_python_formatting_guard_observation()
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

            if model_step.action == "retrieve":
                should_block_retrieve, low_signal = _should_block_retrieve(state, model_step.action_input)
                if should_block_retrieve:
                    observation = _build_repeated_retrieve_guard_observation(low_signal=low_signal)
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

            if _must_converge_to_answer(state) and model_step.action not in {"answer", "execute_python"}:
                observation = _build_answer_convergence_guard_observation()
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

            if state.steps:
                expected_path, expected_sql = _extract_selected_sql_from_pipeline_observation(previous_observation)
                if expected_sql is not None:
                    selected_path = expected_path
                    if model_step.action != "execute_context_sql":
                        observation = _build_pipeline_execution_guard_observation(
                            expected_path=selected_path,
                            expected_sql=expected_sql,
                        )
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

                    submitted_sql = model_step.action_input.get("sql")
                    submitted_path = model_step.action_input.get("path")
                    normalized_submitted_sql = str(submitted_sql).strip() if submitted_sql is not None else ""
                    normalized_expected_sql = expected_sql.strip()
                    normalized_submitted_path = (
                        str(submitted_path).strip() if submitted_path is not None else ""
                    )
                    normalized_expected_path = selected_path.strip() if selected_path is not None else ""

                    if normalized_submitted_sql != normalized_expected_sql or (
                        normalized_expected_path and normalized_submitted_path != normalized_expected_path
                    ):
                        observation = _build_pipeline_execution_guard_observation(
                            expected_path=selected_path,
                            expected_sql=expected_sql,
                        )
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

            try:
                tool_result = self.tools.execute(
                    task,
                    model_step.action,
                    model_step.action_input,
                    model=self.model,
                )
            except Exception as exc:
                observation = _build_tool_error_observation(exc, model_step.action)
                observation = _augment_repeated_tool_error_hint(state, observation)
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
