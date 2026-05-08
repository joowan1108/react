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
    lowered = error_text.casefold()
    if "no such table" in lowered or "no such column" in lowered:
        observation["retry_hint"] = (
            "Inspect the database schema again and only use observed table and column names."
        )
    elif "missing sqlite asset" in lowered or "path escapes context dir" in lowered:
        observation["retry_hint"] = (
            "Reuse a valid observed database path or context-relative path instead of inventing one."
        )
    elif "missing context asset" in lowered:
        observation["retry_hint"] = (
            "Use an exact path returned by list_context instead of a glob or guessed path."
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
            "Stop repeating the same SQL guess. Inspect the schema for this database path again "
            "or switch to a different observed database path before writing more SQL."
        )
        updated["suggested_next_actions"] = ["inspect_sqlite_schema", "list_context"]
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
