from __future__ import annotations

import json

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.tools.sql_generate import _extract_json_payload, _render_schema


def build_revise_sql_candidate_messages(
    *,
    question: str,
    schema_info: dict[str, object],
    verification_result: dict[str, object],
    schema_link_context: dict[str, object] | None = None,
    num_candidates: int = 2,
) -> list[ModelMessage]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    schema_text = _render_schema(schema_info)
    verification_text = json.dumps(verification_result, ensure_ascii=False, indent=2)
    linking_text = (
        json.dumps(schema_link_context, ensure_ascii=False, indent=2)
        if schema_link_context
        else ""
    )

    instructions = [
        f"Revise the previous SQL attempts and generate up to {num_candidates} improved grounded read-only SQL candidates.",
        "Use the verification results to avoid repeated errors and poor result shapes.",
        "Use only observed tables and columns from the schema.",
        "Each candidate must target exactly one sqlite database.",
        "Return only a JSON array.",
        'Each array item must be an object with keys "sql" and "rationale".',
        "Do not include markdown fences or extra commentary.",
    ]

    parts = [
        "\n".join(instructions),
        f"Question:\n{question}",
        f"Observed schema:\n{schema_text}",
        f"Verification result to address:\n{verification_text}",
    ]
    if linking_text:
        parts.append(f"Schema linking context:\n{linking_text}")

    return [
        ModelMessage(
            role="system",
            content=(
                "You are a SQL revision helper for a ReAct-style data agent. "
                "Revise SQL candidates using verification feedback and return only JSON."
            ),
        ),
        ModelMessage(role="user", content="\n\n".join(parts)),
    ]


def revise_sql_candidates_with_model(
    model: ModelAdapter,
    *,
    question: str,
    schema_info: dict[str, object],
    verification_result: dict[str, object],
    schema_link_context: dict[str, object] | None = None,
    num_candidates: int = 2,
) -> dict[str, object]:
    messages = build_revise_sql_candidate_messages(
        question=question,
        schema_info=schema_info,
        verification_result=verification_result,
        schema_link_context=schema_link_context,
        num_candidates=num_candidates,
    )
    raw_response = model.complete(messages)
    payload = _extract_json_payload(raw_response)

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"revise_sql_candidates returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("revise_sql_candidates must return a non-empty JSON array.")

    candidates: list[dict[str, str]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise RuntimeError(f"revise_sql_candidates item {index} is not an object.")
        sql = str(item.get("sql") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        if not sql:
            raise RuntimeError(f"revise_sql_candidates item {index} is missing SQL text.")
        candidates.append({"sql": sql, "rationale": rationale})

    return {
        "question": question,
        "candidate_count": len(candidates),
        "candidates": candidates[:num_candidates],
        "raw_response": raw_response,
    }
