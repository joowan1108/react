from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.tools.sql_generate import generate_sql_candidates_with_model
from data_agent_baseline.tools.sql_linking import build_schema_link_context
from data_agent_baseline.tools.sql_revise import revise_sql_candidates_with_model
from data_agent_baseline.tools.sql_verify import verify_sql_candidates
from data_agent_baseline.tools.sqlite import inspect_sqlite_schema


def _find_best_result(verification_result: dict[str, object]) -> dict[str, object] | None:
    results = verification_result.get("results")
    best_index = verification_result.get("best_candidate_index")
    if isinstance(results, list) and isinstance(best_index, int):
        for result in results:
            if isinstance(result, dict) and int(result.get("candidate_index", -1)) == best_index:
                return result
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0]
    return None


def run_sql_pipeline_with_model(
    model: ModelAdapter,
    *,
    path: Path,
    question: str,
    num_candidates: int = 3,
    revision_rounds: int = 1,
    verify_limit: int = 50,
    schema_link_context: dict[str, object] | None = None,
) -> dict[str, object]:
    schema_info = inspect_sqlite_schema(path)
    link_context = schema_link_context or build_schema_link_context(
        question=question,
        database_path=str(path),
        schema_info=schema_info,
    )

    generated = generate_sql_candidates_with_model(
        model,
        question=question,
        schema_info=schema_info,
        schema_link_context=link_context,
        num_candidates=num_candidates,
    )
    verification = verify_sql_candidates(
        path=path,
        candidates=generated["candidates"],
        question=question,
        schema_link_context=link_context,
        limit=verify_limit,
    )

    revisions: list[dict[str, object]] = []
    current_verification = verification

    for revision_index in range(revision_rounds):
        acceptability = str(current_verification.get("best_candidate_acceptability") or "")
        if acceptability == "accept":
            break

        best_result = _find_best_result(current_verification)
        if not isinstance(best_result, dict):
            break

        revised = revise_sql_candidates_with_model(
            model,
            question=question,
            schema_info=schema_info,
            verification_result=best_result,
            schema_link_context=link_context,
            num_candidates=max(1, min(num_candidates, 2)),
        )
        revised_verification = verify_sql_candidates(
            path=path,
            candidates=revised["candidates"],
            question=question,
            schema_link_context=link_context,
            limit=verify_limit,
        )
        revisions.append(
            {
                "revision_index": revision_index + 1,
                "revision_input": best_result,
                "revised_candidates": revised,
                "verification": revised_verification,
            }
        )
        current_verification = revised_verification

    best_result = _find_best_result(current_verification)
    selected_sql = str(current_verification.get("best_candidate_sql") or "")
    selected_acceptability = str(current_verification.get("best_candidate_acceptability") or "")

    recommended_next_action = "execute_context_sql" if selected_sql else "inspect_sqlite_schema"
    if selected_acceptability == "reject":
        recommended_next_action = "run_sql_pipeline"

    return {
        "path": str(path),
        "question": question,
        "schema_link_context": link_context,
        "generated_candidates": generated,
        "initial_verification": verification,
        "revisions": revisions,
        "final_verification": current_verification,
        "selected_sql": selected_sql,
        "selected_acceptability": selected_acceptability,
        "selected_candidate": best_result,
        "recommended_next_action": recommended_next_action,
    }
