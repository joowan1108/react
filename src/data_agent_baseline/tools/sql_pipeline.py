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


def _acceptability_rank(acceptability: str) -> int:
    if acceptability == "accept":
        return 2
    if acceptability == "weak_accept":
        return 1
    return 0


def _should_run_revision(verification_result: dict[str, object]) -> bool:
    best_result = _find_best_result(verification_result)
    if not isinstance(best_result, dict):
        return False

    if not bool(best_result.get("valid")):
        return True
    if int(best_result.get("row_count") or 0) == 0:
        return True
    if str(best_result.get("acceptability") or "") == "reject":
        return True
    return float(best_result.get("score") or 0.0) < 0.72


def _select_best_result_across_rounds(
    verification_rounds: list[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    best_result: dict[str, object] | None = None
    best_verification: dict[str, object] | None = None

    for verification in verification_rounds:
        candidate = _find_best_result(verification)
        if not isinstance(candidate, dict):
            continue
        if best_result is None:
            best_result = candidate
            best_verification = verification
            continue

        current_key = (
            float(candidate.get("score") or 0.0),
            _acceptability_rank(str(candidate.get("acceptability") or "")),
            bool(candidate.get("valid")),
            -int(candidate.get("row_count") or 0),
        )
        best_key = (
            float(best_result.get("score") or 0.0),
            _acceptability_rank(str(best_result.get("acceptability") or "")),
            bool(best_result.get("valid")),
            -int(best_result.get("row_count") or 0),
        )
        if current_key > best_key:
            best_result = candidate
            best_verification = verification

    return best_result, best_verification


def run_sql_pipeline_with_model(
    model: ModelAdapter,
    *,
    path: Path,
    question: str,
    num_candidates: int = 3,
    revision_rounds: int = 1,
    verify_limit: int = 50,
    schema_link_context: dict[str, object] | None = None,
    knowledge_text: str | None = None,
) -> dict[str, object]:
    num_candidates = max(1, min(num_candidates, 2))
    schema_info = inspect_sqlite_schema(path)
    link_context = schema_link_context or build_schema_link_context(
        question=question,
        database_path=str(path),
        schema_info=schema_info,
        knowledge_text=knowledge_text,
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
        if not _should_run_revision(current_verification):
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

    verification_rounds = [verification] + [
        item["verification"]
        for item in revisions
        if isinstance(item.get("verification"), dict)
    ]
    best_result, selected_verification = _select_best_result_across_rounds(verification_rounds)
    selected_sql = str(best_result.get("sql") or "") if isinstance(best_result, dict) else ""
    selected_acceptability = (
        str(best_result.get("acceptability") or "")
        if isinstance(best_result, dict)
        else ""
    )

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
        "selected_verification": selected_verification,
        "selected_sql": selected_sql,
        "selected_acceptability": selected_acceptability,
        "selected_candidate": best_result,
        "recommended_next_action": recommended_next_action,
    }
