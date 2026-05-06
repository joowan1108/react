from __future__ import annotations

import csv
import json
import os
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig, AgentConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import _run_single_task_with_timeout

INPUT_ROOT = Path("/input")
OUTPUT_ROOT = Path("/output")
LOG_ROOT = Path("/logs")
TMP_RUN_ROOT = Path("/tmp/data-agent-submission")


def _normalize_api_base(raw_value: str) -> str:
    normalized = raw_value.strip().rstrip("/")
    if not normalized:
        raise RuntimeError("MODEL_API_URL must not be empty.")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _load_submission_config() -> AppConfig:
    model_api_url = os.environ.get("MODEL_API_URL", "").strip()
    if not model_api_url:
        raise RuntimeError("Missing required environment variable: MODEL_API_URL")

    model_name = os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b").strip()
    if not model_name:
        raise RuntimeError("MODEL_NAME must not be empty.")

    api_key = os.environ.get("MODEL_API_KEY", "EMPTY")
    max_steps = int(os.environ.get("AGENT_MAX_STEPS", "16"))
    temperature = float(os.environ.get("MODEL_TEMPERATURE", "0.0"))
    task_timeout_seconds = int(os.environ.get("TASK_TIMEOUT_SECONDS", "600"))

    return AppConfig(
        dataset=DatasetConfig(root_path=INPUT_ROOT),
        agent=AgentConfig(
            model=model_name,
            api_base=_normalize_api_base(model_api_url),
            api_key=api_key,
            max_steps=max_steps,
            temperature=temperature,
        ),
        run=RunConfig(
            output_dir=TMP_RUN_ROOT,
            run_id="submission",
            max_workers=1,
            task_timeout_seconds=task_timeout_seconds,
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_prediction_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _process_task(task_id: str, config: AppConfig) -> dict[str, Any]:
    started_at = perf_counter()
    try:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    except BaseException as exc:  # noqa: BLE001
        run_result = _failure_payload(task_id, f"Task failed with uncaught error: {exc}")
        run_result["exception"] = "".join(traceback.format_exception(exc))
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return run_result


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    config = _load_submission_config()
    dataset = DABenchPublicDataset(INPUT_ROOT)
    if not dataset.exists:
        raise RuntimeError(f"Input root does not exist: {INPUT_ROOT}")

    task_summaries: list[dict[str, Any]] = []

    for task in dataset.iter_tasks():
        run_result = _process_task(task.task_id, config)
        _write_json(LOG_ROOT / task.task_id / "trace.json", run_result)

        answer = run_result.get("answer")
        prediction_path: str | None = None
        if isinstance(answer, dict):
            prediction_csv = OUTPUT_ROOT / task.task_id / "prediction.csv"
            _write_prediction_csv(
                prediction_csv,
                [str(column) for column in answer.get("columns", [])],
                [list(row) for row in answer.get("rows", [])],
            )
            prediction_path = str(prediction_csv)

        task_summaries.append(
            {
                "task_id": task.task_id,
                "succeeded": bool(run_result.get("succeeded")),
                "failure_reason": run_result.get("failure_reason"),
                "prediction_csv_path": prediction_path,
                "trace_path": str(LOG_ROOT / task.task_id / "trace.json"),
                "elapsed_seconds": run_result.get("e2e_elapsed_seconds"),
            }
        )

    _write_json(
        LOG_ROOT / "summary.json",
        {
            "task_count": len(task_summaries),
            "succeeded_task_count": sum(1 for item in task_summaries if item["succeeded"]),
            "failed_task_count": sum(1 for item in task_summaries if not item["succeeded"]),
            "tasks": task_summaries,
        },
    )


if __name__ == "__main__":
    main()
