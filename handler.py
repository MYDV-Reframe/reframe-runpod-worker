"""RunPod serverless entrypoint. Dispatches jobs by task type.

Tasks:
  classify-image                 exterior/interior routing + preview cutout
  refine-full-resolution-alpha   full-res matte
  harmonize-exterior             composite onto background + AI harmonisation
  harmonize-interior             full-frame cabin edit through the glass

The two harmonize tasks are the stage that makes automotive glass transparent.
Segmentation only cuts a silhouette; the glass is handled here.
"""

from __future__ import annotations

import logging
from typing import Callable

import runpod

from app.tasks.classify_image import run_classify_image
from app.tasks.harmonize_exterior import run_harmonize_exterior
from app.tasks.harmonize_interior import run_harmonize_interior
from app.tasks.refine_full_alpha import run_refine_full_alpha

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASKS: dict[str, Callable[[dict[str, object]], dict[str, object]]] = {
    "classify-image": run_classify_image,
    "refine-full-resolution-alpha": run_refine_full_alpha,
    "harmonize-exterior": run_harmonize_exterior,
    "harmonize-interior": run_harmonize_interior,
}


def handler(job: dict[str, object]) -> dict[str, object]:
    payload = job.get("input", {}) or {}
    task = payload.get("task")
    run_task = TASKS.get(str(task)) if task else None

    if run_task is None:
        return {
            "status": "failed",
            "job_id": payload.get("job_id", "unknown"),
            "task": task or "unknown",
            "trace_id": payload.get("trace_id", "unknown"),
            "image_type": None,
            "artifacts": None,
            "runtime_ms": None,
            "error": {
                "code": "unsupported_task",
                "stage": "dispatch",
                "message": f"No handler registered for task: {task!r}",
                "retryable": False,
            },
        }

    try:
        return run_task(payload)
    except Exception as error:  # noqa: BLE001 - convert to a structured failure payload
        logger.exception("Task failed: task=%s job_id=%s", task, payload.get("job_id"))
        return {
            "status": "failed",
            "job_id": payload.get("job_id", "unknown"),
            "task": task,
            "trace_id": payload.get("trace_id", "unknown"),
            "image_type": None,
            "artifacts": None,
            "runtime_ms": None,
            "error": {
                "code": "task_exception",
                "stage": str(task),
                "message": str(error)[:500],
                "retryable": True,
            },
        }


runpod.serverless.start({"handler": handler})
