"""RunPod serverless entrypoint. Dispatches jobs by task type.

Tasks:
  classify-image                 exterior/interior routing + preview cutout
  refine-full-resolution-alpha   full-res matte
  harmonize-exterior             composite onto background + AI harmonisation
  harmonize-interior             full-frame cabin edit through the glass
  diag                           environment self-check, returns plain strings

ERROR REPORTING
---------------
RunPod's job-done endpoint was returning 400 on failed jobs and the nested
`error` object was being dropped from the returned output, so failures came
back with no explanation at all.

Two changes work around that:

  1. Every exception is printed to stdout with a full traceback. print()
     output always reaches the endpoint Logs tab regardless of what happens
     to the returned payload.
  2. The returned error is flattened to top-level string fields
     (error_code / error_message / error_stage) rather than a nested dict.

Both are diagnostics, not behaviour changes - the task functions are
untouched.
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable

import runpod

from app.tasks.classify_image import run_classify_image
from app.tasks.harmonize_exterior import run_harmonize_exterior
from app.tasks.harmonize_interior import run_harmonize_interior
from app.tasks.refine_full_alpha import run_refine_full_alpha


def _log(message: str) -> None:
    """stdout, flushed. Survives whatever happens to the returned payload."""

    print(message, flush=True)


def run_diag(payload: dict[str, object]) -> dict[str, object]:
    """Report what this container actually has. All values are strings."""

    from app.settings import ASSETS_DIR, OPENAI_IMAGE_MODEL, get_imagemagick_command

    presets_file = ASSETS_DIR / "backgrounds" / "presets.json"
    magick = get_imagemagick_command()

    try:
        import cv2

        cv2_version = cv2.__version__
    except Exception as error:  # noqa: BLE001
        cv2_version = f"NOT AVAILABLE: {error}"

    generated = ASSETS_DIR / "backgrounds" / "generated"
    if generated.is_dir():
        background_count = str(len(list(generated.glob("*"))))
    else:
        background_count = "directory missing"

    info = {
        "tasks_registered": ", ".join(sorted(TASKS)),
        "assets_dir": str(ASSETS_DIR),
        "assets_dir_exists": str(ASSETS_DIR.is_dir()),
        "presets_file": str(presets_file),
        "presets_file_exists": str(presets_file.is_file()),
        "backgrounds_found": background_count,
        "imagemagick_command": magick,
        "imagemagick_on_path": str(shutil.which(magick) is not None or Path(magick).exists()),
        "opencv_version": cv2_version,
        "openai_key_set": str(bool(os.environ.get("OPENAI_API_KEY"))),
        "openai_image_model": OPENAI_IMAGE_MODEL,
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
    }

    _log("=== DIAG ===")
    for key, value in info.items():
        _log(f"  {key}: {value}")
    _log("=== END DIAG ===")

    return {
        "status": "completed",
        "job_id": str(payload.get("job_id", "diag")),
        "task": "diag",
        "trace_id": str(payload.get("trace_id", "diag")),
        "image_type": None,
        "artifacts": None,
        "runtime_ms": 0,
        **info,
    }


TASKS: dict[str, Callable[[dict[str, object]], dict[str, object]]] = {
    "classify-image": run_classify_image,
    "refine-full-resolution-alpha": run_refine_full_alpha,
    "harmonize-exterior": run_harmonize_exterior,
    "harmonize-interior": run_harmonize_interior,
    "diag": run_diag,
}


def handler(job: dict[str, object]) -> dict[str, object]:
    payload = job.get("input", {}) or {}
    task = payload.get("task")
    job_id = str(payload.get("job_id", "unknown"))
    trace_id = str(payload.get("trace_id", "unknown"))

    _log(f"--- job start: task={task!r} job_id={job_id}")

    run_task = TASKS.get(str(task)) if task else None

    if run_task is None:
        message = f"No handler registered for task: {task!r}. Registered: {', '.join(sorted(TASKS))}"
        _log(f"!!! {message}")
        return {
            "status": "failed",
            "job_id": job_id,
            "task": str(task) if task else "unknown",
            "trace_id": trace_id,
            "image_type": None,
            "artifacts": None,
            "runtime_ms": None,
            "error_code": "unsupported_task",
            "error_stage": "dispatch",
            "error_message": message,
            "error_retryable": "false",
        }

    try:
        result = run_task(payload)
        _log(f"--- job ok: task={task!r} runtime_ms={result.get('runtime_ms')}")
        return result
    except Exception as error:  # noqa: BLE001 - convert to a structured failure
        detail = traceback.format_exc()
        _log(f"!!! TASK FAILED: task={task!r} job_id={job_id}")
        _log(f"!!! {type(error).__name__}: {error}")
        _log(detail)
        return {
            "status": "failed",
            "job_id": job_id,
            "task": str(task),
            "trace_id": trace_id,
            "image_type": None,
            "artifacts": None,
            "runtime_ms": None,
            "error_code": "task_exception",
            "error_stage": str(task),
            "error_type": type(error).__name__,
            "error_message": str(error)[:900],
            "error_traceback": detail[-1800:],
            "error_retryable": "true",
        }


_log("handler.py loaded. Tasks: " + ", ".join(sorted(TASKS)))

runpod.serverless.start({"handler": handler})
