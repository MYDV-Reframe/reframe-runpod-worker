"""Worker task: classification + preview-resolution segmentation.

Mirrors the backend's PreprocessingWorker._preprocess_image for the
classify-image job type, adapted to be stateless: instead of reading/writing
through a local SessionStore, it downloads its input by R2 key and returns a
structured result for the backend's callback route to apply.

Known Phase 1 simplification: the analysis JSON this produces omits the
"cutout" bounding-box/measurement sub-object that the backend's
processing/user_flow.py:analyze_cutout() normally adds. That module is part
of the render-export/harmonize-export pipeline, explicitly out of scope for
this phase, so it isn't ported here. This must be reconciled before
render-export migrates in a later phase - it does not affect classify-image
or refine-full-resolution-alpha correctness, and the local path (used
whenever RUNPOD_MIGRATION_ENABLED is false) is unaffected and still produces
the full analysis payload.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from app.pipeline.background_removal import remove_background
from app.pipeline.image_classification import classify_vehicle_image
from app.settings import PREPROCESSING_SEGMENTATION_MODEL
from app.storage.r2_client import R2Client, private_object_key


def run_classify_image(payload: dict[str, object]) -> dict[str, object]:
    started = time.time()
    job_id = str(payload["job_id"])
    trace_id = str(payload["trace_id"])
    image_id = str(payload["image_id"])
    session_id = str(payload["session_id"])
    owner_id = str(payload["owner_id"])
    processing_mode = str(payload.get("processing_mode", "auto"))
    segmentation_model = str(
        payload.get("segmentation_model", PREPROCESSING_SEGMENTATION_MODEL)
    )
    processing_copy_key = str(payload["r2"]["processing_copy_key"])

    storage = R2Client()
    source_bytes = storage.get_bytes(processing_copy_key)

    with tempfile.TemporaryDirectory(prefix="vehicle-preprocess-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input-1024.webp"
        input_path.write_bytes(source_bytes)

        classification = classify_vehicle_image(input_path, mode=processing_mode)

        if classification.image_type == "uncertain":
            return {
                "status": "failed",
                "job_id": job_id,
                "task": "classify-image",
                "trace_id": trace_id,
                "image_type": None,
                "artifacts": None,
                "runtime_ms": _elapsed_ms(started),
                "error": {
                    "code": "classification_uncertain",
                    "stage": "classify-image",
                    "message": "Image classification was uncertain.",
                    "retryable": True,
                },
            }

        if classification.image_type == "interior":
            return {
                "status": "completed",
                "job_id": job_id,
                "task": "classify-image",
                "trace_id": trace_id,
                "image_type": "interior",
                "artifacts": {},
                "runtime_ms": _elapsed_ms(started),
                "error": None,
            }

        cutout_path = temp_path / "preview-cutout-v1.png"
        remove_background(input_path, cutout_path, model_name=segmentation_model)
        cutout_bytes = cutout_path.read_bytes()

        analysis = {
            "schema_version": 1,
            "image_id": image_id,
            "source": "processing_copy",
            "classification": classification.to_dict(),
        }
        analysis_bytes = json.dumps(analysis, indent=2, sort_keys=True).encode("utf-8")

    cutout_key = private_object_key(
        "mattes", owner_id, session_id, image_id, "preview-cutout-v1.png"
    )
    analysis_key = private_object_key(
        "analysis", owner_id, session_id, image_id, "vehicle-v1.json"
    )

    cutout_response = storage.put_bytes(cutout_key, cutout_bytes, "image/png")
    analysis_response = storage.put_bytes(analysis_key, analysis_bytes, "application/json")

    return {
        "status": "completed",
        "job_id": job_id,
        "task": "classify-image",
        "trace_id": trace_id,
        "image_type": "exterior",
        "artifacts": {
            "preview_cutout": {
                "object_key": cutout_key,
                "content_type": "image/png",
                "size_bytes": len(cutout_bytes),
                "etag": cutout_response.get("ETag"),
            },
            "analysis": {
                "object_key": analysis_key,
                "content_type": "application/json",
                "size_bytes": len(analysis_bytes),
                "etag": analysis_response.get("ETag"),
            },
        },
        "runtime_ms": _elapsed_ms(started),
        "error": None,
    }


def _elapsed_ms(started: float) -> int:
    return round((time.time() - started) * 1000)
