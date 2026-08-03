"""Worker task: full-resolution alpha refinement.

Mirrors the backend's PreprocessingWorker._refine_full_resolution_alpha for
the refine-full-resolution-alpha job type, adapted to be stateless. Fetches
the original by R2 key, re-segments at full resolution, merges the result
into the existing analysis JSON (fetched by key rather than read from a local
store), and writes the matte/cutout/analysis back to R2 under the same
filenames the local path uses (full-matte-birefnet-v1.png,
full-cutout-birefnet-v1.png, vehicle-v2.json).
"""

from __future__ import annotations

import json
import tempfile
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.pipeline.background_removal import remove_background
from app.settings import PREPROCESSING_SEGMENTATION_MODEL
from app.storage.r2_client import R2Client, private_object_key


def run_refine_full_alpha(payload: dict[str, object]) -> dict[str, object]:
    started = time.time()
    job_id = str(payload["job_id"])
    trace_id = str(payload["trace_id"])
    image_id = str(payload["image_id"])
    session_id = str(payload["session_id"])
    owner_id = str(payload["owner_id"])
    segmentation_model = str(
        payload.get("segmentation_model", PREPROCESSING_SEGMENTATION_MODEL)
    )
    r2_input = payload["r2"]
    original_key = str(r2_input["original_key"])
    existing_analysis_key = r2_input.get("analysis_key")

    storage = R2Client()
    original_bytes = storage.get_bytes(original_key)

    with tempfile.TemporaryDirectory(prefix="vehicle-full-segmentation-") as temp_dir:
        temp_path = Path(temp_dir)
        original_path = temp_path / "original"
        full_cutout_path = temp_path / "full-cutout.png"
        original_path.write_bytes(original_bytes)
        remove_background(original_path, full_cutout_path, model_name=segmentation_model)

        cutout_bytes = full_cutout_path.read_bytes()
        with Image.open(full_cutout_path) as full_cutout:
            if "A" not in full_cutout.getbands():
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "task": "refine-full-resolution-alpha",
                    "trace_id": trace_id,
                    "image_type": None,
                    "artifacts": None,
                    "runtime_ms": _elapsed_ms(started),
                    "error": {
                        "code": "preprocessing_failed",
                        "stage": "refine-full-resolution-alpha",
                        "message": "Full-resolution cutout has no alpha channel.",
                        "retryable": True,
                    },
                }
            matte = full_cutout.getchannel("A")
            matte_buffer = BytesIO()
            matte.save(matte_buffer, format="PNG", optimize=True)
            matte_bytes = matte_buffer.getvalue()
            bbox = list(matte.getbbox() or (0, 0, 0, 0))
            refinement = {
                "schema_version": 1,
                "method": "direct-birefnet-segmentation-v1",
                "model": segmentation_model,
                "source_size": {"width": full_cutout.width, "height": full_cutout.height},
            }

    analysis_payload: dict[str, object] = {"schema_version": 1}
    if existing_analysis_key:
        try:
            analysis_payload = json.loads(
                storage.get_bytes(str(existing_analysis_key)).decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            analysis_payload = {"schema_version": 1}
    analysis_payload["full_resolution_matte"] = {**refinement, "bbox": bbox}
    analysis_bytes = json.dumps(analysis_payload, indent=2, sort_keys=True).encode("utf-8")

    matte_key = private_object_key(
        "mattes", owner_id, session_id, image_id, "full-matte-birefnet-v1.png"
    )
    cutout_key = private_object_key(
        "mattes", owner_id, session_id, image_id, "full-cutout-birefnet-v1.png"
    )
    analysis_key = private_object_key(
        "analysis", owner_id, session_id, image_id, "vehicle-v2.json"
    )

    matte_response = storage.put_bytes(matte_key, matte_bytes, "image/png")
    cutout_response = storage.put_bytes(cutout_key, cutout_bytes, "image/png")
    analysis_response = storage.put_bytes(analysis_key, analysis_bytes, "application/json")

    return {
        "status": "completed",
        "job_id": job_id,
        "task": "refine-full-resolution-alpha",
        "trace_id": trace_id,
        "image_type": None,
        "artifacts": {
            "full_matte": {
                "object_key": matte_key,
                "content_type": "image/png",
                "size_bytes": len(matte_bytes),
                "etag": matte_response.get("ETag"),
            },
            "full_cutout": {
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
