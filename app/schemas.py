"""Schemas mirroring the backend's callback contract field-for-field.

The worker has no import path to the backend's session_schemas.py (separate
deployable, separate repo), so ProcessingErrorV1 is duplicated here rather
than shared. Field names must stay identical to
backend/app/api/runpod_schemas.py's RunPodTaskResultV1/RunPodArtifactV1/
ProcessingErrorV1 so the callback route can parse this output with no
translation layer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ProcessingErrorV1(BaseModel):
    code: str
    stage: str
    message: str
    retryable: bool


class ArtifactResultV1(BaseModel):
    object_key: str
    content_type: str
    size_bytes: int
    etag: str | None = None


class TaskResultV1(BaseModel):
    status: Literal["completed", "failed"]
    job_id: str
    task: Literal["classify-image", "refine-full-resolution-alpha"]
    trace_id: str
    image_type: Literal["exterior", "interior"] | None = None
    artifacts: dict[str, ArtifactResultV1] | None = None
    runtime_ms: int | None = None
    error: ProcessingErrorV1 | None = None

    def to_output(self) -> dict[str, object]:
        """RunPod expects the handler to return a plain JSON-serializable dict."""

        return self.model_dump(mode="json", exclude_none=False)
