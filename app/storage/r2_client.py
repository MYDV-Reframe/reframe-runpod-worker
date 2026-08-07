"""R2 (S3-compatible) storage client for the worker.

Ported from the backend's storage/r2_private.py, trimmed to just get_bytes/
put_bytes (the worker never issues presigned browser URLs) and the
private_object_key() helper so output keys use the exact same scheme the
backend already reads.
"""

from __future__ import annotations

import re
from typing import Any

import boto3

from app.settings import (
    R2_ACCESS_KEY_ID,
    R2_ACCOUNT_ID,
    R2_PRIVATE_BUCKET,
    R2_SECRET_ACCESS_KEY,
)

PRIVATE_ARTIFACT_PREFIXES = {
    "raw",
    "processing",
    "mattes",
    "analysis",
    "previews",
    "exports",
    "temporary",
}
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class R2ClientConfigurationError(RuntimeError):
    """Raised when R2 configuration is incomplete."""


def _safe_segment(value: str, label: str) -> str:
    if not SAFE_SEGMENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {label} for private object key.")
    return value


def private_object_key(
    artifact_type: str,
    owner_id: str,
    session_id: str,
    image_id: str,
    filename: str,
) -> str:
    """Build the same non-PII object key the backend's r2_private.py builds."""

    if artifact_type not in PRIVATE_ARTIFACT_PREFIXES:
        raise ValueError(f"Unsupported private artifact type: {artifact_type}")

    segments = (
        artifact_type,
        _safe_segment(owner_id, "owner ID"),
        _safe_segment(session_id, "session ID"),
        _safe_segment(image_id, "image ID"),
        _safe_segment(filename, "filename"),
    )
    return "/".join(segments)


class R2Client:
    def __init__(self) -> None:
        if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_PRIVATE_BUCKET):
            raise R2ClientConfigurationError(
                "R2 credentials are not fully configured (R2_ACCOUNT_ID, "
                "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_PRIVATE_BUCKET)."
            )
        self._bucket = R2_PRIVATE_BUCKET
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
        )

    def get_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> dict[str, Any]:
        return self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
