"""Env-driven configuration for the worker.

Trimmed from the backend's core/settings.py: this worker doesn't need
session/database/Clerk config - only what classify-image and
refine-full-resolution-alpha actually touch. In production these are set as
RunPod endpoint secrets, injected directly into the container; locally they
can be exported before running `python handler.py --test_input ...` (see the
repo's .env.example for the full contract).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = Path(
    os.environ.get("MODEL_CACHE_DIR", str(PROJECT_ROOT / "models" / "rembg"))
)

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_CLASSIFIER_MODEL = os.environ.get("OPENAI_CLASSIFIER_MODEL", "gpt-5-mini")

PREPROCESSING_SEGMENTATION_MODEL = os.environ.get(
    "PREPROCESSING_SEGMENTATION_MODEL", "birefnet-general-lite"
)

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
R2_PRIVATE_BUCKET = os.environ.get(
    "R2_PRIVATE_BUCKET", "vehicle-transform-private-dev"
).strip()
