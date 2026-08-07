"""Background removal helpers.

Ported 1:1 from the backend's processing/background_removal.py. Only the
settings import changed (MODEL_CACHE_DIR here vs. REMBG_MODEL_CACHE_DIR
there) - the function bodies and signatures are identical so callers don't
need to change.
"""

import os
from functools import lru_cache
from pathlib import Path

from app.settings import MODEL_CACHE_DIR

MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("U2NET_HOME", str(MODEL_CACHE_DIR))

from rembg import new_session, remove


@lru_cache(maxsize=4)
def get_segmentation_session(model_name: str):
    """Reuse loaded rembg sessions across jobs in the same warm worker."""

    return new_session(model_name)


def remove_background(
    input_image_path: Path,
    output_cutout_path: Path,
    model_name: str = "u2net",
) -> Path:
    """Remove an image background and save a transparent PNG cutout."""

    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    output_cutout_path.parent.mkdir(parents=True, exist_ok=True)

    input_bytes = input_image_path.read_bytes()
    session = get_segmentation_session(model_name)
    output_bytes = remove(input_bytes, session=session)
    output_cutout_path.write_bytes(output_bytes)

    return output_cutout_path
