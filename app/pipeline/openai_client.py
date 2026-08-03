"""Minimal OpenAI auth guard.

Ported from the backend's processing/openai_harmonization.py -
require_api_key() only. The rest of that module (harmonization prompts,
images.edit calls) is out of scope for Phase 1; classify-image needs nothing
from it beyond this check.
"""

import os

from app.settings import OPENAI_API_KEY_ENV


def require_api_key() -> None:
    """Fail fast if a classification call is attempted without a key."""

    if os.environ.get(OPENAI_API_KEY_ENV):
        return

    raise RuntimeError(f"{OPENAI_API_KEY_ENV} is not set.")
