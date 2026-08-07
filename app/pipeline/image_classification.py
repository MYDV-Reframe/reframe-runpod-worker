"""Semantic routing for exterior and interior vehicle photographs.

Ported 1:1 from the backend's processing/image_classification.py. Only two
imports changed: OPENAI_CLASSIFIER_MODEL now comes from app.settings, and
require_api_key() now comes from the local openai_client module instead of
the backend's openai_harmonization.py (which is out of scope for Phase 1).
Everything else - including the OpenAI prompt/schema - is unchanged, so
classification behavior is identical to the local path.
"""

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from app.settings import OPENAI_CLASSIFIER_MODEL
from app.pipeline.openai_client import require_api_key


ImageType = Literal["exterior", "interior", "uncertain"]
ProcessingMode = Literal["auto", "exterior", "interior"]


class _ClassificationPayload(BaseModel):
    image_type: ImageType
    confidence: float = Field(ge=0.0, le=1.0)
    view: str
    visible_glass: list[str]
    reason: str


@dataclass(frozen=True)
class ImageClassification:
    image_type: ImageType
    confidence: float
    view: str
    visible_glass: tuple[str, ...]
    reason: str
    provider: str
    model: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def image_data_url(image_path: Path) -> str:
    """Create a compact, orientation-correct JPEG for the routing request."""

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def classify_vehicle_image(
    image_path: Path,
    mode: ProcessingMode = "auto",
    model: str = OPENAI_CLASSIFIER_MODEL,
) -> ImageClassification:
    """Classify one upload before any destructive segmentation takes place."""

    if mode != "auto":
        return ImageClassification(
            image_type=mode,
            confidence=1.0,
            view="explicit-override",
            visible_glass=(),
            reason=f"Processing mode was explicitly set to {mode}.",
            provider="explicit",
        )

    require_api_key()
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The OpenAI Python package is required for automatic image routing."
        ) from error

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        store=False,
        max_output_tokens=500,
        reasoning={"effort": "minimal"},
        text_format=_ClassificationPayload,
        instructions=(
            "Classify an automotive listing photograph for an editing pipeline. "
            "Use interior when the camera is inside the cabin or the dashboard, "
            "steering wheel, seats, controls, boot/cargo area, or cabin trim is the "
            "main subject. Use exterior when the vehicle body is the main subject. "
            "Use uncertain only when neither route is safe. Do not infer from the "
            "filename. visible_glass should name windows through which a replacement "
            "background could plausibly be visible. Keep reason under 20 words."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Route this uploaded vehicle photograph.",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url(image_path),
                        "detail": "low",
                    },
                ],
            }
        ],
    )
    payload = response.output_parsed
    if payload is None:
        status = getattr(response, "status", "unknown")
        incomplete = getattr(response, "incomplete_details", None)
        raise RuntimeError(
            "The image classifier returned no structured result "
            f"(status={status}, incomplete_details={incomplete})."
        )

    return ImageClassification(
        image_type=payload.image_type,
        confidence=payload.confidence,
        view=payload.view,
        visible_glass=tuple(payload.visible_glass),
        reason=payload.reason,
        provider="openai",
        model=model,
    )
