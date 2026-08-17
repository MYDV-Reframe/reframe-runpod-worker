"""Worker task: interior cabin harmonisation.

Mirrors the interior branch of the backend's processing/routed_flow.py
(_process_interior_preview) for the harmonize-interior job type, adapted to be
stateless.

Interiors are handled completely differently from exteriors: there is no
segmentation, no compositing and no mask. The full cabin frame is preserved
and the model is instructed to edit only the pixels visible through windows
and glass, replacing the outside scenery with the selected environment. The
`visible_glass` list produced by classify-image feeds into that prompt.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from PIL import Image, ImageOps

from app.pipeline.background_presets import get_background_preset
from app.pipeline.openai_harmonization import harmonize_interior_with_openai
from app.settings import (
    DEFAULT_OUTPUT_HEIGHT,
    DEFAULT_OUTPUT_WIDTH,
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_QUALITY,
)
from app.storage.r2_client import R2Client, private_object_key


def create_interior_preview(input_path: Path, output_path: Path, width: int, height: int) -> None:
    """Preserve the complete cabin frame without segmentation or cropping.

    From routed_flow.py: _create_interior_preview.
    """

    with Image.open(input_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.pad(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            color=(18, 18, 18),
            centering=(0.5, 0.5),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=92, optimize=True)


def create_background_reference(
    source_path: Path, output_path: Path, width: int, height: int
) -> None:
    """From routed_flow.py: _create_background_reference."""

    with Image.open(source_path) as source:
        reference = ImageOps.fit(
            ImageOps.exif_transpose(source).convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reference.save(output_path, format="PNG")


def build_interior_prompt(preset_name: str, view: str, visible_glass: list[str]) -> str:
    """From routed_flow.py: _build_interior_prompt, copied verbatim."""

    glass_hint = ", ".join(visible_glass) or "not confidently detected"

    return f"""
Edit Image 1 using Image 2 only as the selected environment reference.

Image 1 is the authoritative vehicle interior photograph. Image 2 is the
selected {preset_name} scene. Image 2 provides environmental context
only; it is not a replacement composition or permission to expand Image 1.

ABSOLUTE COMPOSITION LOCK
- Preserve Image 1's exact camera position, focal length, crop, framing,
  perspective, canvas boundaries and field of view.
- Do not zoom in, zoom out, reframe, outpaint, uncrop, rotate or expand the
  visible cabin.
- Anything not visible in Image 1 must remain invisible in the result.
- Never invent or reveal an off-frame dashboard, steering wheel, console, gear
  selector, seat, floor mat, mirror, control, screen, pedal, door panel, window
  or pillar.

AUTHORITATIVE VEHICLE CONTENT
Preserve every visible vehicle pixel and object from Image 1: dashboard,
steering wheel, badge, gauges, controls, vents, screens, gear lever, seats,
upholstery, stitching, pedals, mats, pillars, mirrors, window frames, bodywork,
geometry, wear, stains, labels and readable text. Keep their exact count,
position, shape, material, colour and condition. Do not redesign, clean,
enhance, repair, move or regenerate any cabin component.

PERMITTED EDIT
Modify only the existing pixels that show the old outside environment through
already-visible windows, glass or open door apertures. Replace that scenery with
a plausible view of the selected environment from the unchanged camera
position. Match its lighting, colour temperature, perspective, depth and
exposure. Preserve realistic automotive glass tint, transmission, restrained
reflections and interior visibility. A minimal physically plausible colour or
light cast may affect existing surfaces, but their details must remain intact.

The detected source view is {view}; visible-glass hints are
{glass_hint}. These
are observations only, not permission to create additional glass or cabin area.
If little or no outside scenery is visible, leave the cabin and framing
unchanged and make only a minimal edit. Do not show more of Image 2 by creating
new windows or widening the shot. Do not paste Image 2 flatly into the glass.
Do not add people, vehicles, objects, logos, text or watermarks.
""".strip()


def run_harmonize_interior(payload: dict[str, object]) -> dict[str, object]:
    started = time.time()
    job_id = str(payload["job_id"])
    trace_id = str(payload["trace_id"])
    image_id = str(payload["image_id"])
    session_id = str(payload["session_id"])
    owner_id = str(payload["owner_id"])
    preset_id = str(payload["background_preset_id"])

    width = int(payload.get("output_width", DEFAULT_OUTPUT_WIDTH))
    height = int(payload.get("output_height", DEFAULT_OUTPUT_HEIGHT))
    openai_quality = str(payload.get("openai_quality", OPENAI_IMAGE_QUALITY))
    skip_ai = bool(payload.get("skip_ai", False))

    # Classification output from classify-image. The backend reads these off
    # the ImageClassification object; here they arrive in the payload.
    view = str(payload.get("view", "unknown"))
    visible_glass_raw = payload.get("visible_glass") or []
    visible_glass = [str(item) for item in visible_glass_raw]

    r2_input = payload["r2"]
    source_key = str(r2_input["processing_copy_key"])

    preset = get_background_preset(preset_id)
    storage = R2Client()
    source_bytes = storage.get_bytes(source_key)

    with tempfile.TemporaryDirectory(prefix="vehicle-interior-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "interior-source"
        base_path = temp_path / "interior-base-preview.jpg"
        reference_path = temp_path / "background-reference.png"
        final_path = temp_path / "final.jpg"
        input_path.write_bytes(source_bytes)

        create_interior_preview(input_path, base_path, width, height)
        prompt = build_interior_prompt(preset.name, view, visible_glass)

        if skip_ai:
            final_path.write_bytes(base_path.read_bytes())
        else:
            create_background_reference(preset.path, reference_path, width, height)
            harmonize_interior_with_openai(
                interior_path=base_path,
                background_reference_path=reference_path,
                output_path=final_path,
                prompt=prompt,
                model=OPENAI_IMAGE_MODEL,
                quality=openai_quality,
                width=width,
                height=height,
            )

        base_bytes = base_path.read_bytes()
        final_bytes = final_path.read_bytes()

    base_key = private_object_key(
        "previews", owner_id, session_id, image_id, f"interior-base-{preset.id}.jpg"
    )
    final_key = private_object_key(
        "exports", owner_id, session_id, image_id, f"interior-harmonised-{preset.id}.jpg"
    )

    base_response = storage.put_bytes(base_key, base_bytes, "image/jpeg")
    final_response = storage.put_bytes(final_key, final_bytes, "image/jpeg")

    return {
        "status": "completed",
        "job_id": job_id,
        "task": "harmonize-interior",
        "trace_id": trace_id,
        "image_type": "interior",
        "artifacts": {
            "base_preview": {
                "object_key": base_key,
                "content_type": "image/jpeg",
                "size_bytes": len(base_bytes),
                "etag": base_response.get("ETag"),
            },
            "harmonised": {
                "object_key": final_key,
                "content_type": "image/jpeg",
                "size_bytes": len(final_bytes),
                "etag": final_response.get("ETag"),
            },
        },
        "background_preset_id": preset.id,
        "interior_processing": {
            "status": "preview-only" if skip_ai else "full-frame-environment-edited",
            "segmentation_bypassed": True,
            "window_background_replacement": "pending" if skip_ai else "completed",
            "visible_glass": visible_glass,
        },
        "ai_skipped": skip_ai,
        "runtime_ms": _elapsed_ms(started),
        "error": None,
    }


def _elapsed_ms(started: float) -> int:
    return round((time.time() - started) * 1000)
