"""Worker task: exterior composite + OpenAI harmonisation.

Mirrors the second half of the backend's processing/user_flow.py
(process_user_flow) for the harmonize-exterior job type, adapted to be
stateless: fetches the cutout by R2 key, composites it onto the selected
background preset, builds the vehicle-only edit mask, runs harmonisation, and
writes the base preview and final image back to R2.

This is the stage that makes automotive glass transparent. Segmentation
produces an opaque silhouette; harmonisation replaces what shows through the
windows with the selected environment.

Phase 1 simplification: the backend chooses placement via
processing/opencv_geometry.py:analyze_car_geometry(). OpenCV is not in this
image, so placement falls back to the backend's own legacy path
(user_flow.py:analyze_cutout + aspect-ratio bands), which is pure PIL. The
caller can override placement entirely by passing a `placement` block in the
payload. Port opencv_geometry.py to close this gap.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from app.pipeline.background_presets import get_background_preset
from app.pipeline.compositor import (
    ProcessingOptions,
    ProcessingRequest,
    process_car_image,
)
from app.pipeline.openai_harmonization import (
    build_harmonization_prompt,
    harmonize_with_openai,
)
from app.settings import (
    DEFAULT_OUTPUT_HEIGHT,
    DEFAULT_OUTPUT_WIDTH,
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_QUALITY,
)
from app.storage.r2_client import R2Client, private_object_key


def _pick(profile: dict, preset: dict, key: str, default: Any) -> Any:
    if key in profile:
        return profile[key]
    if key in preset:
        return preset[key]
    return default


def analyze_cutout(cutout_path: Path) -> dict[str, float]:
    """Measure visible alpha bounds for fallback placement.

    Copied from the backend's user_flow.py.
    """

    with Image.open(cutout_path) as image:
        image = image.convert("RGBA")
        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            return {"aspect_ratio": 1.8, "visible_width_ratio": 1.0,
                    "visible_height_ratio": 1.0}

        left, top, right, bottom = bbox
        visible_width = max(1, right - left)
        visible_height = max(1, bottom - top)
        return {
            "aspect_ratio": visible_width / visible_height,
            "visible_width_ratio": visible_width / max(1, image.width),
            "visible_height_ratio": visible_height / max(1, image.height),
        }


def build_legacy_profile(cutout_path: Path, preset_options: dict) -> dict[str, Any]:
    """Backend's legacy (non-OpenCV) placement path from user_flow.py."""

    aspect_ratio = analyze_cutout(cutout_path)["aspect_ratio"]

    if aspect_ratio >= 2.35:
        car_width_percent, vertical_offset, pose = 82, 24, "side"
    elif aspect_ratio >= 1.85:
        car_width_percent, vertical_offset, pose = 78, 28, "side"
    elif aspect_ratio >= 1.45:
        car_width_percent, vertical_offset, pose = 74, 32, "three-quarter"
    else:
        car_width_percent, vertical_offset, pose = 68, 36, "front"

    preset_width = preset_options.get("car_width_percent")
    if isinstance(preset_width, int):
        car_width_percent = round((car_width_percent + preset_width) / 2)

    preset_offset = preset_options.get("vertical_offset")
    if isinstance(preset_offset, int):
        vertical_offset = round((vertical_offset + preset_offset) / 2)

    return {
        "car_width_percent": max(64, min(car_width_percent, 84)),
        "vertical_offset": max(12, min(vertical_offset, 44)),
        "_placement_engine": "legacy",
        "_pose": pose,
        "_aspect_ratio": aspect_ratio,
    }


def build_options(width, height, profile, preset) -> ProcessingOptions:
    """Build renderer options from profile and preset. From user_flow.py."""

    return ProcessingOptions(
        output_width=width,
        output_height=height,
        horizontal_offset=_pick(profile, preset, "horizontal_offset", 0),
        vertical_offset=_pick(profile, preset, "vertical_offset", 40),
        car_scale_percent=_pick(profile, preset, "car_scale", 100),
        car_width_percent=_pick(profile, preset, "car_width_percent", None),
        car_height_percent=_pick(profile, preset, "car_height_percent", None),
        trim_transparent_padding=True,
        car_brightness=_pick(profile, preset, "car_brightness", 100),
        car_saturation=_pick(profile, preset, "car_saturation", 100),
        car_contrast=_pick(profile, preset, "car_contrast", 0),
        car_warmth_percent=_pick(profile, preset, "car_warmth", 0),
        car_softness=_pick(profile, preset, "car_softness", 0.0),
        alpha_threshold_percent=_pick(profile, preset, "alpha_threshold", 0),
        edge_erode_pixels=_pick(profile, preset, "edge_erode", 0),
        edge_feather=_pick(profile, preset, "edge_feather", 0.0),
        edge_decontaminate_percent=_pick(profile, preset, "edge_decontaminate", 0),
        edge_decontaminate_width=_pick(profile, preset, "edge_decontaminate_width", 3),
        shadow_enabled=bool(preset.get("shadow_enabled", True)),
        shadow_opacity_percent=_pick(profile, preset, "shadow_opacity", 35),
        shadow_blur=_pick(profile, preset, "shadow_blur", 25.0),
        shadow_reference_width=_pick(profile, preset, "shadow_reference_width", width),
        shadow_width_percent=_pick(profile, preset, "shadow_width", 100),
        shadow_height_percent=_pick(profile, preset, "shadow_height", 22),
        shadow_horizontal_offset=_pick(profile, preset, "shadow_horizontal_offset", 0),
        shadow_vertical_offset=_pick(profile, preset, "shadow_vertical_offset", 20),
        contact_shadow_enabled=bool(preset.get("contact_shadow_enabled", False)),
        contact_shadow_opacity_percent=_pick(profile, preset, "contact_shadow_opacity", 22),
        contact_shadow_blur=_pick(profile, preset, "contact_shadow_blur", 4.0),
        contact_shadow_band_percent=_pick(profile, preset, "contact_shadow_band_percent", 3),
        reflection_enabled=bool(preset.get("reflection", False)),
        reflection_opacity_percent=_pick(profile, preset, "reflection_opacity", 18),
        reflection_blur=_pick(profile, preset, "reflection_blur", 2.0),
        reflection_height_percent=_pick(profile, preset, "reflection_height", 35),
        reflection_vertical_offset=_pick(profile, preset, "reflection_vertical_offset", 140),
    )


def create_editor_cutout(cutout_path, editor_cutout_path, options) -> dict[str, int]:
    """Trim/resize the vehicle layer and return exact canvas placement.

    Copied from the backend's user_flow.py so the mask lines up with what the
    ImageMagick composite produces.
    """

    with Image.open(cutout_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        if options.trim_transparent_padding:
            bbox = image.getchannel("A").getbbox()
            if bbox is not None:
                image = image.crop(bbox)

        source_width, source_height = image.width, image.height
        if options.car_width_percent is not None and options.car_height_percent is not None:
            tw = round(options.output_width * options.car_width_percent / 100)
            th = round(options.output_height * options.car_height_percent / 100)
            scale = min(tw / max(1, image.width), th / max(1, image.height))
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        elif options.car_width_percent is not None:
            tw = round(options.output_width * options.car_width_percent / 100)
            th = round(image.height * tw / max(1, image.width))
            image = image.resize((tw, th), Image.Resampling.LANCZOS)
        elif options.car_height_percent is not None:
            th = round(options.output_height * options.car_height_percent / 100)
            tw = round(image.width * th / max(1, image.height))
            image = image.resize((tw, th), Image.Resampling.LANCZOS)
        elif options.car_scale_percent != 100:
            image = image.resize(
                (round(image.width * options.car_scale_percent / 100),
                 round(image.height * options.car_scale_percent / 100)),
                Image.Resampling.LANCZOS,
            )

        editor_cutout_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(editor_cutout_path)

    x = round((options.output_width - image.width) / 2 + options.horizontal_offset)
    y = options.output_height - image.height - options.vertical_offset
    return {
        "x": x, "y": y, "width": image.width, "height": image.height,
        "canvas_width": options.output_width, "canvas_height": options.output_height,
        "source_width": source_width, "source_height": source_height,
        "scale_factor": round(image.width / max(1, source_width), 4),
    }


def create_openai_edit_assets(*, base_preview_path, editor_cutout_path, placement,
                              input_path, mask_path, edge_allowance_pixels=4) -> None:
    """Create same-size PNG image and vehicle-only mask. From user_flow.py."""

    canvas_size = (int(placement["canvas_width"]), int(placement["canvas_height"]))
    with Image.open(base_preview_path) as base_preview:
        base_preview.convert("RGB").save(input_path, format="PNG")

    with Image.open(editor_cutout_path) as editor_cutout:
        editor_alpha = editor_cutout.convert("RGBA").getchannel("A")

    vehicle_alpha = Image.new("L", canvas_size, 0)
    vehicle_alpha.paste(editor_alpha, (int(placement["x"]), int(placement["y"])))
    if edge_allowance_pixels > 0:
        vehicle_alpha = vehicle_alpha.filter(
            ImageFilter.MaxFilter(edge_allowance_pixels * 2 + 1)
        )

    mask = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    mask.putalpha(ImageOps.invert(vehicle_alpha))
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path, format="PNG")


def run_harmonize_exterior(payload: dict[str, object]) -> dict[str, object]:
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

    r2_input = payload["r2"]
    cutout_key = str(r2_input["cutout_key"])

    preset = get_background_preset(preset_id)
    storage = R2Client()
    cutout_bytes = storage.get_bytes(cutout_key)

    with tempfile.TemporaryDirectory(prefix="vehicle-harmonize-") as temp_dir:
        temp_path = Path(temp_dir)
        cutout_path = temp_path / "cutout.png"
        editor_cutout_path = temp_path / "editor-cutout.png"
        base_path = temp_path / "base-preview.jpg"
        openai_input_path = temp_path / "openai-input.png"
        openai_mask_path = temp_path / "openai-mask.png"
        final_path = temp_path / "final.jpg"
        cutout_path.write_bytes(cutout_bytes)

        preset_options = preset.recommended
        override = payload.get("placement")
        if isinstance(override, dict) and override:
            profile_options = dict(override)
            profile_options.setdefault("_placement_engine", "payload-override")
            profile_options.setdefault("_pose", "unknown")
        else:
            profile_options = build_legacy_profile(cutout_path, preset_options)

        options = build_options(width, height, profile_options, preset_options)

        vehicle_analysis = {
            "pose": profile_options.get("_pose", "unknown"),
            "vehicle_class": str(payload.get("vehicle_class", "unknown")),
            "class_confidence": 0.0,
            "cutout_aspect_ratio": profile_options.get("_aspect_ratio"),
            "reflection_risk": bool(payload.get("reflection_risk", False)),
        }
        prompt = build_harmonization_prompt(
            preset_name=preset.name,
            scene=preset.scene,
            harmonisation=preset.harmonisation,
            vehicle_analysis=vehicle_analysis,
        )

        editor_placement = create_editor_cutout(
            cutout_path=cutout_path,
            editor_cutout_path=editor_cutout_path,
            options=options,
        )

        process_car_image(
            ProcessingRequest(
                car_image_path=cutout_path,
                background_image_path=preset.path,
                output_image_path=base_path,
                options=options,
            )
        )

        if skip_ai:
            final_path.write_bytes(base_path.read_bytes())
        else:
            create_openai_edit_assets(
                base_preview_path=base_path,
                editor_cutout_path=editor_cutout_path,
                placement=editor_placement,
                input_path=openai_input_path,
                mask_path=openai_mask_path,
            )
            harmonize_with_openai(
                input_path=openai_input_path,
                output_path=final_path,
                prompt=prompt,
                model=OPENAI_IMAGE_MODEL,
                quality=openai_quality,
                width=width,
                height=height,
                output_format="jpeg",
                compression=92,
                mask_path=openai_mask_path,
            )

        base_bytes = base_path.read_bytes()
        final_bytes = final_path.read_bytes()

    base_key = private_object_key(
        "previews", owner_id, session_id, image_id, "base-preview-v1.jpg"
    )
    final_key = private_object_key(
        "exports", owner_id, session_id, image_id, "harmonised-v1.jpg"
    )

    base_response = storage.put_bytes(base_key, base_bytes, "image/jpeg")
    final_response = storage.put_bytes(final_key, final_bytes, "image/jpeg")

    return {
        "status": "completed",
        "job_id": job_id,
        "task": "harmonize-exterior",
        "trace_id": trace_id,
        "image_type": "exterior",
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
        "placement": editor_placement,
        "ai_skipped": skip_ai,
        "runtime_ms": _elapsed_ms(started),
        "error": None,
    }


def _elapsed_ms(started: float) -> int:
    return round((time.time() - started) * 1000)
