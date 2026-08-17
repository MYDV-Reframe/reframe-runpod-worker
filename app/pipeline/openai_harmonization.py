"""OpenAI image-edit harmonization for completed base composites.

Ported from the backend's processing/openai_harmonization.py. The prompt
builder is copied verbatim so worker output matches the local path exactly;
only the settings import is repointed at the worker's own settings module.

This is the module that makes automotive glass transparent - see REQUIRED
EDITS item 4 in the generated prompt.
"""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from app.settings import OPENAI_API_KEY_ENV


DEFAULT_HARMONIZATION_PROMPT = """
You are performing professional photographic harmonisation on a completed
vehicle composite.

The supplied image already consists of:
- the original vehicle
- a professionally selected replacement background
- correct vehicle placement

Your task is NOT to redesign, recreate, restyle or regenerate the image.

Your only objective is to transform the composite into a completely natural,
photorealistic image that appears to have been captured in a single camera
exposure inside the selected environment.

Preserve the vehicle exactly as photographed while improving only its
photographic integration with the environment.

Glass:
Treat all automotive glass as real transparent glass. Update reflections,
transmitted scenery, tint, interior visibility, colour cast, exposure,
refraction and Fresnel reflections. Glass should naturally reflect the selected
environment while still allowing realistic visibility into the cabin where
appropriate.

Remove every remaining visual cue from the original photograph. No evidence of
the original capture environment should remain on reflective surfaces.

Eliminate all cut-out or segmentation artefacts. Blend the vehicle naturally
into the background while preserving crisp automotive edges.

Do NOT move, resize, rotate or crop the vehicle, change perspective, redesign
any component, or add people, vehicles, props, text or watermarks.
""".strip()


def build_harmonization_prompt(
    *,
    preset_name: str,
    scene: dict[str, object],
    harmonisation: dict[str, object],
    vehicle_analysis: dict[str, object],
) -> str:
    """Build a compact prompt from the selected scene and detected vehicle."""

    if not scene:
        return DEFAULT_HARMONIZATION_PROMPT

    camera = scene.get("camera", {})
    lighting = scene.get("lighting", {})
    floor = scene.get("floor", {})
    shadow = scene.get("shadow", {})
    reflections = scene.get("reflections", {})
    if not isinstance(camera, dict):
        camera = {}
    if not isinstance(lighting, dict):
        lighting = {}
    if not isinstance(floor, dict):
        floor = {}
    if not isinstance(shadow, dict):
        shadow = {}
    if not isinstance(reflections, dict):
        reflections = {}

    pose = vehicle_analysis.get("pose", "unknown")
    vehicle_class = vehicle_analysis.get("vehicle_class", "unknown")
    objective = harmonisation.get(
        "objective",
        "Make the completed composite look like one natural camera exposure.",
    )
    quality_target = harmonisation.get(
        "quality_target",
        "A natural, photorealistic dealership listing photograph.",
    )
    reflection_risk_instruction = ""
    permitted_reflection_pattern = reflections.get(
        "permitted_pattern",
        "restrained highlights produced only by the target scene lighting",
    )
    source_reflections_to_remove = reflections.get(
        "remove",
        "all reflections that reveal the previous environment",
    )
    surface_guidance = reflections.get("surface_guidance")
    surface_guidance_instruction = ""
    if isinstance(surface_guidance, str) and surface_guidance.strip():
        surface_guidance_instruction = f"""

PRESET-SPECIFIC SURFACE GUIDANCE
- {surface_guidance.strip()}
- Apply this as photographic reflection cleanup only. Do not alter body panels,
  door boundaries, handles, trim or paint identity.
""".rstrip()
    glass_guidance = reflections.get("glass_guidance")
    glass_guidance_instruction = ""
    if isinstance(glass_guidance, str) and glass_guidance.strip():
        glass_guidance_instruction = f"""

PRESET-SPECIFIC GLASS GUIDANCE
- {glass_guidance.strip()}
- Apply this as a glass reflection and transmission adjustment only. Do not
  redesign the windscreen, cabin, dashboard, pillars or mirrors.
""".rstrip()
    paint_highlight_instruction = f"""2. Paint, roof and bonnet may show only reflections and highlights produced
   by the target scene: {permitted_reflection_pattern}. Remove
   {source_reflections_to_remove}. Preserve natural clear-coat gloss and panel
   curvature without inventing recognisable objects."""
    if vehicle_analysis.get("reflection_risk"):
        reflection_risk_instruction = f"""

HIGH-GLOSS DARK PAINT
- The vehicle has dark glossy paint with unusually large bright regions from
  the source environment. Treat those bright shapes as reflections, not as
  permanent paint markings or body details.
- Remove this source reflection content: {source_reflections_to_remove}.
- Replace it only with this target-scene reflection pattern:
  {permitted_reflection_pattern}.
- Preserve bright but unclipped specular cores, natural falloff across curved
  panels and deep paint colour between highlights. Do not make the paint matte,
  satin, grey, uniformly dark or diffusely shaded.
""".rstrip()
        paint_highlight_instruction = f"""2. Preserve a high-gloss automotive clear-coat finish on the dark paint.
   Use only the target-scene pattern described here:
   {permitted_reflection_pattern}. Keep bright but unclipped specular cores,
   smooth falloff across curved panels and deep paint colour between highlights.
   Remove recognisable source-scene shapes without reducing gloss, flattening
   panel curvature or creating a plastic/CGI finish."""

    return f"""
Perform restrained photographic harmonisation on this completed vehicle
composite. {objective}

LOCKED CONTENT
- Keep the background, framing, banner, logo and all readable text pixel-stable.
- Keep the vehicle's identity, geometry, position, scale, orientation, paint
  colour, wheels, lights, badges, mirrors, panel lines and number plate area.
- Do not add, remove, move, resize or redesign anything.

VEHICLE OBSERVATION
- Detected pose: {pose}
- Detected vehicle class: {vehicle_class}
{reflection_risk_instruction}
{surface_guidance_instruction}
{glass_guidance_instruction}

TARGET SCENE: {preset_name}
- Environment: {scene.get('environment', 'indoor dealership showroom')}
- Overall look: {scene.get('overall_look', 'clean and realistic')}
- Camera: {camera.get('perspective', 'eye level')},
  {camera.get('angle', 'straight-on')},
  {camera.get('focal_length', 'natural automotive photography')}
- Lighting: {lighting.get('softness', 'soft')}
  {lighting.get('direction', 'overhead')} light from
  {lighting.get('type', 'showroom fixtures')}
- Ambient level: {lighting.get('ambient_level', 'medium')}
- Colour temperature: {lighting.get('colour_temperature', 'neutral')}K
- Floor: {floor.get('colour', 'grey')}
  {floor.get('material', 'showroom floor')},
  {floor.get('finish', 'natural finish')},
  reflectivity {floor.get('reflectivity', 'medium')}
- Target reflections: {scene.get('reflection_environment', 'the selected scene')}
- Permitted reflection pattern: {permitted_reflection_pattern}
- Source reflections to remove: {source_reflections_to_remove}
- Ground shadow: {shadow.get('style', 'soft diffuse')},
  {shadow.get('direction', 'directly beneath the vehicle')}

REQUIRED EDITS, IN PRIORITY ORDER
1. Replace source-environment reflections and colour contamination on paint,
   glass, chrome and gloss trim. Do not merely dim or recolour the old
   reflections. Remove their shapes and replace them with the permitted target
   reflection pattern described above. Reflections are photographic properties,
   not vehicle redesign.
{paint_highlight_instruction}
3. Match exposure, white balance, highlight roll-off, black level and local
   contrast to the target lighting. Avoid HDR, artificial glow and CGI polish.
4. Keep glass transparent and preserve plausible cabin visibility while removing
   visible source walls, sky, buildings, trees, vehicles and light fixtures.
5. Improve tyre contact and subtle underbody grounding using the target shadow
   description. Do not create a mirror-image floor reflection.
6. Remove cutout halos while retaining crisp roof, mirror, wheel and tyre edges.

When preservation conflicts with harmonisation, preserve physical vehicle
identity and geometry, but change source-environment lighting and reflections.

QUALITY TARGET
{quality_target}
""".strip()


def require_api_key() -> None:
    """Fail fast if OpenAI harmonization is requested without a key."""

    import os

    if os.environ.get(OPENAI_API_KEY_ENV):
        return

    raise RuntimeError(f"{OPENAI_API_KEY_ENV} is not set on the endpoint.")


def decode_image_response(response: object) -> bytes:
    """Extract base64 image bytes from an OpenAI image response."""

    data = getattr(response, "data", None)
    if not data:
        raise RuntimeError("OpenAI image edit response did not contain image data.")

    first = data[0]
    b64_json = getattr(first, "b64_json", None)
    if not b64_json and isinstance(first, dict):
        b64_json = first.get("b64_json")

    if not b64_json:
        raise RuntimeError("OpenAI image edit response did not include b64_json.")

    return base64.b64decode(b64_json)


def normalize_size(image_path: Path, output_path: Path, width: int, height: int) -> None:
    """Ensure the final image has the exact requested dimensions."""

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            image.save(output_path, format="JPEG", quality=92, optimize=True)
        elif suffix == ".webp":
            image.save(output_path, format="WEBP", quality=92)
        else:
            image.save(output_path, format="PNG")


def harmonize_with_openai(
    input_path: Path,
    output_path: Path,
    prompt: str,
    model: str,
    quality: str,
    width: int,
    height: int,
    output_format: str = "jpeg",
    compression: int = 92,
    mask_path: Path | None = None,
) -> Path:
    """Run OpenAI image editing on a completed base composite."""

    require_api_key()

    from openai import OpenAI

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if mask_path is not None and not mask_path.exists():
        raise FileNotFoundError(f"OpenAI edit mask not found: {mask_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_suffix(f".openai-raw.{output_format}")
    size = f"{width}x{height}"
    compression = max(0, min(compression, 100))

    client = OpenAI()
    request_options = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "output_compression": compression,
    }
    with input_path.open("rb") as image_file:
        if mask_path is None:
            response = client.images.edit(image=image_file, **request_options)
        else:
            with mask_path.open("rb") as mask_file:
                response = client.images.edit(
                    image=image_file,
                    mask=mask_file,
                    **request_options,
                )

    raw_output_path.write_bytes(decode_image_response(response))
    normalize_size(raw_output_path, output_path, width, height)
    return output_path
