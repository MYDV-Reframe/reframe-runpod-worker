"""OpenAI image-edit harmonization for completed base composites.

Ported from the backend's processing/openai_harmonization.py. The prompt
builder is copied verbatim so worker output matches the local path exactly;
only the settings import is repointed at the worker's own settings module.

This is the module that makes automotive glass transparent - see REQUIRED
EDITS item 4 in the generated prompt.

PROMPT ORDERING
---------------
PAINT_REALISM_ENHANCEMENT is emitted at the START of the prompt, before the
scene-specific instructions. It was originally appended at the end and the
effect was weak - the model gives less weight to the tail of a long prompt.
Moving it to the front made a visible difference.

Note the enhancement still says it is additional to, not a replacement for,
the harmonisation instructions, so the ordering does not change its meaning.
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


PAINT_REALISM_ENHANCEMENT = """
ADDITIONAL VEHICLE PAINT & SURFACE REALISM ENHANCEMENT

In addition to all existing harmonisation requirements, apply the following
enhancement to the vehicle itself.

The vehicle must continue to be treated as the exact original vehicle being
harmonised into the supplied environment. Do not replace, redesign, repaint,
reshape or otherwise alter the vehicle. This is an additional realism and
finish pass on the existing vehicle only.

AUTOMOTIVE PAINT REALISM
Improve the optical appearance of the vehicle's existing paint so that it looks
like a genuine, professionally prepared and freshly machine-polished vehicle
photographed for high-quality commercial automotive advertising.

Increase the perceived paint depth, colour richness, colour density, clear-coat
quality, surface smoothness, natural gloss and three-dimensionality.

Correct any flat, dull, hazy, grey, milky or washed-out appearance caused by
compositing or harmonisation.

Preserve the vehicle's exact original paint colour and character. Do not simply
increase global saturation or brightness.
- Dark colours should remain genuinely deep and rich rather than becoming grey.
- Silver and grey vehicles should retain convincing metallic depth and variation
  rather than appearing as a flat grey surface.
- White vehicles should remain clean and bright while retaining subtle
  body-surface definition.
- Coloured vehicles should have richer, deeper paint without becoming
  oversaturated or artificially vivid.
- Where the original vehicle has metallic or pearl paint, preserve and enhance
  its natural metallic/pearl response without inventing excessive sparkle or
  visible artificial texture.

CLEAR-COAT AND POLISHED FINISH
Give the existing paint a realistic automotive clear-coat appearance. The
surface should appear clean, smooth, deep and highly polished, similar to a
vehicle that has just received professional detailing and paint correction.

Create realistic specular highlights and reflected light across the existing
bodywork using soft elongated highlights, broad reflected light, subtle bright
reflection bands, darker reflected areas, smooth transitions between highlights
and shadows, and reflections that follow the curvature of individual panels.

Do not apply an identical gloss or brightness level to the whole vehicle.
Do not make the paint look wet, oily, chrome-like, plastic or artificially
mirror-polished. The objective is premium automotive paint, not an obvious
"shine effect".

ENVIRONMENTAL REFLECTIONS
Where physically appropriate, make the vehicle's paint visibly respond to the
actual environment into which it has been placed. Use the supplied background
and its lighting as the source of realistic environmental reflections.

For outdoor environments, consider naturally reflected sky, clouds, surrounding
architecture, buildings, darker structures and surrounding environmental tones.

For indoor showroom environments, consider naturally reflected ceiling lights,
LED panels, windows, large glass areas, walls, architectural features and
darker areas of the showroom.

For example, long ceiling LED fixtures should produce subtle elongated reflected
highlights across appropriate bonnet, roof, shoulder and side panels. Large
windows should create broad, soft reflections. Dark architectural areas should
produce correspondingly darker reflected regions.

Reflections must follow the curvature, orientation and geometry of the existing
vehicle panels. Do not add generic studio reflections that do not correspond to
the supplied environment. Do not create reflections that contradict the
direction, colour or intensity of the scene lighting.

BODYWORK DIMENSIONALITY
Use realistic light, shade and reflection to make the existing vehicle bodywork
appear more three-dimensional and sculpted. Subtly enhance the visual definition
of existing bonnet contours, shoulder lines, door surfaces, wing surfaces, wheel
arches, roof curvature, factory character lines and other existing body
contours.

Allow recessed areas and surfaces facing away from the principal light source to
remain naturally deeper. Maintain realistic tonal separation between illuminated
and shaded panels. Do not artificially sharpen panel edges or invent additional
contours. Do not change the vehicle's geometry.

LIGHTING INTEGRATION
All new paint highlights, reflections and surface contrast must be consistent
with the existing harmonisation and the supplied environment. The vehicle should
appear to have been physically photographed in the environment rather than cut
out and placed onto it.

Maintain consistency with light direction, light intensity, colour temperature,
softness of the light, environmental brightness and shadows already created
during harmonisation. The paint should respond naturally to the same light
sources affecting the surrounding environment.

MATERIAL-SPECIFIC BEHAVIOUR
Preserve the natural optical characteristics of different vehicle materials.
- Paint should have realistic automotive clear-coat reflections.
- Glass should remain glass and should not receive the same treatment as
  painted bodywork.
- Chrome and metallic trim should retain their own reflective characteristics.
- Black plastic, rubber, tyres and textured trim should not be artificially
  glossed.
- Lights should retain their existing clarity and construction.
- Wheels should remain realistic metal/alloy surfaces rather than being treated
  as painted bodywork.

Apply the strongest paint enhancement selectively to genuine painted body
panels.

VEHICLE IDENTITY PRESERVATION
This enhancement must never change the identity or specification of the vehicle.
Preserve exactly: body shape, proportions, panel geometry, original paint
colour, wheels, tyres, badges, registration plate, headlights, taillights,
grille, windows, mirrors, trim, factory features and existing vehicle details.

Do not invent, remove or redesign vehicle features. Do not modify the
registration plate. Do not make the vehicle appear newer, modified or like a
different model.

COMMERCIAL PHOTOGRAPHY TARGET
The final vehicle should look like a real car photographed by a professional
automotive photographer for a premium UK dealership advertisement immediately
after professional preparation and machine polishing.

Desired: deep paint, rich colour, clean clear coat, realistic environmental
reflections, controlled specular highlights, natural panel definition and
convincing three-dimensional surface depth.

Avoid: flat paint, washed-out colour, grey haze, weak reflections, artificial
HDR, excessive saturation, plastic-looking gloss, mirror-like reflections,
excessive sharpening and fake studio reflections.

IMPORTANT PRIORITY
This is an additional enhancement to the existing harmonisation process, not a
replacement for it. Retain and execute all existing vehicle placement,
perspective, scale, lighting, shadow, colour matching, environmental integration
and realism instructions. Then additionally improve the vehicle's paint and
surface response as described above.

Do not achieve the effect simply by increasing brightness, contrast, saturation
or sharpness. The desired result comes primarily from realistic automotive paint
response to the surrounding environment: deeper colour, clear-coat depth,
physically plausible reflected light and controlled specular highlights
following the existing vehicle's body geometry.

The result must remain photorealistic and commercially usable.
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
        return PAINT_REALISM_ENHANCEMENT + "\n\n" + DEFAULT_HARMONIZATION_PROMPT

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
{PAINT_REALISM_ENHANCEMENT}

================================================================================

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


def harmonize_interior_with_openai(
    *,
    interior_path: Path,
    background_reference_path: Path,
    output_path: Path,
    prompt: str,
    model: str,
    quality: str,
    width: int,
    height: int,
    compression: int = 92,
) -> Path:
    """Edit an interior with a scene reference and normalize it as the final.

    Ported from the backend. Unlike the exterior path this sends two images -
    the cabin photo and the selected environment - with no mask, because the
    edit is scoped by the prompt rather than by a vehicle silhouette.
    """

    require_api_key()

    from openai import OpenAI

    for path in (interior_path, background_reference_path):
        if not path.exists():
            raise FileNotFoundError(f"Interior edit asset not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_suffix(".openai-raw.jpeg")
    request_options = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "quality": quality,
        "output_format": "jpeg",
        "output_compression": max(0, min(compression, 100)),
    }
    with (
        interior_path.open("rb") as interior_file,
        background_reference_path.open("rb") as background_file,
    ):
        response = OpenAI().images.edit(
            image=[interior_file, background_file],
            **request_options,
        )

    raw_output_path.write_bytes(decode_image_response(response))
    normalize_size(raw_output_path, output_path, width, height)
    return output_path
