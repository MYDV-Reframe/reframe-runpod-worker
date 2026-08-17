"""Deterministic vehicle/background compositor.

Ported from the backend's processing/pipeline.py. Requires ImageMagick, which
must be installed in the worker image (see the Dockerfile change).

Unchanged from the backend apart from the settings import.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from app.settings import (
    DEFAULT_OUTPUT_HEIGHT,
    DEFAULT_OUTPUT_WIDTH,
    DEFAULT_VERTICAL_OFFSET,
    get_imagemagick_command,
)


@dataclass(frozen=True)
class ProcessingOptions:
    output_width: int = DEFAULT_OUTPUT_WIDTH
    output_height: int = DEFAULT_OUTPUT_HEIGHT
    horizontal_offset: int = 0
    vertical_offset: int = DEFAULT_VERTICAL_OFFSET
    car_scale_percent: int = 100
    car_width_percent: int | None = None
    car_height_percent: int | None = None
    trim_transparent_padding: bool = True
    car_brightness: int = 100
    car_saturation: int = 100
    car_contrast: int = 0
    car_warmth_percent: int = 0
    car_softness: float = 0.0
    alpha_threshold_percent: int = 0
    edge_erode_pixels: int = 0
    edge_feather: float = 0.0
    edge_decontaminate_percent: int = 0
    edge_decontaminate_width: int = 3
    shadow_enabled: bool = True
    shadow_opacity_percent: int = 35
    shadow_blur: float = 25.0
    shadow_reference_width: int = DEFAULT_OUTPUT_WIDTH
    shadow_width_percent: int = 100
    shadow_height_percent: int = 22
    shadow_horizontal_offset: int = 0
    shadow_vertical_offset: int = 20
    shadow_gamma_space: str = "srgb"
    contact_shadow_enabled: bool = False
    contact_shadow_opacity_percent: int = 22
    contact_shadow_blur: float = 4.0
    contact_shadow_band_percent: int = 3
    reflection_enabled: bool = False
    reflection_opacity_percent: int = 18
    reflection_blur: float = 2.0
    reflection_height_percent: int = 35
    reflection_vertical_offset: int = 140


@dataclass(frozen=True)
class ProcessingRequest:
    car_image_path: Path
    background_image_path: Path
    output_image_path: Path
    options: ProcessingOptions = ProcessingOptions()


def process_car_image(request: ProcessingRequest) -> Path:
    """Composite a transparent car cutout over a background image."""

    car_image_path = request.car_image_path
    background_image_path = request.background_image_path
    output_image_path = request.output_image_path
    options = request.options

    if not car_image_path.exists():
        raise FileNotFoundError(f"Car image not found: {car_image_path}")
    if not background_image_path.exists():
        raise FileNotFoundError(f"Background image not found: {background_image_path}")

    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        prepared_car_path = temp_path / "prepared-car.png"
        shadow_path = temp_path / "floor-shadow.png"
        contact_shadow_path = temp_path / "contact-shadow.png"
        reflection_path = temp_path / "floor-reflection.png"
        reflection_base_path = temp_path / "floor-reflection-base.png"
        reflection_mask_path = temp_path / "floor-reflection-mask.png"

        _prepare_car_cutout(car_image_path, prepared_car_path, options)
        _decontaminate_edge_color(prepared_car_path, options)

        if options.reflection_enabled:
            _create_floor_reflection(
                prepared_car_path=prepared_car_path,
                reflection_base_path=reflection_base_path,
                reflection_mask_path=reflection_mask_path,
                reflection_path=reflection_path,
                options=options,
            )
        if options.shadow_enabled:
            _create_floor_shadow(prepared_car_path, shadow_path, options)
        if options.contact_shadow_enabled:
            _create_contact_shadow(prepared_car_path, contact_shadow_path, options)

        _composite_layers(
            background_image_path=background_image_path,
            car_image_path=prepared_car_path,
            reflection_image_path=reflection_path if options.reflection_enabled else None,
            shadow_image_path=shadow_path if options.shadow_enabled else None,
            contact_shadow_image_path=(
                contact_shadow_path if options.contact_shadow_enabled else None
            ),
            output_image_path=output_image_path,
            options=options,
        )

    return output_image_path


def _prepare_car_cutout(car_image_path, prepared_car_path, options) -> None:
    command = [get_imagemagick_command(), str(car_image_path), "-auto-orient"]

    if options.trim_transparent_padding:
        command.extend(["-trim", "+repage"])

    if options.car_width_percent is not None and options.car_height_percent is not None:
        tw = round(options.output_width * options.car_width_percent / 100)
        th = round(options.output_height * options.car_height_percent / 100)
        resize_value = f"{tw}x{th}"
    elif options.car_width_percent is not None:
        tw = round(options.output_width * options.car_width_percent / 100)
        resize_value = f"{tw}x"
    elif options.car_height_percent is not None:
        th = round(options.output_height * options.car_height_percent / 100)
        resize_value = f"x{th}"
    else:
        resize_value = f"{options.car_scale_percent}%"
    command.extend(["-resize", resize_value])

    if options.alpha_threshold_percent > 0:
        command.extend(["-channel", "A", "-black-threshold",
                        f"{options.alpha_threshold_percent}%", "+channel"])
    if options.edge_erode_pixels > 0:
        command.extend(["-channel", "A", "-morphology", "Erode",
                        f"Disk:{options.edge_erode_pixels}", "+channel"])
    if options.edge_feather > 0:
        command.extend(["-channel", "A", "-blur", f"0x{options.edge_feather}", "+channel"])

    command.extend(["-modulate",
                    f"{options.car_brightness},{options.car_saturation},100",
                    "-brightness-contrast", f"0x{options.car_contrast}"])

    if options.car_warmth_percent > 0:
        command.extend(["-fill", "#b89462", "-colorize", f"{options.car_warmth_percent}%"])
    if options.car_softness > 0:
        command.extend(["-blur", f"0x{options.car_softness}"])

    command.append(str(prepared_car_path))
    _run_imagemagick(command, "prepare car cutout")


def _decontaminate_edge_color(image_path: Path, options) -> None:
    if options.edge_decontaminate_percent <= 0:
        return

    amount = max(0, min(options.edge_decontaminate_percent, 100)) / 100
    edge_width = max(1, options.edge_decontaminate_width)
    filter_size = edge_width * 2 + 1

    image = Image.open(image_path).convert("RGBA")
    red, green, blue, alpha = image.split()
    rgb = Image.merge("RGB", (red, green, blue))

    visible_mask = alpha.point(lambda value: 255 if value > 8 else 0)
    eroded_visible_mask = visible_mask.filter(ImageFilter.MinFilter(filter_size))
    edge_mask = ImageChops.subtract(visible_mask, eroded_visible_mask)

    desaturated = ImageEnhance.Color(rgb).enhance(1 - amount)
    cleaned_rgb = Image.composite(desaturated, rgb, edge_mask)
    Image.merge("RGBA", (*cleaned_rgb.split(), alpha)).save(image_path)


def _create_floor_shadow(prepared_car_path, shadow_path, options) -> None:
    blur_sigma = options.shadow_blur * (
        options.output_width / max(1, options.shadow_reference_width)
    )
    command = [
        get_imagemagick_command(), str(prepared_car_path),
        "-alpha", "extract",
        "-resize", f"{options.shadow_width_percent}%x{options.shadow_height_percent}%!",
        "-colorspace", "sRGB",
        "-blur", f"0x{blur_sigma:g}",
        "-background", "black",
        "-alpha", "shape",
        "-channel", "A", "-evaluate", "multiply",
        str(options.shadow_opacity_percent / 100), "+channel",
        str(shadow_path),
    ]
    _run_imagemagick(command, "create floor shadow")


def _create_contact_shadow(prepared_car_path, shadow_path, options) -> None:
    with Image.open(prepared_car_path) as source:
        alpha = source.convert("RGBA").getchannel("A")

    width, height = alpha.size
    band_height = max(2, round(height * max(1, options.contact_shadow_band_percent) / 100))
    contact_band = alpha.crop((0, height - band_height, width, height))
    active_columns = [
        contact_band.crop((x, 0, x + 1, band_height)).getbbox() is not None
        for x in range(width)
    ]

    strip_height = max(10, round(height * 0.025))
    shadow = Image.new("L", (width, strip_height), 0)
    draw = ImageDraw.Draw(shadow)
    spread = max(3, round(width * 0.004))
    ellipse_height = max(4, round(strip_height * 0.55))
    index = 0
    while index < width:
        if not active_columns[index]:
            index += 1
            continue
        start = index
        while index < width and active_columns[index]:
            index += 1
        end = index - 1
        draw.ellipse(
            (max(0, start - spread), strip_height - ellipse_height,
             min(width - 1, end + spread), strip_height - 1),
            fill=max(0, min(options.contact_shadow_opacity_percent, 100)) * 255 // 100,
        )

    if options.contact_shadow_blur > 0:
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=options.contact_shadow_blur))
    rgba_shadow = Image.new("RGBA", shadow.size, (0, 0, 0, 0))
    rgba_shadow.putalpha(shadow)
    rgba_shadow.save(shadow_path)


def _create_floor_reflection(prepared_car_path, reflection_base_path,
                             reflection_mask_path, reflection_path, options) -> None:
    _run_imagemagick([
        get_imagemagick_command(), str(prepared_car_path), "-flip",
        "-resize", f"100%x{options.reflection_height_percent}%!",
        "-blur", f"0x{options.reflection_blur}",
        str(reflection_base_path),
    ], "create floor reflection base")

    rw, rh = _get_image_dimensions(reflection_base_path)
    _run_imagemagick([
        get_imagemagick_command(), "-size", f"{rw}x{rh}",
        "gradient:white-black", str(reflection_mask_path),
    ], "create floor reflection mask")

    _run_imagemagick([
        get_imagemagick_command(), str(reflection_base_path),
        "(", str(reflection_base_path), "-alpha", "extract",
        str(reflection_mask_path), "-compose", "multiply", "-composite", ")",
        "-alpha", "off", "-compose", "CopyOpacity", "-composite",
        "-channel", "A", "-evaluate", "multiply",
        str(options.reflection_opacity_percent / 100), "+channel",
        str(reflection_path),
    ], "fade floor reflection")


def _composite_layers(background_image_path, car_image_path, reflection_image_path,
                      shadow_image_path, contact_shadow_image_path,
                      output_image_path, options) -> None:
    output_size = f"{options.output_width}x{options.output_height}"
    car_geometry = f"{options.horizontal_offset:+d}{options.vertical_offset:+d}"
    command = [
        get_imagemagick_command(), str(background_image_path),
        "-resize", f"{output_size}^", "-gravity", "center", "-extent", output_size,
    ]

    if reflection_image_path is not None:
        reflection_geometry = (
            f"{options.horizontal_offset:+d}{options.reflection_vertical_offset:+d}"
        )
        command.extend([str(reflection_image_path), "-gravity", "south",
                        "-geometry", reflection_geometry, "-composite"])

    shadow_geometry = None
    if shadow_image_path is not None:
        shadow_geometry = (
            f"{options.horizontal_offset + options.shadow_horizontal_offset:+d}"
            f"{options.shadow_vertical_offset:+d}"
        )

    if contact_shadow_image_path is not None:
        contact_height = _get_image_dimensions(contact_shadow_image_path)[1]
        contact_vertical_offset = max(0, options.vertical_offset - round(contact_height * 0.45))
        contact_geometry = f"{options.horizontal_offset:+d}{contact_vertical_offset:+d}"
        command.extend([str(contact_shadow_image_path), "-gravity", "south",
                        "-geometry", contact_geometry, "-composite"])

    if shadow_image_path is not None:
        command.extend([str(shadow_image_path), "-gravity", "south",
                        "-geometry", shadow_geometry, "-composite"])

    command.extend([str(car_image_path), "-gravity", "south",
                    "-geometry", car_geometry, "-composite", str(output_image_path)])

    _run_imagemagick(command, "composite layers")


def _run_imagemagick(command: list[str], action: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"ImageMagick is not installed in this image ({action})."
        ) from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"ImageMagick {action} failed: {message}") from error


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Return image dimensions.

    The backend shells out to `magick identify`. That only works on
    ImageMagick 7; the worker image ships ImageMagick 6, where the binary is
    `convert` and `convert identify ...` treats "identify" as an input
    filename:

        convert: unable to open image `identify': No such file or directory

    Pillow gives the same answer without the subprocess, so the dependency is
    dropped here rather than papered over with a version check.

    This only surfaced on presets with contact_shadow enabled
    (outdoor-black-dealership-lot) and on the reflection path, since those are
    the only callers.
    """

    with Image.open(image_path) as image:
        return image.width, image.height
