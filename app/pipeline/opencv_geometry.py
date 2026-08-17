"""OpenCV-based geometry analysis for car placement.

Ported verbatim from the backend's processing/opencv_geometry.py. The module
has no backend imports, so it is copied unchanged - placement decisions on
RunPod now match the local path exactly.

Requires opencv-python-headless (see requirements.txt).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeometryHints:
    """Placement values derived from the transparent cutout alpha channel."""

    car_width_percent: int
    car_height_percent: int
    vertical_offset: int
    horizontal_offset: int
    pose: str
    aspect_ratio: float
    visible_width_ratio: float
    visible_height_ratio: float
    contact_width_ratio: float
    floor_line_percent: int
    visible_width_pixels: int
    visible_height_pixels: int
    sizing_strategy: str
    camera_angle_risk: bool
    reflection_risk: bool
    median_luminance: float
    bright_pixel_ratio: float


def analyze_car_geometry(
    cutout_path: Path,
    output_width: int,
    output_height: int,
    preset_options: dict[str, Any],
) -> GeometryHints:
    """Return OpenCV placement hints for one transparent car cutout."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed in this image. Add opencv-python-headless "
            "to requirements.txt."
        ) from error

    image = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cutout image not found: {cutout_path}")

    if image.ndim < 3 or image.shape[2] < 4:
        raise ValueError(f"Cutout image does not have an alpha channel: {cutout_path}")

    alpha = image[:, :, 3]
    _, mask = cv2.threshold(alpha, 8, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(mask)
    if points is None:
        return GeometryHints(
            car_width_percent=74,
            car_height_percent=76,
            vertical_offset=round(output_height * 0.06),
            horizontal_offset=0,
            pose="unknown",
            aspect_ratio=1.8,
            visible_width_ratio=1.0,
            visible_height_ratio=1.0,
            contact_width_ratio=0.0,
            floor_line_percent=94,
            visible_width_pixels=0,
            visible_height_pixels=0,
            sizing_strategy="fallback",
            camera_angle_risk=False,
            reflection_risk=False,
            median_luminance=0.0,
            bright_pixel_ratio=0.0,
        )

    x, y, width, height = cv2.boundingRect(points)
    aspect_ratio = width / max(1, height)
    visible_width_ratio = width / max(1, image.shape[1])
    visible_height_ratio = height / max(1, image.shape[0])
    bottom_band_start = max(y, y + height - max(8, round(height * 0.08)))
    bottom_band = mask[bottom_band_start : y + height, x : x + width]
    column_has_contact = np.any(bottom_band > 0, axis=0)
    contact_width_ratio = float(np.count_nonzero(column_has_contact)) / max(1, width)
    pose = classify_pose(aspect_ratio, contact_width_ratio)
    camera_angle_risk = aspect_ratio < 1.35 and contact_width_ratio < 0.75

    opaque_pixels = alpha > 200
    luminance = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)[opaque_pixels]
    median_luminance = float(np.median(luminance)) if luminance.size else 0.0
    bright_pixel_ratio = (
        float(np.count_nonzero(luminance > 190)) / max(1, luminance.size)
    )
    reflection_risk = median_luminance < 55 and bright_pixel_ratio > 0.15

    car_width_percent, car_height_percent = default_size_for_pose(pose)
    preset_width = pose_option(
        preset_options,
        pose,
        "car_width_percent",
        preset_options.get("car_width_percent"),
    )
    preset_height = pose_option(
        preset_options,
        pose,
        "car_height_percent",
        preset_options.get("car_height_percent"),
    )
    sizing_strategy = "pose-default"
    if preset_options.get("_canonical_schema"):
        adaptive_width = adaptive_width_for_pose(
            preset_options=preset_options,
            pose=pose,
            aspect_ratio=aspect_ratio,
        )
        if adaptive_width is not None:
            car_width_percent = adaptive_width
            sizing_strategy = "pose-aspect-range"
        elif isinstance(preset_width, int):
            car_width_percent = preset_width
            sizing_strategy = "pose-target"
        if isinstance(preset_height, int):
            car_height_percent = preset_height
    else:
        car_width_percent = blend_int(car_width_percent, preset_width, weight=0.35)
        car_height_percent = blend_int(car_height_percent, preset_height, weight=0.35)

    floor_line_percent = int(
        pose_option(
            preset_options,
            pose,
            "floor_line_percent",
            preset_options.get("floor_line_percent", 94),
        )
    )
    if camera_angle_risk:
        car_height_percent = min(car_height_percent, 52)
        floor_line_percent = max(floor_line_percent, 95)
        sizing_strategy = f"{sizing_strategy}-elevated-camera"
    vertical_offset = round(output_height * (100 - floor_line_percent) / 100)
    vertical_offset = blend_int(
        vertical_offset,
        pose_option(
            preset_options, pose, "vertical_offset", preset_options.get("vertical_offset")
        ),
        weight=0.2,
    )

    horizontal_offset = int(
        pose_option(
            preset_options,
            pose,
            "horizontal_offset",
            preset_options.get("horizontal_offset", 0),
        )
    )

    pose_width_limit = pose_option(preset_options, pose, "max_car_width_percent")
    if isinstance(pose_width_limit, int):
        car_width_percent = min(car_width_percent, pose_width_limit)

    pose_height_limit = pose_option(preset_options, pose, "max_car_height_percent")
    if isinstance(pose_height_limit, int):
        car_height_percent = min(car_height_percent, pose_height_limit)

    if pose == "front":
        car_width_percent = min(car_width_percent, 58)
        car_height_percent = min(car_height_percent, 68)
    elif pose == "three-quarter":
        car_width_percent = min(car_width_percent, 76)
        car_height_percent = min(car_height_percent, 78)
    else:
        car_width_percent = min(car_width_percent, 84)
        car_height_percent = min(car_height_percent, 72)

    return GeometryHints(
        car_width_percent=max(52, min(car_width_percent, 84)),
        car_height_percent=max(48, min(car_height_percent, 82)),
        vertical_offset=max(18, min(vertical_offset, round(output_height * 0.12))),
        horizontal_offset=horizontal_offset,
        pose=pose,
        aspect_ratio=aspect_ratio,
        visible_width_ratio=visible_width_ratio,
        visible_height_ratio=visible_height_ratio,
        contact_width_ratio=contact_width_ratio,
        floor_line_percent=floor_line_percent,
        visible_width_pixels=width,
        visible_height_pixels=height,
        sizing_strategy=sizing_strategy,
        camera_angle_risk=camera_angle_risk,
        reflection_risk=reflection_risk,
        median_luminance=round(median_luminance, 2),
        bright_pixel_ratio=round(bright_pixel_ratio, 4),
    )


def adaptive_width_for_pose(
    preset_options: dict[str, Any],
    pose: str,
    aspect_ratio: float,
) -> int | None:
    """Interpolate occupancy inside a preset's pose-specific width range."""

    minimum = pose_option(preset_options, pose, "min_car_width_percent")
    preferred_maximum = pose_option(preset_options, pose, "car_width_percent")
    if not isinstance(minimum, int) or not isinstance(preferred_maximum, int):
        return None

    default_aspect_ranges = {
        "front": (0.85, 1.35),
        "three-quarter": (1.35, 1.85),
        "side": (1.85, 2.65),
    }
    default_minimum, default_maximum = default_aspect_ranges.get(pose, (1.0, 2.5))
    aspect_minimum = pose_option(preset_options, pose, "aspect_ratio_min", default_minimum)
    aspect_maximum = pose_option(preset_options, pose, "aspect_ratio_max", default_maximum)
    if not isinstance(aspect_minimum, (int, float)) or not isinstance(
        aspect_maximum, (int, float)
    ):
        return None

    span = max(0.01, float(aspect_maximum) - float(aspect_minimum))
    progress = (aspect_ratio - float(aspect_minimum)) / span
    progress = max(0.0, min(progress, 1.0))
    return round(minimum + (preferred_maximum - minimum) * progress)


def classify_pose(aspect_ratio: float, contact_width_ratio: float | None = None) -> str:
    """Classify pose from silhouette aspect plus a conservative contact cue."""

    if aspect_ratio < 1.35:
        if contact_width_ratio is not None and contact_width_ratio < 0.75:
            return "three-quarter"
        return "front"
    if aspect_ratio < 1.85:
        return "three-quarter"
    return "side"


def default_size_for_pose(pose: str) -> tuple[int, int]:
    """Return width and height caps as output percentages for a pose class."""

    if pose == "front":
        return 56, 68
    if pose == "three-quarter":
        return 74, 76
    return 82, 70


def blend_int(base_value: int, preset_value: object, weight: float) -> int:
    """Blend a computed value with a preset value when the preset is numeric."""

    if not isinstance(preset_value, int):
        return base_value
    return round(base_value * (1 - weight) + preset_value * weight)


def pose_option(
    preset_options: dict[str, Any],
    pose: str,
    key: str,
    default: object = None,
) -> object:
    """Read a pose-specific preset option with a generic fallback."""

    pose_rules = preset_options.get("pose_rules")
    if isinstance(pose_rules, dict):
        pose_rule = pose_rules.get(pose)
        if isinstance(pose_rule, dict) and key in pose_rule:
            return pose_rule[key]
    return default
