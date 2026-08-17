"""Background preset loading for the worker.

Ported from the backend's core/background_presets.py. The backend's
shadow_spec normalisation is not ported; this reads the flat `recommended`
block plus the canonical `placement` and `render` sections directly.

Presets and their images are baked into the worker image under ASSETS_DIR,
see the Dockerfile COPY step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.settings import ASSETS_DIR

PRESETS_FILE = ASSETS_DIR / "backgrounds" / "presets.json"


@dataclass(frozen=True)
class BackgroundPreset:
    id: str
    name: str
    category: str
    path: Path
    recommended: dict[str, Any]
    description: str = ""
    scene: dict[str, Any] = field(default_factory=dict)
    placement: dict[str, Any] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    harmonisation: dict[str, Any] = field(default_factory=dict)


def _canonical_pose_name(name: str) -> str:
    """Normalize pose names used by older and newer preset files."""

    return name.replace("_", "-")


def _flatten_canonical_options(item: dict[str, Any]) -> dict[str, Any]:
    """Expose canonical preset sections to the deterministic renderer.

    The `_canonical_schema` flag and normalised `pose_rules` matter: without
    them opencv_geometry.py falls back to blended pose defaults instead of the
    preset's own pose-specific ranges.
    """

    options = dict(item.get("recommended", {}))
    placement = item.get("placement", {})
    render = item.get("render", {})
    if placement or render or item.get("scene"):
        options["_canonical_schema"] = True

    placement_keys = {
        "default_width_percent": "car_width_percent",
        "default_car_width_percent": "car_width_percent",
        "default_floor_line_percent": "floor_line_percent",
        "default_horizontal_offset": "horizontal_offset",
    }
    for source_key, target_key in placement_keys.items():
        if source_key in placement:
            options[target_key] = placement[source_key]

    pose_rules = placement.get("pose_rules")
    if isinstance(pose_rules, dict):
        normalized_rules: dict[str, Any] = {}
        for pose_name, rule in pose_rules.items():
            if not isinstance(rule, dict):
                continue
            normalized_rule = dict(rule)
            renames = {
                "target_width_percent": "car_width_percent",
                "min_width_percent": "min_car_width_percent",
                "target_height_percent": "car_height_percent",
                "max_width_percent": "max_car_width_percent",
                "max_height_percent": "max_car_height_percent",
            }
            for old_key, new_key in renames.items():
                if old_key in normalized_rule:
                    normalized_rule[new_key] = normalized_rule.pop(old_key)
            normalized_rules[_canonical_pose_name(pose_name)] = normalized_rule
        options["pose_rules"] = normalized_rules

    vehicle_adjustments = render.get("vehicle_adjustments", {})
    adjustment_keys = {
        "brightness": "car_brightness",
        "contrast": "car_contrast",
        "saturation": "car_saturation",
        "warmth": "car_warmth",
    }
    for source_key, target_key in adjustment_keys.items():
        if source_key in vehicle_adjustments:
            options[target_key] = vehicle_adjustments[source_key]

    edges = render.get("edges", {})
    if "erode" in edges:
        options["edge_erode"] = edges["erode"]
    if "feather" in edges:
        options["edge_feather"] = edges["feather"]

    shadow = render.get("shadow", {})
    shadow_keys = {
        "opacity": "shadow_opacity",
        "blur": "shadow_blur",
        "height": "shadow_height",
        "vertical_offset": "shadow_vertical_offset",
    }
    for source_key, target_key in shadow_keys.items():
        if source_key in shadow:
            options[target_key] = shadow[source_key]

    contact_shadow = render.get("contact_shadow", {})
    for key in ("enabled", "opacity", "blur", "band_percent"):
        if key in contact_shadow:
            options[f"contact_shadow_{key}"] = contact_shadow[key]

    reflection = render.get("reflection", {})
    if "enabled" in reflection:
        options["reflection"] = reflection["enabled"]
    if "strength" in reflection:
        options["reflection_opacity"] = reflection["strength"]

    return options


def load_background_presets() -> list[BackgroundPreset]:
    if not PRESETS_FILE.exists():
        raise FileNotFoundError(
            f"presets.json not found at {PRESETS_FILE}. "
            "Check the Dockerfile COPY of assets/."
        )

    data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    presets: list[BackgroundPreset] = []

    for item in data:
        # Backend paths look like "assets/backgrounds/generated/x.png";
        # in the worker image assets live at ASSETS_DIR, so strip the prefix.
        relative = item["path"]
        if relative.startswith("assets/"):
            relative = relative[len("assets/") :]

        presets.append(
            BackgroundPreset(
                id=item["id"],
                name=item["name"],
                category=item["category"],
                path=ASSETS_DIR / relative,
                recommended=_flatten_canonical_options(item),
                description=item.get("description", ""),
                scene=item.get("scene", {}),
                placement=item.get("placement", {}),
                render=item.get("render", {}),
                harmonisation=item.get("harmonisation", {}),
            )
        )

    return presets


def get_background_preset(preset_id: str) -> BackgroundPreset:
    presets = load_background_presets()
    for preset in presets:
        if preset.id == preset_id:
            return preset

    available = ", ".join(preset.id for preset in presets)
    raise ValueError(f"Unknown background preset '{preset_id}'. Available: {available}")
