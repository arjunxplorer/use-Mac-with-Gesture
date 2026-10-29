"""Persistent user settings (JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_DIR = Path.home() / "Library" / "Application Support" / "GestureMac"
DEFAULT_SETTINGS_PATH = DEFAULT_SETTINGS_DIR / "settings.json"

# Ranges documented for tuning; values are clamped on load/save.
POINTER_SENSITIVITY_RANGE = (100.0, 5000.0)
POINTER_GAIN_RANGE = (0.1, 10.0)
POINTER_ACCEL_SPEED_RANGE = (0.01, 5.0)
SMOOTHING_ALPHA_RANGE = (0.05, 1.0)
PINCH_THRESHOLD_RANGE = (0.05, 0.8)
SCROLL_SECONDS_RANGE = (0.0, 1.0)
SCROLL_DEAD_ZONE_RANGE = (0.0, 0.3)
SCROLL_MAX_RANGE_RANGE = (0.01, 1.0)
SCROLL_SPEED_RANGE = (0.0, 6000.0)
SCROLL_SLACK_RANGE = (1.0, 3.0)
SCROLL_SEPARATION_RANGE = (0.05, 3.0)
SCROLL_ANGLE_RANGE = (1.0, 90.0)
SCROLL_EXTENSION_RANGE = (0.0, 1.0)
DRAG_HOLD_SECONDS_RANGE = (0.1, 2.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class Settings:
    pointer_sensitivity: float = 1500.0
    pointer_min_gain: float = 0.4
    pointer_max_gain: float = 3.5
    pointer_accel_slow_speed: float = 0.08
    pointer_accel_fast_speed: float = 0.8
    smoothing_alpha: float = 0.35
    invert_scroll: bool = False
    scroll_activation_hold_seconds: float = 0.15
    scroll_dead_zone: float = 0.03
    scroll_max_range: float = 0.15
    scroll_min_speed: float = 60.0
    scroll_max_speed: float = 1500.0
    scroll_smoothing_alpha: float = 0.3
    scroll_finger_separation_max: float = 0.5
    scroll_finger_parallel_tolerance_degrees: float = 25.0
    scroll_finger_extension_min: float = 0.8
    scroll_max_tilt_degrees: float = 45.0
    scroll_release_slack: float = 1.5
    scroll_exit_grace_seconds: float = 0.1
    pinch_down_threshold: float = 0.25
    pinch_up_threshold: float = 0.35
    drag_hold_seconds: float = 0.5
    show_diagnostics: bool = False

    def clamped(self) -> Settings:
        down = _clamp(
            self.pinch_down_threshold,
            PINCH_THRESHOLD_RANGE[0],
            PINCH_THRESHOLD_RANGE[1],
        )
        up = _clamp(
            self.pinch_up_threshold,
            PINCH_THRESHOLD_RANGE[0],
            PINCH_THRESHOLD_RANGE[1],
        )
        if up <= down:
            up = min(PINCH_THRESHOLD_RANGE[1], down + 0.05)

        min_gain = _clamp(self.pointer_min_gain, *POINTER_GAIN_RANGE)
        max_gain = max(min_gain, _clamp(self.pointer_max_gain, *POINTER_GAIN_RANGE))
        slow = _clamp(self.pointer_accel_slow_speed, *POINTER_ACCEL_SPEED_RANGE)
        fast = _clamp(self.pointer_accel_fast_speed, *POINTER_ACCEL_SPEED_RANGE)
        if fast <= slow:
            fast = slow + 0.1

        scroll_min = _clamp(self.scroll_min_speed, *SCROLL_SPEED_RANGE)
        scroll_max = max(scroll_min, _clamp(self.scroll_max_speed, *SCROLL_SPEED_RANGE))

        return Settings(
            pointer_sensitivity=_clamp(
                self.pointer_sensitivity,
                *POINTER_SENSITIVITY_RANGE,
            ),
            pointer_min_gain=min_gain,
            pointer_max_gain=max_gain,
            pointer_accel_slow_speed=slow,
            pointer_accel_fast_speed=fast,
            smoothing_alpha=_clamp(self.smoothing_alpha, *SMOOTHING_ALPHA_RANGE),
            invert_scroll=bool(self.invert_scroll),
            scroll_activation_hold_seconds=_clamp(
                self.scroll_activation_hold_seconds, *SCROLL_SECONDS_RANGE
            ),
            scroll_dead_zone=_clamp(self.scroll_dead_zone, *SCROLL_DEAD_ZONE_RANGE),
            scroll_max_range=_clamp(self.scroll_max_range, *SCROLL_MAX_RANGE_RANGE),
            scroll_min_speed=scroll_min,
            scroll_max_speed=scroll_max,
            scroll_smoothing_alpha=_clamp(self.scroll_smoothing_alpha, *SMOOTHING_ALPHA_RANGE),
            scroll_finger_separation_max=_clamp(
                self.scroll_finger_separation_max, *SCROLL_SEPARATION_RANGE
            ),
            scroll_finger_parallel_tolerance_degrees=_clamp(
                self.scroll_finger_parallel_tolerance_degrees, *SCROLL_ANGLE_RANGE
            ),
            scroll_finger_extension_min=_clamp(
                self.scroll_finger_extension_min, *SCROLL_EXTENSION_RANGE
            ),
            scroll_max_tilt_degrees=_clamp(self.scroll_max_tilt_degrees, *SCROLL_ANGLE_RANGE),
            scroll_release_slack=_clamp(self.scroll_release_slack, *SCROLL_SLACK_RANGE),
            scroll_exit_grace_seconds=_clamp(self.scroll_exit_grace_seconds, *SCROLL_SECONDS_RANGE),
            pinch_down_threshold=down,
            pinch_up_threshold=up,
            drag_hold_seconds=_clamp(self.drag_hold_seconds, *DRAG_HOLD_SECONDS_RANGE),
            show_diagnostics=bool(self.show_diagnostics),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered).clamped()


def default_settings_path() -> Path:
    return DEFAULT_SETTINGS_PATH


def load(path: Path | None = None) -> Settings:
    """Load settings from JSON; missing or corrupt file returns defaults."""
    target = path or default_settings_path()
    if not target.is_file():
        return Settings().clamped()

    try:
        raw = target.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return Settings().clamped()
        return Settings.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return Settings().clamped()


def save(settings: Settings, path: Path | None = None) -> Path:
    """Persist clamped settings; creates parent directory if needed."""
    target = path or default_settings_path()
    clamped = settings.clamped()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(clamped.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
