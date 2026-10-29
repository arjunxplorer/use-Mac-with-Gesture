"""Tunable gesture parameters (from settings + spec defaults)."""

from __future__ import annotations

from dataclasses import dataclass

from gesturemac.settings import Settings

DEFAULT_DOUBLE_CLICK_INTERVAL = 0.35


def system_double_click_interval() -> float:
    try:
        from AppKit import NSEvent  # type: ignore[import-untyped]

        interval = float(NSEvent.doubleClickInterval())
        if 0.1 <= interval <= 5.0:
            return interval
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_DOUBLE_CLICK_INTERVAL


@dataclass
class GestureConfig:
    min_confidence: float = 0.5
    # Screen points per unit of hand motion at gain 1, where 1 unit = the camera image's short side.
    pointer_sensitivity: float = 1500.0
    # Gain ramps from min (at or below slow speed) to max (at or above fast speed), in units/s.
    # Set min == max to disable acceleration.
    pointer_min_gain: float = 0.4
    pointer_max_gain: float = 3.5
    pointer_accel_slow_speed: float = 0.08
    pointer_accel_fast_speed: float = 0.8
    speed_smoothing: float = 0.5
    smoothing_alpha: float = 0.35
    stable_pose_seconds: float = 0.120
    # Hand motion (units) below this is treated as tremor; slow drift still accumulates past it.
    jitter_dead_zone: float = 0.0008
    # After any anchor reset, keep re-anchoring this long so an uncurling finger doesn't move the cursor.
    reanchor_settle_seconds: float = 0.08
    max_delta_points: float = 250.0
    frame_gap_reset_seconds: float = 0.25
    hand_loss_seconds: float = 0.200
    finger_extend_ratio: float = 1.05
    # Index counts as folded (clutch) below this tip/PIP distance ratio; also blocks fist "pinches".
    index_fold_ratio: float = 0.9
    min_palm_scale: float = 0.02
    pinch_down_threshold: float = 0.25
    pinch_up_threshold: float = 0.35
    double_click_interval: float = DEFAULT_DOUBLE_CLICK_INTERVAL
    double_click_distance: float = 8.0
    # A pinch held this long presses the button and starts a drag instead of clicking on release.
    drag_hold_seconds: float = 0.35
    # Joystick scrolling. Distances are in units (1 = the camera image's short side), speeds in
    # pixels per second. Fingers above the neutral center scroll down, below scroll up.
    invert_scroll: bool = False
    scroll_activation_hold_seconds: float = 0.15
    scroll_dead_zone: float = 0.03
    scroll_max_range: float = 0.15
    scroll_min_speed: float = 60.0
    scroll_max_speed: float = 1500.0
    scroll_smoothing_alpha: float = 0.3
    # Side-view pose: tip gap / palm length, angle between the two fingers, MCP->tip straightness
    # (1.0 = perfectly straight), and how far the fingers may tilt from horizontal.
    scroll_finger_separation_max: float = 0.5
    scroll_finger_parallel_tolerance_degrees: float = 25.0
    scroll_finger_extension_min: float = 0.8
    scroll_max_tilt_degrees: float = 45.0
    # Once scrolling, the pose limits above loosen by this factor so landmark noise can't stutter it.
    scroll_release_slack: float = 1.5
    # A pose dropout shorter than this keeps the neutral center; scrolling still pauses at once.
    scroll_exit_grace_seconds: float = 0.1
    max_scroll_pixels: int = 120

    @classmethod
    def from_settings(cls, settings: Settings) -> GestureConfig:
        return cls(
            pointer_sensitivity=settings.pointer_sensitivity,
            pointer_min_gain=settings.pointer_min_gain,
            pointer_max_gain=settings.pointer_max_gain,
            pointer_accel_slow_speed=settings.pointer_accel_slow_speed,
            pointer_accel_fast_speed=settings.pointer_accel_fast_speed,
            smoothing_alpha=settings.smoothing_alpha,
            pinch_down_threshold=settings.pinch_down_threshold,
            pinch_up_threshold=settings.pinch_up_threshold,
            double_click_interval=system_double_click_interval(),
            drag_hold_seconds=settings.drag_hold_seconds,
            invert_scroll=settings.invert_scroll,
            scroll_activation_hold_seconds=settings.scroll_activation_hold_seconds,
            scroll_dead_zone=settings.scroll_dead_zone,
            scroll_max_range=settings.scroll_max_range,
            scroll_min_speed=settings.scroll_min_speed,
            scroll_max_speed=settings.scroll_max_speed,
            scroll_smoothing_alpha=settings.scroll_smoothing_alpha,
            scroll_finger_separation_max=settings.scroll_finger_separation_max,
            scroll_finger_parallel_tolerance_degrees=settings.scroll_finger_parallel_tolerance_degrees,
            scroll_finger_extension_min=settings.scroll_finger_extension_min,
            scroll_max_tilt_degrees=settings.scroll_max_tilt_degrees,
            scroll_release_slack=settings.scroll_release_slack,
            scroll_exit_grace_seconds=settings.scroll_exit_grace_seconds,
        )
