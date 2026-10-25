"""Joystick-style scrolling: side-view two-finger pose, neutral center, position -> speed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from gesturemac.gestures.commands import ScrollWheel
from gesturemac.gestures.config import GestureConfig
from gesturemac.landmarks import HandFrame, HandJoint, HandPoint


MAX_FRAME_DT = 0.1


class ScrollPhase(str, Enum):
    INACTIVE = "inactive"
    ARMING = "arming"
    ACTIVE = "active"


def _point_ok(frame: HandFrame, joint: HandJoint, min_confidence: float) -> HandPoint | None:
    pt = frame.get(joint)
    if pt is None or pt.confidence < min_confidence:
        return None
    return pt


def _straightness(mcp: tuple[float, float], pip: tuple[float, float], tip: tuple[float, float]) -> float:
    """1.0 for a perfectly straight MCP -> PIP -> tip chain, lower as the finger bends."""
    path = math.dist(mcp, pip) + math.dist(pip, tip)
    return math.dist(mcp, tip) / path if path > 0 else 0.0


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cos))


def is_side_scroll_pose(frame: HandFrame, config: GestureConfig, *, holding: bool = False) -> bool:
    """Index and middle extended straight, pressed together, parallel, and roughly horizontal.

    Uses only geometry visible from the side (no palm-facing assumptions). Ring/pinky folding is
    checked by the caller's pose classifier. `holding` applies the looser limits used once
    scrolling has started (hysteresis), so a noisy frame doesn't pause or re-center the scroll.
    """
    slack = config.scroll_release_slack if holding else 1.0
    max_separation = config.scroll_finger_separation_max * slack
    max_parallel = min(90.0, config.scroll_finger_parallel_tolerance_degrees * slack)
    max_tilt = min(90.0, config.scroll_max_tilt_degrees * slack)
    min_extension = config.scroll_finger_extension_min - 0.2 * (slack - 1.0)
    mc = config.min_confidence
    joints = (
        HandJoint.WRIST,
        HandJoint.MIDDLE_MCP,
        HandJoint.INDEX_MCP,
        HandJoint.INDEX_PIP,
        HandJoint.INDEX_TIP,
        HandJoint.MIDDLE_PIP,
        HandJoint.MIDDLE_TIP,
    )
    pts = [_point_ok(frame, j, mc) for j in joints]
    if any(p is None for p in pts):
        return False
    sx, sy = frame.axis_scale()
    wrist, m_mcp, i_mcp, i_pip, i_tip, m_pip, m_tip = ((p.x * sx, p.y * sy) for p in pts)  # type: ignore[union-attr]

    palm = math.dist(wrist, m_mcp)
    if palm < config.min_palm_scale:
        return False

    if _straightness(i_mcp, i_pip, i_tip) < min_extension:
        return False
    if _straightness(m_mcp, m_pip, m_tip) < min_extension:
        return False

    if math.dist(i_tip, m_tip) / palm > max_separation:
        return False

    index_dir = (i_tip[0] - i_mcp[0], i_tip[1] - i_mcp[1])
    middle_dir = (m_tip[0] - m_mcp[0], m_tip[1] - m_mcp[1])
    if _angle_between(index_dir, middle_dir) > max_parallel:
        return False

    mean_dx = index_dir[0] + middle_dir[0]
    mean_dy = index_dir[1] + middle_dir[1]
    tilt = math.degrees(math.atan2(abs(mean_dy), abs(mean_dx)))
    return tilt <= max_tilt


def scroll_rate(displacement: float, config: GestureConfig) -> float:
    """Signed scroll pixels/second for a displacement (units) from the neutral center.

    Image Y grows downward, so fingers above center give a negative displacement and a negative
    rate, which scrolls the page down; fingers below center scroll it up.
    """
    magnitude = abs(displacement)
    if magnitude <= config.scroll_dead_zone:
        return 0.0
    span = config.scroll_max_range
    normalized = 1.0 if span <= 0 else min((magnitude - config.scroll_dead_zone) / span, 1.0)
    low, high = config.scroll_min_speed, max(config.scroll_min_speed, config.scroll_max_speed)
    return math.copysign(low + normalized * (high - low), displacement)


@dataclass
class JoystickScroller:
    phase: ScrollPhase = ScrollPhase.INACTIVE
    neutral_y: float | None = None
    smoothed_y: float | None = None
    _armed_at: float = 0.0
    _last_ts: float | None = None
    _invalid_since: float | None = None
    _accum: float = 0.0

    @property
    def displacement(self) -> float | None:
        if self.phase is not ScrollPhase.ACTIVE or self.neutral_y is None or self.smoothed_y is None:
            return None
        return self.smoothed_y - self.neutral_y

    def reset(self) -> None:
        self.phase = ScrollPhase.INACTIVE
        self.neutral_y = None
        self.smoothed_y = None
        self._last_ts = None
        self._invalid_since = None
        self._accum = 0.0

    def hold(self, timestamp: float, config: GestureConfig) -> None:
        """Pose briefly invalid: emit nothing now, and reset if it stays invalid past the grace period."""
        if self.phase is ScrollPhase.INACTIVE:
            return
        self._accum = 0.0
        self._last_ts = timestamp
        if self._invalid_since is None:
            self._invalid_since = timestamp
        if timestamp - self._invalid_since >= config.scroll_exit_grace_seconds:
            self.reset()

    def update(self, frame: HandFrame, config: GestureConfig) -> list[ScrollWheel]:
        mc = config.min_confidence
        index_tip = _point_ok(frame, HandJoint.INDEX_TIP, mc)
        middle_tip = _point_ok(frame, HandJoint.MIDDLE_TIP, mc)
        if index_tip is None or middle_tip is None:
            self.reset()
            return []

        ts = frame.timestamp
        _, sy = frame.axis_scale()
        y = (index_tip.y + middle_tip.y) / 2.0 * sy
        self._invalid_since = None

        if self.phase is ScrollPhase.INACTIVE or self.smoothed_y is None or self._last_ts is None:
            self.phase = ScrollPhase.ARMING
            self.smoothed_y = y
            self._armed_at = ts
            self._last_ts = ts
            self._accum = 0.0
            return []

        alpha = config.scroll_smoothing_alpha
        self.smoothed_y = alpha * y + (1.0 - alpha) * self.smoothed_y
        # Capped so a stalled frame can't turn into one big jump.
        dt = min(max(0.0, ts - self._last_ts), MAX_FRAME_DT)
        self._last_ts = ts

        if self.phase is ScrollPhase.ARMING:
            if ts - self._armed_at < config.scroll_activation_hold_seconds:
                return []
            # Neutral = the smoothed height right now, so displacement starts at exactly zero.
            self.neutral_y = self.smoothed_y
            self.phase = ScrollPhase.ACTIVE
            return []

        assert self.neutral_y is not None
        rate = scroll_rate(self.smoothed_y - self.neutral_y, config)
        if config.invert_scroll:
            rate = -rate
        if rate == 0.0 or (self._accum != 0.0 and math.copysign(1.0, rate) != math.copysign(1.0, self._accum)):
            # Fractions never carry across center, so crossing back can't fire a reverse pixel.
            self._accum = 0.0
        if rate == 0.0:
            return []

        self._accum += rate * dt
        whole = math.trunc(self._accum)
        if whole == 0:
            return []
        self._accum -= whole
        limit = config.max_scroll_pixels
        return [ScrollWheel(vertical_pixels=max(-limit, min(limit, whole)))]
