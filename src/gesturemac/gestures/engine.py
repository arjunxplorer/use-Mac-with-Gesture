"""Pure gesture state machine: pointer movement, pinch click/double-click/drag, joystick scroll."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from gesturemac.gestures.commands import (
    ClickPointer,
    GestureCommand,
    MovePointer,
    PressPointer,
    ReleasePointer,
)
from gesturemac.gestures.config import GestureConfig
from gesturemac.gestures.scroll import JoystickScroller, ScrollPhase, is_side_scroll_pose
from gesturemac.landmarks import HandFrame, HandJoint, HandPoint


class GestureMode(str, Enum):
    IDLE = "idle"
    POINTER = "pointer"
    PINCHING = "pinching"
    DRAGGING = "dragging"
    SCROLLING = "scrolling"


def _dist(a: HandPoint, b: HandPoint, sx: float = 1.0, sy: float = 1.0) -> float:
    return math.hypot((a.x - b.x) * sx, (a.y - b.y) * sy)


def _point_ok(frame: HandFrame, joint: HandJoint, min_confidence: float) -> HandPoint | None:
    pt = frame.get(joint)
    if pt is None or pt.confidence < min_confidence:
        return None
    return pt


def finger_extended(
    tip: HandPoint,
    pip: HandPoint,
    wrist: HandPoint,
    *,
    ratio: float,
    sx: float = 1.0,
    sy: float = 1.0,
) -> bool:
    return _dist(tip, wrist, sx, sy) > _dist(pip, wrist, sx, sy) * ratio


def pointer_gain(speed: float, config: GestureConfig) -> float:
    """Acceleration curve: min gain when slow, max gain when fast, smoothstep in between."""
    low, high = config.pointer_min_gain, config.pointer_max_gain
    if high <= low:
        return low
    span = config.pointer_accel_fast_speed - config.pointer_accel_slow_speed
    if span <= 0:
        return high if speed >= config.pointer_accel_fast_speed else low
    t = max(0.0, min(1.0, (speed - config.pointer_accel_slow_speed) / span))
    t = t * t * (3.0 - 2.0 * t)
    return low + (high - low) * t


@dataclass
class GestureEngine:
    config: GestureConfig = field(default_factory=GestureConfig)

    mode: GestureMode = GestureMode.IDLE
    last_pinch_ratio: float | None = None
    clicks_emitted: int = 0
    drags_emitted: int = 0
    _candidate: GestureMode | None = None
    _candidate_since: float | None = None
    _smooth_x: float | None = None
    _smooth_y: float | None = None
    _anchor_x: float = 0.0
    _anchor_y: float = 0.0
    _anchor_ts: float = 0.0
    _settle_until: float = 0.0
    _speed: float | None = None
    _remainder_x: float = 0.0
    _remainder_y: float = 0.0
    _last_timestamp: float | None = None
    _last_hand_at: float | None = None
    _last_click_at: float | None = None
    _last_click_count: int = 0
    _moved_since_click: float = 0.0
    _pinch_started_at: float = 0.0
    _pinch_seen_at: float = 0.0
    _scroller: JoystickScroller = field(default_factory=JoystickScroller)

    @property
    def scroll_phase(self) -> ScrollPhase:
        return self._scroller.phase

    @property
    def scroll_displacement(self) -> float | None:
        return self._scroller.displacement

    def reset(self) -> None:
        self.mode = GestureMode.IDLE
        self.last_pinch_ratio = None
        self._candidate = None
        self._candidate_since = None
        self._reset_anchors()
        self._last_timestamp = None
        self._last_hand_at = None
        self._last_click_at = None
        self._last_click_count = 0
        self._moved_since_click = 0.0

    def process(self, frame: HandFrame | None) -> list[GestureCommand]:
        if frame is None:
            return self._on_hand_missing(None)

        ts = frame.timestamp
        if self._last_timestamp is not None:
            if ts < self._last_timestamp:
                return []
            if ts - self._last_timestamp > self.config.frame_gap_reset_seconds:
                self._reset_anchors()
        self._last_timestamp = ts
        self._last_hand_at = ts

        pinch = self._pinch_ratio(frame)
        self.last_pinch_ratio = pinch

        if self.mode is GestureMode.PINCHING:
            return self._continue_pinch(pinch, ts)
        if self.mode is GestureMode.DRAGGING:
            return self._continue_drag(frame, pinch, ts)

        pose = self._classify_pose(frame)
        self._update_mode(pose, ts)

        if self.mode is GestureMode.SCROLLING:
            # Pointer and click paths are unreachable here, so both stay suppressed while scrolling.
            if pose is not GestureMode.SCROLLING:
                self._scroller.hold(ts, self.config)
                return []
            return list(self._scroller.update(frame, self.config))

        if self.mode is not GestureMode.POINTER:
            return []

        if self._index_folded(frame):
            # Clutch: a folded index stops the cursor at once (the stable-pose window only
            # delays entering a mode). Checked before pinch so a fist with the thumb over the
            # folded finger can't register as a click.
            self._reset_anchors()
            return []
        if pose is GestureMode.SCROLLING:
            # Forming the scroll pose: freeze the cursor and don't let a resting thumb start a pinch.
            self._reset_anchors()
            return []
        if pinch is not None and pinch < self.config.pinch_down_threshold:
            self.mode = GestureMode.PINCHING
            self._pinch_started_at = ts
            self._reset_anchors()
            return []
        if pinch is not None and pinch < self.config.pinch_up_threshold:
            # Thumb is closing on the index tip: hold the cursor still so the click lands on target.
            self._reset_anchors()
            return []
        if pose is not GestureMode.POINTER:
            self._reset_anchors()
            return []

        return self._pointer_moves(frame, ts)

    def note_hand_missing(self, timestamp: float) -> list[GestureCommand]:
        return self._on_hand_missing(timestamp)

    def _on_hand_missing(self, timestamp: float | None) -> list[GestureCommand]:
        self.last_pinch_ratio = None
        if self.mode is GestureMode.DRAGGING:
            self._reset_anchors()
            # Ride out a brief tracking dropout so the dragged item isn't dropped mid-move.
            grace = self.config.hand_loss_seconds
            if timestamp is not None and timestamp - self._pinch_seen_at <= grace:
                return []
            return self._end_drag(GestureMode.IDLE)
        if self.mode is GestureMode.PINCHING:
            self.mode = GestureMode.IDLE
            self._candidate = None
        self._reset_anchors()
        if (
            timestamp is not None
            and self._last_hand_at is not None
            and timestamp - self._last_hand_at > self.config.hand_loss_seconds
        ):
            self.mode = GestureMode.IDLE
            self._candidate = None
        return []

    def _continue_pinch(self, pinch: float | None, ts: float) -> list[GestureCommand]:
        if pinch is None:
            self.mode = GestureMode.IDLE
            self._candidate = None
            self._reset_anchors()
            return []
        if pinch <= self.config.pinch_up_threshold:
            if ts - self._pinch_started_at < self.config.drag_hold_seconds:
                return []
            self.mode = GestureMode.DRAGGING
            self._pinch_seen_at = ts
            self._reset_anchors()
            # A drag breaks any click chain: the next pinch is a fresh single click.
            self._last_click_count = 0
            self.drags_emitted += 1
            return [PressPointer()]
        self.mode = GestureMode.POINTER
        self._reset_anchors()
        return [self._make_click(ts)]

    def _continue_drag(
        self, frame: HandFrame, pinch: float | None, ts: float
    ) -> list[GestureCommand]:
        if pinch is None:
            self._reset_anchors()
            if ts - self._pinch_seen_at > self.config.hand_loss_seconds:
                return self._end_drag(GestureMode.IDLE)
            return []
        self._pinch_seen_at = ts
        if pinch > self.config.pinch_up_threshold:
            return self._end_drag(GestureMode.POINTER)
        return self._pointer_moves(frame, ts)

    def _end_drag(self, next_mode: GestureMode) -> list[GestureCommand]:
        self.mode = next_mode
        self._candidate = None
        self._reset_anchors()
        return [ReleasePointer()]

    def _make_click(self, ts: float) -> ClickPointer:
        count = 1
        if (
            self._last_click_count == 1
            and self._last_click_at is not None
            and ts - self._last_click_at <= self.config.double_click_interval
            and self._moved_since_click <= self.config.double_click_distance
        ):
            count = 2
        self._last_click_at = ts
        self._last_click_count = count
        self._moved_since_click = 0.0
        self.clicks_emitted += 1
        return ClickPointer(click_count=count)

    def _pinch_ratio(self, frame: HandFrame) -> float | None:
        mc = self.config.min_confidence
        wrist = _point_ok(frame, HandJoint.WRIST, mc)
        middle_mcp = _point_ok(frame, HandJoint.MIDDLE_MCP, mc)
        thumb_tip = _point_ok(frame, HandJoint.THUMB_TIP, mc)
        index_tip = _point_ok(frame, HandJoint.INDEX_TIP, mc)
        if not (wrist and middle_mcp and thumb_tip and index_tip):
            return None
        sx, sy = frame.axis_scale()
        palm = _dist(wrist, middle_mcp, sx, sy)
        if palm < self.config.min_palm_scale:
            return None
        return _dist(thumb_tip, index_tip, sx, sy) / palm

    def _index_folded(self, frame: HandFrame) -> bool:
        mc = self.config.min_confidence
        wrist = _point_ok(frame, HandJoint.WRIST, mc)
        tip = _point_ok(frame, HandJoint.INDEX_TIP, mc)
        pip = _point_ok(frame, HandJoint.INDEX_PIP, mc)
        if not (wrist and tip and pip):
            return False
        sx, sy = frame.axis_scale()
        return not finger_extended(tip, pip, wrist, ratio=self.config.index_fold_ratio, sx=sx, sy=sy)

    def _reset_anchors(self) -> None:
        self._smooth_x = None
        self._smooth_y = None
        self._speed = None
        self._remainder_x = 0.0
        self._remainder_y = 0.0
        self._scroller.reset()

    def _classify_pose(self, frame: HandFrame) -> GestureMode:
        mc = self.config.min_confidence
        wrist = _point_ok(frame, HandJoint.WRIST, mc)
        index_tip = _point_ok(frame, HandJoint.INDEX_TIP, mc)
        index_pip = _point_ok(frame, HandJoint.INDEX_PIP, mc)
        middle_tip = _point_ok(frame, HandJoint.MIDDLE_TIP, mc)
        middle_pip = _point_ok(frame, HandJoint.MIDDLE_PIP, mc)
        ring_tip = _point_ok(frame, HandJoint.RING_TIP, mc)
        ring_pip = _point_ok(frame, HandJoint.RING_PIP, mc)
        pinky_tip = _point_ok(frame, HandJoint.PINKY_TIP, mc)
        pinky_pip = _point_ok(frame, HandJoint.PINKY_PIP, mc)

        if not (
            wrist and index_tip and index_pip and middle_tip and middle_pip
            and ring_tip and ring_pip and pinky_tip and pinky_pip
        ):
            return GestureMode.IDLE

        ratio = self.config.finger_extend_ratio
        sx, sy = frame.axis_scale()
        index_out = finger_extended(index_tip, index_pip, wrist, ratio=ratio, sx=sx, sy=sy)
        middle_out = finger_extended(middle_tip, middle_pip, wrist, ratio=ratio, sx=sx, sy=sy)
        ring_out = finger_extended(ring_tip, ring_pip, wrist, ratio=ratio, sx=sx, sy=sy)
        pinky_out = finger_extended(pinky_tip, pinky_pip, wrist, ratio=ratio, sx=sx, sy=sy)

        if ring_out or pinky_out or not index_out:
            return GestureMode.IDLE
        if not middle_out:
            return GestureMode.POINTER
        # Two fingers out only scroll when held together sideways; a V-sign or upright pair is idle.
        holding = self.mode is GestureMode.SCROLLING
        if is_side_scroll_pose(frame, self.config, holding=holding):
            return GestureMode.SCROLLING
        return GestureMode.IDLE

    def _update_mode(self, desired: GestureMode, timestamp: float) -> None:
        if desired == self.mode:
            self._candidate = None
            self._candidate_since = None
            return

        if self._candidate != desired:
            self._candidate = desired
            self._candidate_since = timestamp
            return

        assert self._candidate_since is not None
        if timestamp - self._candidate_since < self.config.stable_pose_seconds:
            return

        self._reset_anchors()
        self.mode = desired
        self._candidate = None
        self._candidate_since = None

    def _pointer_moves(self, frame: HandFrame, timestamp: float) -> list[GestureCommand]:
        tip = _point_ok(frame, HandJoint.INDEX_TIP, self.config.min_confidence)
        if tip is None:
            return []

        # Work in units where the image's short side = 1 so both axes move at the same rate.
        sx, sy = frame.axis_scale()
        x = tip.x * sx
        y = tip.y * sy

        cfg = self.config
        if self._smooth_x is None or self._smooth_y is None:
            # Fresh anchor at the finger's current spot: the cursor continues from wherever it
            # is now, never from where the previous stroke ended.
            self._smooth_x, self._smooth_y = x, y
            self._anchor_x, self._anchor_y, self._anchor_ts = x, y, timestamp
            self._settle_until = timestamp + cfg.reanchor_settle_seconds
            return []

        alpha = cfg.smoothing_alpha
        self._smooth_x = alpha * x + (1.0 - alpha) * self._smooth_x
        self._smooth_y = alpha * y + (1.0 - alpha) * self._smooth_y

        if timestamp < self._settle_until:
            self._anchor_x, self._anchor_y, self._anchor_ts = self._smooth_x, self._smooth_y, timestamp
            return []

        # Landmarks are already mirrored, and image Y grows downward like Quartz screen Y.
        move_x = self._smooth_x - self._anchor_x
        move_y = self._smooth_y - self._anchor_y
        distance = math.hypot(move_x, move_y)
        if distance < cfg.jitter_dead_zone:
            # Anchor stays put, so slow deliberate drift adds up until it clears the dead zone.
            return []

        dt = timestamp - self._anchor_ts
        self._anchor_x, self._anchor_y, self._anchor_ts = self._smooth_x, self._smooth_y, timestamp
        speed = distance / dt if dt > 0 else 0.0
        if self._speed is None:
            self._speed = speed
        else:
            self._speed = cfg.speed_smoothing * speed + (1.0 - cfg.speed_smoothing) * self._speed

        gain = pointer_gain(self._speed, cfg)
        limit = cfg.max_delta_points
        fx = max(-limit, min(limit, move_x * cfg.pointer_sensitivity * gain + self._remainder_x))
        fy = max(-limit, min(limit, move_y * cfg.pointer_sensitivity * gain + self._remainder_y))
        # Emit whole points and carry the fraction, so sub-point motion is never lost.
        dx, dy = math.trunc(fx), math.trunc(fy)
        self._remainder_x, self._remainder_y = fx - dx, fy - dy
        if dx == 0 and dy == 0:
            return []
        self._moved_since_click += math.hypot(dx, dy)
        return [MovePointer(dx=float(dx), dy=float(dy))]
