"""Pointer acceleration, sub-point accumulation, clutching, and tracking resets."""

import pytest

from gesturemac.gestures.commands import ClickPointer, MovePointer
from gesturemac.gestures.config import GestureConfig
from gesturemac.gestures.engine import GestureEngine, GestureMode, pointer_gain
from gesturemac.landmarks import HandFrame, HandJoint

from test_gesture_engine import _pt, curled, pointer_pose
from test_joystick_scroll import side_pose

FRAME = 1 / 30


def make_engine(**overrides) -> GestureEngine:
    params = dict(
        stable_pose_seconds=0.0,
        smoothing_alpha=1.0,
        pointer_sensitivity=1000.0,
        pointer_min_gain=1.0,
        pointer_max_gain=1.0,
        jitter_dead_zone=0.0,
        reanchor_settle_seconds=0.0,
    )
    params.update(overrides)
    return GestureEngine(GestureConfig(**params))


def total_dx(commands: list) -> float:
    return sum(c.dx for c in commands if isinstance(c, MovePointer))


def run_stroke(engine, x0: float, x1: float, t0: float, frames: int) -> tuple[list, float]:
    """Index finger moves from x0 to x1 over `frames` frames; returns commands and end time."""
    out = []
    t = t0
    for i in range(frames + 1):
        out += engine.process(pointer_pose(t, index_x=x0 + (x1 - x0) * i / frames))
        t += FRAME
    return out, t


def settle(engine, x: float, t: float, frames: int = 4) -> float:
    for _ in range(frames):
        engine.process(pointer_pose(t, index_x=x))
        t += FRAME
    return t


# --- acceleration curve -------------------------------------------------------

ACCEL = GestureConfig(
    pointer_min_gain=0.4,
    pointer_max_gain=3.0,
    pointer_accel_slow_speed=0.1,
    pointer_accel_fast_speed=1.0,
)


def test_gain_floor_below_slow_speed() -> None:
    assert pointer_gain(0.0, ACCEL) == pytest.approx(0.4)
    assert pointer_gain(0.1, ACCEL) == pytest.approx(0.4)


def test_gain_ceiling_above_fast_speed() -> None:
    assert pointer_gain(1.0, ACCEL) == pytest.approx(3.0)
    assert pointer_gain(5.0, ACCEL) == pytest.approx(3.0)


def test_gain_rises_monotonically_between() -> None:
    gains = [pointer_gain(0.1 + i * 0.09, ACCEL) for i in range(11)]
    assert gains == sorted(gains)
    assert gains[5] == pytest.approx((0.4 + 3.0) / 2)  # smoothstep midpoint


def test_equal_gains_disable_acceleration() -> None:
    flat = GestureConfig(pointer_min_gain=1.5, pointer_max_gain=1.5)
    assert pointer_gain(0.0, flat) == pointer_gain(10.0, flat) == 1.5


def test_same_distance_fast_stroke_travels_farther() -> None:
    accel = dict(pointer_min_gain=0.4, pointer_max_gain=3.0)
    slow_engine, fast_engine = make_engine(**accel), make_engine(**accel)
    settle(slow_engine, 0.40, 0.0)
    settle(fast_engine, 0.40, 0.0)
    # Same 0.15-unit hand movement: 2 s (0.075 u/s) vs 0.15 s (1 u/s).
    slow_cmds, _ = run_stroke(slow_engine, 0.40, 0.55, 4 * FRAME, frames=60)
    fast_cmds, _ = run_stroke(fast_engine, 0.40, 0.55, 4 * FRAME, frames=5)
    slow_px, fast_px = total_dx(slow_cmds), total_dx(fast_cmds)
    assert slow_px == pytest.approx(0.15 * 1000 * 0.4, abs=2)  # precision gain
    assert fast_px > slow_px * 4


def test_comfortable_sweep_crosses_macbook_screen() -> None:
    """Defaults: one quick 0.35-unit sweep (about 1/3 of the camera height) covers 1512 pt."""
    engine = GestureEngine(GestureConfig(stable_pose_seconds=0.0))
    t = settle(engine, 0.30, 0.0, frames=6)
    cmds, t = run_stroke(engine, 0.30, 0.65, t, frames=10)  # ~0.33 s
    for _ in range(6):  # hold still while the smoothing filter catches up
        cmds += engine.process(pointer_pose(t, index_x=0.65))
        t += FRAME
    assert total_dx(cmds) >= 1512


def test_precise_slow_move_stays_small_with_defaults() -> None:
    """Defaults: creeping 0.02 units (~1 cm of hand travel) over a second moves only a few points."""
    engine = GestureEngine(GestureConfig(stable_pose_seconds=0.0))
    t = settle(engine, 0.40, 0.0, frames=6)
    cmds, _ = run_stroke(engine, 0.40, 0.42, t, frames=30)
    assert 0 < total_dx(cmds) <= 15


# --- sub-point accumulation ---------------------------------------------------

def test_sub_point_motion_accumulates() -> None:
    engine = make_engine()
    t = settle(engine, 0.40, 0.0)
    # 0.0004 units/frame * 1000 = 0.4 pt per frame: below one point every single frame.
    commands = []
    for i in range(1, 26):
        commands += engine.process(pointer_pose(t, index_x=0.40 + 0.0004 * i))
        t += FRAME
    assert total_dx(commands) == pytest.approx(10, abs=1)  # 25 * 0.4 = 10 pt, none lost
    assert all(c.dx == int(c.dx) for c in commands)


def test_tremor_inside_dead_zone_does_not_move() -> None:
    engine = make_engine(jitter_dead_zone=0.001)
    t = settle(engine, 0.40, 0.0)
    commands = []
    for i in range(30):
        commands += engine.process(pointer_pose(t, index_x=0.40 + (0.0004 if i % 2 else -0.0004)))
        t += FRAME
    assert commands == []


def test_slow_drift_still_escapes_dead_zone() -> None:
    engine = make_engine(jitter_dead_zone=0.001)
    t = settle(engine, 0.40, 0.0)
    commands = []
    for i in range(1, 30):
        commands += engine.process(pointer_pose(t, index_x=0.40 + 0.0003 * i))
        t += FRAME
    assert total_dx(commands) > 5


# --- clutch -------------------------------------------------------------------

def test_folding_index_stops_cursor_immediately() -> None:
    engine = make_engine(stable_pose_seconds=0.12)
    t = settle(engine, 0.40, 0.0, frames=6)
    assert engine.mode is GestureMode.POINTER
    for _ in range(3):  # inside the 120 ms window: mode unchanged, but no motion
        assert engine.process(curled(t)) == []
        t += FRAME
    assert engine.mode is GestureMode.POINTER


def test_reextending_resets_anchor_to_new_finger_position() -> None:
    engine = make_engine()
    t = settle(engine, 0.40, 0.0)
    first, t = run_stroke(engine, 0.40, 0.55, t, frames=10)
    for _ in range(4):
        engine.process(curled(t))
        t += FRAME
    # Re-extend with the hand moved far back to the left: anchors there, no backward jump.
    second = []
    for _ in range(3):
        second += engine.process(pointer_pose(t, index_x=0.30))
        t += FRAME
    assert second == []
    third, _ = run_stroke(engine, 0.30, 0.45, t, frames=10)
    assert total_dx(third) == pytest.approx(total_dx(first), abs=2)


def test_uncurling_finger_does_not_nudge_cursor() -> None:
    engine = make_engine(reanchor_settle_seconds=0.08)
    t = settle(engine, 0.40, 0.0)
    engine.process(curled(t))
    t += FRAME
    # The tip keeps travelling for ~2 frames while the finger straightens.
    commands = []
    for y in (0.40, 0.36, 0.35):
        commands += engine.process(pointer_pose(t, index_y=y))
        t += FRAME
    assert commands == []


def test_fist_with_thumb_over_index_is_not_a_click() -> None:
    engine = make_engine()
    t = settle(engine, 0.40, 0.0)
    # Folded index with the thumb resting on its tip: pinch-like distance, but a fist.
    fist = curled(t)
    joints = dict(fist.joints)
    joints[HandJoint.THUMB_TIP] = _pt(0.505, 0.62)
    commands = engine.process(HandFrame(timestamp=t, joints=joints))
    commands += engine.process(pointer_pose(t + FRAME))
    assert engine.mode is not GestureMode.PINCHING
    assert not any(isinstance(c, ClickPointer) for c in commands)


# --- tracking reset -----------------------------------------------------------

def test_no_jump_after_tracking_loss() -> None:
    engine = make_engine(hand_loss_seconds=0.2)
    t = settle(engine, 0.40, 0.0)
    for _ in range(3):  # brief dropout, mode survives
        engine.note_hand_missing(t)
        t += FRAME
    assert engine.mode is GestureMode.POINTER
    resumed = engine.process(pointer_pose(t, index_x=0.70))
    assert resumed == []
    assert engine.process(pointer_pose(t + FRAME, index_x=0.70)) == []


def test_no_jump_after_frame_gap() -> None:
    engine = make_engine(frame_gap_reset_seconds=0.25)
    t = settle(engine, 0.40, 0.0)
    assert engine.process(pointer_pose(t + 0.5, index_x=0.70)) == []


def test_no_jump_after_scroll_to_pointer_switch() -> None:
    engine = make_engine()
    t = 0.0
    for dy in (0.0, -0.02, -0.04, -0.06):
        engine.process(side_pose(t, dy=dy))
        t += FRAME
    assert engine.mode is GestureMode.SCROLLING
    commands = []
    for _ in range(3):
        commands += engine.process(pointer_pose(t, index_x=0.75))
        t += FRAME
    assert engine.mode is GestureMode.POINTER
    assert not any(isinstance(c, MovePointer) for c in commands)


def test_remainder_cleared_on_reset() -> None:
    engine = make_engine()
    t = settle(engine, 0.40, 0.0)
    engine.process(pointer_pose(t, index_x=0.4009))  # 0.9 pt pending, nothing emitted
    t += FRAME
    engine.note_hand_missing(t)
    t += FRAME
    t = settle(engine, 0.40, t, frames=2)
    out = engine.process(pointer_pose(t, index_x=0.4005))  # 0.5 pt: would emit 1 if 0.9 leaked
    assert total_dx(out) == 0
