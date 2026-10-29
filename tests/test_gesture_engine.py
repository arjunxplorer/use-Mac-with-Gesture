from gesturemac.gestures.commands import ClickPointer, MovePointer, PressPointer, ReleasePointer
from gesturemac.gestures.config import GestureConfig
from gesturemac.gestures.engine import GestureEngine, GestureMode
from gesturemac.landmarks import HandFrame, HandJoint, HandPoint

OPEN_THUMB = (0.35, 0.60)


def _pt(x: float, y: float, c: float = 0.95) -> HandPoint:
    return HandPoint(x=x, y=y, confidence=c)


def pointer_pose(
    timestamp: float,
    index_x: float = 0.5,
    index_y: float = 0.35,
    thumb: tuple[float, float] = OPEN_THUMB,
    aspect: float = 1.0,
) -> HandFrame:
    return HandFrame(
        timestamp=timestamp,
        aspect=aspect,
        joints={
            HandJoint.WRIST: _pt(0.5, 0.75),
            HandJoint.MIDDLE_MCP: _pt(0.5, 0.60),
            HandJoint.THUMB_TIP: _pt(*thumb),
            HandJoint.INDEX_TIP: _pt(index_x, index_y),
            HandJoint.INDEX_PIP: _pt(0.5, 0.55),
            HandJoint.MIDDLE_TIP: _pt(0.52, 0.72),
            HandJoint.MIDDLE_PIP: _pt(0.51, 0.70),
            HandJoint.RING_TIP: _pt(0.48, 0.73),
            HandJoint.RING_PIP: _pt(0.49, 0.71),
            HandJoint.PINKY_TIP: _pt(0.485, 0.735),
            HandJoint.PINKY_PIP: _pt(0.48, 0.72),
        },
    )


def pinched(timestamp: float, index_x: float = 0.5, index_y: float = 0.35) -> HandFrame:
    return pointer_pose(timestamp, index_x=index_x, index_y=index_y, thumb=(index_x + 0.005, index_y))


def _engine(**overrides: float) -> GestureEngine:
    params = dict(
        stable_pose_seconds=0.0,
        smoothing_alpha=1.0,
        pointer_min_gain=1.0,
        pointer_max_gain=1.0,
        jitter_dead_zone=0.0,
        reanchor_settle_seconds=0.0,
        double_click_interval=0.5,
    )
    params.update(overrides)
    return GestureEngine(GestureConfig(**params))


def _enter_pointer(engine: GestureEngine, t: float = 0.0) -> float:
    engine.process(pointer_pose(t))
    engine.process(pointer_pose(t + 0.03))
    assert engine.mode is GestureMode.POINTER
    return t + 0.03


def _pinch_and_release(engine: GestureEngine, t: float) -> list:
    engine.process(pinched(t))
    engine.process(pinched(t + 0.03))
    return engine.process(pointer_pose(t + 0.06))


def test_idle_without_hand() -> None:
    engine = GestureEngine(GestureConfig(stable_pose_seconds=0.0))
    assert engine.process(None) == []


def test_pointer_requires_stable_pose() -> None:
    engine = GestureEngine(GestureConfig(stable_pose_seconds=0.12))
    assert engine.process(pointer_pose(0.0)) == []
    assert engine.mode is GestureMode.IDLE
    engine.process(pointer_pose(0.13))
    assert engine.mode is GestureMode.POINTER


def test_pointer_follows_finger_direction() -> None:
    engine = _engine(pointer_sensitivity=1000.0)
    t = _enter_pointer(engine)
    engine.process(pointer_pose(t + 0.03))
    engine.process(pointer_pose(t + 0.06))
    moves = engine.process(pointer_pose(t + 0.09, index_x=0.52, index_y=0.33))
    assert len(moves) == 1
    move = moves[0]
    assert isinstance(move, MovePointer)
    assert move.dx > 0  # finger right -> cursor right
    assert move.dy < 0  # finger up -> cursor up (Quartz Y grows downward)


def test_wide_frame_axes_move_equally() -> None:
    engine = _engine(pointer_sensitivity=1000.0)
    aspect = 16 / 9
    t = 0.0
    for i in range(4):
        engine.process(pointer_pose(t + i * 0.03, aspect=aspect))
    t += 0.12
    # Same physical distance: 0.02 of the short side = 0.02 / aspect of the width.
    move = engine.process(
        pointer_pose(t, index_x=0.5 + 0.02 / aspect, index_y=0.35 - 0.02, aspect=aspect)
    )[0]
    assert isinstance(move, MovePointer)
    # Whole-point output: the axes may differ by at most the 1-point truncation.
    assert abs(abs(move.dx) - abs(move.dy)) <= 1.0


def curled(timestamp: float, index_y: float = 0.5) -> HandFrame:
    """Index curled back toward the palm: no longer the pointer pose."""
    frame = pointer_pose(timestamp, index_y=index_y)
    joints = dict(frame.joints)
    joints[HandJoint.INDEX_TIP] = _pt(0.5, 0.62)
    return HandFrame(timestamp=timestamp, joints=joints)


def test_curling_finger_stops_cursor_immediately() -> None:
    engine = _engine(stable_pose_seconds=0.12)
    for i in range(6):
        engine.process(pointer_pose(i * 0.03))
    assert engine.mode is GestureMode.POINTER
    # Still within the 120 ms window, so the mode hasn't switched yet, but nothing may move.
    assert engine.process(curled(0.18)) == []
    assert engine.process(curled(0.21)) == []
    assert engine.mode is GestureMode.POINTER


def test_repointing_after_clutch_does_not_jump() -> None:
    engine = _engine(stable_pose_seconds=0.0)
    _enter_pointer(engine)
    engine.process(curled(0.06))
    engine.process(curled(0.09))
    assert engine.mode is GestureMode.IDLE
    # Hand repositioned far away, then points again: first frames only re-anchor.
    assert engine.process(pointer_pose(0.12, index_x=0.8)) == []
    assert engine.process(pointer_pose(0.15, index_x=0.8)) == []


def test_reset_clears_state() -> None:
    engine = _engine()
    _enter_pointer(engine)
    engine.reset()
    assert engine.mode is GestureMode.IDLE


def test_hand_missing_clears_mode() -> None:
    engine = _engine(hand_loss_seconds=0.2)
    _enter_pointer(engine)
    engine.note_hand_missing(0.30)
    assert engine.mode is GestureMode.IDLE


def test_pinch_release_emits_one_click() -> None:
    engine = _engine()
    t = _enter_pointer(engine)
    assert engine.process(pinched(t + 0.03)) == []
    assert engine.mode is GestureMode.PINCHING
    for i in range(10):
        assert engine.process(pinched(t + 0.06 + i * 0.03)) == []
    commands = engine.process(pointer_pose(t + 0.5))
    assert commands == [ClickPointer(click_count=1)]
    assert engine.mode is GestureMode.POINTER


def test_no_pointer_movement_while_pinching() -> None:
    engine = _engine()
    t = _enter_pointer(engine)
    engine.process(pinched(t + 0.03))
    moved = pointer_pose(t + 0.06, index_x=0.6, thumb=(0.605, 0.35))
    assert engine.process(moved) == []


def test_hand_loss_mid_pinch_no_click() -> None:
    engine = _engine()
    t = _enter_pointer(engine)
    engine.process(pinched(t + 0.03))
    engine.note_hand_missing(t + 0.06)
    assert engine.mode is GestureMode.IDLE
    commands = engine.process(pointer_pose(t + 0.09))
    assert not any(isinstance(c, ClickPointer) for c in commands)


def test_two_quick_pinches_double_click() -> None:
    engine = _engine()
    t = _enter_pointer(engine)
    first = _pinch_and_release(engine, t + 0.03)
    second = _pinch_and_release(engine, t + 0.15)
    assert first == [ClickPointer(click_count=1)]
    assert second == [ClickPointer(click_count=2)]


def test_slow_pinches_are_separate_clicks() -> None:
    engine = _engine(double_click_interval=0.3)
    t = _enter_pointer(engine)
    _pinch_and_release(engine, t + 0.03)
    second = _pinch_and_release(engine, t + 1.0)
    assert second == [ClickPointer(click_count=1)]


def test_third_pinch_starts_new_click() -> None:
    engine = _engine()
    t = _enter_pointer(engine)
    _pinch_and_release(engine, t + 0.03)
    _pinch_and_release(engine, t + 0.15)
    third = _pinch_and_release(engine, t + 0.27)
    assert third == [ClickPointer(click_count=1)]


def _start_drag(engine: GestureEngine, t: float) -> float:
    """Pinch and hold past drag_hold_seconds (0.3 in these tests); returns the press time."""
    assert engine.process(pinched(t)) == []
    assert engine.process(pinched(t + 0.15)) == []
    assert engine.process(pinched(t + 0.31)) == [PressPointer()]
    assert engine.mode is GestureMode.DRAGGING
    return t + 0.31


def test_held_pinch_presses_drags_and_releases() -> None:
    engine = _engine(drag_hold_seconds=0.3, pointer_sensitivity=1000.0)
    t = _start_drag(engine, _enter_pointer(engine) + 0.03)
    engine.process(pinched(t + 0.03))
    moves = engine.process(pinched(t + 0.06, index_x=0.55))
    assert len(moves) == 1 and isinstance(moves[0], MovePointer) and moves[0].dx > 0
    released = engine.process(pointer_pose(t + 0.09, index_x=0.55))
    assert released == [ReleasePointer()]
    assert engine.mode is GestureMode.POINTER
    assert engine.clicks_emitted == 0
    assert engine.drags_emitted == 1


def test_quick_pinch_still_clicks_not_drags() -> None:
    engine = _engine(drag_hold_seconds=0.3)
    t = _enter_pointer(engine)
    assert _pinch_and_release(engine, t + 0.03) == [ClickPointer(click_count=1)]
    assert engine.drags_emitted == 0


def test_click_after_drag_is_single() -> None:
    engine = _engine(drag_hold_seconds=0.3)
    t = _enter_pointer(engine)
    _pinch_and_release(engine, t + 0.03)
    t = _start_drag(engine, t + 0.15)
    engine.process(pointer_pose(t + 0.03))
    assert _pinch_and_release(engine, t + 0.06) == [ClickPointer(click_count=1)]


def test_brief_hand_dropout_keeps_drag() -> None:
    engine = _engine(drag_hold_seconds=0.3, hand_loss_seconds=0.2)
    t = _start_drag(engine, _enter_pointer(engine) + 0.03)
    assert engine.note_hand_missing(t + 0.1) == []
    assert engine.mode is GestureMode.DRAGGING
    assert engine.process(pinched(t + 0.13)) == []
    assert engine.mode is GestureMode.DRAGGING


def test_hand_lost_during_drag_releases() -> None:
    engine = _engine(drag_hold_seconds=0.3, hand_loss_seconds=0.2)
    t = _start_drag(engine, _enter_pointer(engine) + 0.03)
    assert engine.note_hand_missing(t + 0.1) == []
    assert engine.note_hand_missing(t + 0.25) == [ReleasePointer()]
    assert engine.mode is GestureMode.IDLE
    assert engine.process(None) == []
