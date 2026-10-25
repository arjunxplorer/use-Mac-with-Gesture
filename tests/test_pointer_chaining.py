"""End-to-end: engine + input controller chaining strokes like lifting a mouse."""

import pytest

from gesturemac.gestures.config import GestureConfig
from gesturemac.gestures.engine import GestureEngine
from gesturemac.input.system_input import SystemInputController
from test_gesture_engine import curled, pointer_pose


class SimulatedCursor:
    """Backend whose moves change the position that get_cursor_position reports."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.clicks: list[int] = []

    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def move_pointer(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    def click(self, click_count: int) -> None:
        self.clicks.append(click_count)

    def scroll(self, vertical_pixels: int) -> None:
        pass


@pytest.fixture
def rig(monkeypatch):
    cursor = SimulatedCursor(500.0, 500.0)
    monkeypatch.setattr("gesturemac.input.system_input.accessibility_trusted", lambda: True)
    monkeypatch.setattr("gesturemac.input.system_input.get_cursor_position", cursor.position)
    monkeypatch.setattr(
        "gesturemac.input.system_input.clamp_to_displays",
        lambda x, y, _d=None: (x, y),
    )
    engine = GestureEngine(
        GestureConfig(
            stable_pose_seconds=0.12,
            smoothing_alpha=1.0,
            pointer_min_gain=1.0,
            pointer_max_gain=1.0,
            pointer_sensitivity=1000.0,
            jitter_dead_zone=0.0,
        )
    )
    controller = SystemInputController(backend=cursor)

    class Rig:
        t = 0.0

        def feed(self, frame_fn, **kwargs) -> None:
            frame = frame_fn(self.t, **kwargs)
            for command in engine.process(frame):
                controller.execute(command)
            self.t += 0.033

        def missing(self) -> None:
            for command in engine.note_hand_missing(self.t):
                controller.execute(command)
            self.t += 0.033

    return Rig(), cursor


def _stroke(rig, start: float, end: float, steps: int = 10) -> None:
    for i in range(steps + 1):
        rig.feed(pointer_pose, index_x=start + (end - start) * i / steps)


def _settle(rig, x: float, frames: int = 8) -> None:
    for _ in range(frames):
        rig.feed(pointer_pose, index_x=x)


def _curl_and_return(rig, back_to: float, frames: int = 8) -> None:
    for _ in range(frames):
        rig.feed(curled)
    _settle(rig, back_to)


def test_strokes_chain_after_curl(rig) -> None:
    r, cursor = rig
    _settle(r, 0.40)
    _stroke(r, 0.40, 0.55)
    after_first = cursor.x
    assert after_first > 500.0

    _curl_and_return(r, 0.40)
    assert cursor.x == pytest.approx(after_first)  # repositioning the hand moves nothing

    _stroke(r, 0.40, 0.55)
    assert cursor.x > after_first + 100  # second stroke continues from the new spot


def test_strokes_chain_after_hand_leaves_view(rig) -> None:
    r, cursor = rig
    _settle(r, 0.40)
    _stroke(r, 0.40, 0.55)
    after_first = cursor.x
    for _ in range(10):
        r.missing()
    _settle(r, 0.40)
    assert cursor.x == pytest.approx(after_first)
    _stroke(r, 0.40, 0.55)
    assert cursor.x > after_first + 100


def test_many_chained_strokes_keep_accumulating(rig) -> None:
    r, cursor = rig
    _settle(r, 0.40)
    positions = []
    for _ in range(4):
        _stroke(r, 0.40, 0.55)
        positions.append(cursor.x)
        _curl_and_return(r, 0.40)
    assert positions == sorted(positions)
    assert positions[-1] - positions[0] > 300
