import time
from typing import Any

import numpy as np

from gesturemac.camera import CameraManager


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        time.sleep(0.02)
        return True, frame

    def release(self) -> None:
        self.released = True


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_camera_counts_frames_and_stops() -> None:
    cap = FakeCapture([_frame(), _frame(), _frame()])
    stats_seen: list[int] = []

    def on_stats(stats: object) -> None:
        from gesturemac.camera import CameraStats

        assert isinstance(stats, CameraStats)
        stats_seen.append(stats.frames_captured)

    cam = CameraManager(capture_factory=lambda _i: cap, on_stats=on_stats)
    cam.start(generation=1)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        st = cam.stats()
        if st.frames_captured >= 2 and not st.running:
            break
        if st.frames_captured >= 3:
            break
        time.sleep(0.05)

    cam.stop()
    final = cam.stats()
    assert final.frames_captured >= 2
    assert cap.released is True


def test_get_latest_frame_copy() -> None:
    cap = FakeCapture([_frame()])
    cam = CameraManager(capture_factory=lambda _i: cap)
    cam.start(generation=7)
    time.sleep(0.15)
    gen, frame = cam.get_latest_frame()
    cam.stop()
    assert gen == 7
    assert frame is not None
    assert frame.shape == (480, 640, 3)
