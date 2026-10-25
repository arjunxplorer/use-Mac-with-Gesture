import numpy as np
import pytest

from gesturemac.landmarks import HandFrame, HandJoint, HandPoint
from gesturemac.tracking import (
    draw_debug_overlay,
    map_landmark_to_frame_norm,
    mean_keypoint_confidence,
    prepare_model_input,
    _CropMeta,
)


class StubTracker:
    def __init__(self, hand: HandFrame | None) -> None:
        self.hand = hand
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> HandFrame | None:
        self.calls += 1
        return self.hand


def _hand(conf: float = 0.9) -> HandFrame:
    joints = {
        HandJoint.INDEX_TIP: HandPoint(0.5, 0.5, conf),
        HandJoint.THUMB_TIP: HandPoint(0.4, 0.5, conf),
        HandJoint.MIDDLE_TIP: HandPoint(0.55, 0.5, conf),
    }
    return HandFrame(timestamp=1.0, joints=joints)


def test_mean_keypoint_confidence() -> None:
    assert mean_keypoint_confidence(_hand(0.8)) == pytest.approx(0.8)
    assert mean_keypoint_confidence(None) is None


def test_prepare_model_input_square_rgb() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb, meta = prepare_model_input(frame)
    assert rgb.shape == (256, 256, 3)
    assert meta.crop_side == 480


def test_map_landmark_to_frame_norm() -> None:
    meta = _CropMeta(full_width=640, full_height=480, crop_x=80, crop_y=0, crop_side=480)
    nx, ny = map_landmark_to_frame_norm(0.5, 0.5, meta)
    assert nx == pytest.approx(0.5)
    assert ny == pytest.approx(0.5)


def test_draw_debug_overlay_changes_pixels() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_debug_overlay(frame, _hand())
    assert not np.array_equal(out, frame)


def test_tracking_worker_processes_frames() -> None:
    from gesturemac.camera import CameraManager
    from gesturemac.coordinator import AppCoordinator
    from gesturemac.tracking import TrackingWorker

    class OneFrameCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, *_a, **_k) -> None:
            return None

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.released:
                return False, None
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    coord = AppCoordinator()
    cam = CameraManager(capture_factory=lambda _i: OneFrameCapture())
    tracker = StubTracker(_hand())
    worker = TrackingWorker(cam, coord, tracker=tracker)
    coord.start()
    gen = coord.generation
    cam.start(gen)
    worker.start(gen)

    import time

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and tracker.calls < 1:
        time.sleep(0.05)

    assert tracker.calls >= 1
    assert worker.stats().hand_detected is True
    worker.stop()
    cam.stop()
