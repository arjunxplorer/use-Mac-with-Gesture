"""MediaPipe hand tracking and background inference loop."""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Protocol

import numpy as np

from gesturemac.camera import CameraManager
from gesturemac.coordinator import AppCoordinator
from gesturemac.gestures.engine import GestureEngine, GestureMode
from gesturemac.input.system_input import SystemInputController
from gesturemac.landmarks import HandFrame, HandJoint, HandPoint, mediapipe_joint_map
from gesturemac.settings import DEFAULT_SETTINGS_DIR

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None  # type: ignore[assignment]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"
PREVIEW_WINDOW = "GestureMac Diagnostics"

# BGR colors for index / thumb / middle tips
_DEBUG_COLORS: dict[HandJoint, tuple[int, int, int]] = {
    HandJoint.INDEX_TIP: (0, 255, 0),
    HandJoint.THUMB_TIP: (0, 165, 255),
    HandJoint.MIDDLE_TIP: (255, 0, 255),
}


@dataclass(frozen=True)
class TrackingStats:
    generation: int
    processed_frames: int
    inference_fps: float
    hand_detected: bool
    running: bool
    last_error: str | None
    keypoint_confidence: float | None


StatsListener = Callable[[TrackingStats], None]

MODEL_INPUT_SIZE = 256


@dataclass(frozen=True)
class _CropMeta:
    """Maps model square input back to mirrored camera coordinates."""

    full_width: int
    full_height: int
    crop_x: int
    crop_y: int
    crop_side: int


def prepare_model_input(frame_bgr: np.ndarray) -> tuple[np.ndarray, _CropMeta]:
    """
    Selfie mirror + center square crop + resize for MediaPipe.

    Non-square camera frames otherwise trigger landmark projection warnings and
    often fail to detect hands.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed")
    mirrored = cv2.flip(frame_bgr, 1)
    height, width = mirrored.shape[:2]
    side = min(height, width)
    crop_y = (height - side) // 2
    crop_x = (width - side) // 2
    crop = mirrored[crop_y : crop_y + side, crop_x : crop_x + side]
    resized = cv2.resize(
        crop,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    meta = _CropMeta(
        full_width=width,
        full_height=height,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_side=side,
    )
    return rgb, meta


def map_landmark_to_frame_norm(
    lm_x: float,
    lm_y: float,
    meta: _CropMeta,
) -> tuple[float, float]:
    px = meta.crop_x + lm_x * meta.crop_side
    py = meta.crop_y + lm_y * meta.crop_side
    return px / meta.full_width, py / meta.full_height


class HandLandmarkerLike(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> HandFrame | None: ...


def default_model_path() -> Path:
    return DEFAULT_SETTINGS_DIR / MODEL_FILENAME


def ensure_model_file(path: Path | None = None) -> Path:
    target = path or default_model_path()
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, target)  # noqa: S310
    return target


class MediaPipeHandTracker:
    """Wraps MediaPipe Tasks HandLandmarker (num_hands=1)."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = ensure_model_file(model_path)
        self._last_timestamp_ms = 0
        self._landmarker = self._create_landmarker()
        self._joint_index = mediapipe_joint_map()

    def _create_landmarker(self) -> object:
        from mediapipe.tasks.python import vision  # type: ignore[import-untyped]
        from mediapipe.tasks import python as mp_python  # type: ignore[import-untyped]

        base = mp_python.BaseOptions(model_asset_path=str(self._model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        return vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> HandFrame | None:
        import mediapipe as mp  # type: ignore[import-untyped]

        rgb, meta = prepare_model_input(frame_bgr)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(timestamp * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)  # type: ignore[union-attr]
        if not result.hand_landmarks:
            return None
        landmarks = result.hand_landmarks[0]
        joints: dict[HandJoint, HandPoint] = {}
        for joint, index in self._joint_index.items():
            lm = landmarks[index]
            visibility = lm.visibility if lm.visibility is not None else 1.0
            presence = lm.presence if getattr(lm, "presence", None) is not None else 1.0
            confidence = float(min(visibility, presence))
            nx, ny = map_landmark_to_frame_norm(float(lm.x), float(lm.y), meta)
            joints[joint] = HandPoint(x=nx, y=ny, confidence=confidence)
        return HandFrame(
            timestamp=timestamp,
            joints=joints,
            aspect=meta.full_width / meta.full_height,
        )


def mean_keypoint_confidence(frame: HandFrame | None) -> float | None:
    if frame is None:
        return None
    tips = (
        HandJoint.INDEX_TIP,
        HandJoint.THUMB_TIP,
        HandJoint.MIDDLE_TIP,
    )
    values = [frame.joints[j].confidence for j in tips if j in frame.joints]
    if not values:
        return None
    return sum(values) / len(values)


def draw_debug_overlay(frame_bgr: np.ndarray, hand: HandFrame | None) -> np.ndarray:
    if cv2 is None:
        return frame_bgr
    out = cv2.flip(frame_bgr, 1)
    if hand is None:
        return out
    h, w = out.shape[:2]
    for joint, color in _DEBUG_COLORS.items():
        pt = hand.get(joint)
        if pt is None or pt.confidence < 0.5:
            continue
        cx = int(pt.x * w)
        cy = int(pt.y * h)
        cv2.circle(out, (cx, cy), 8, color, 2)
        cv2.putText(
            out,
            joint.value.split("_")[0][:3],
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


@dataclass
class _WorkerState:
    generation: int = 0
    processed_frames: int = 0
    inference_fps: float = 0.0
    hand_detected: bool = False
    running: bool = False
    last_error: str | None = None
    keypoint_confidence: float | None = None
    last_hand: HandFrame | None = None


class TrackingWorker:
    """
    Polls the camera for new frames, runs hand tracking off the main thread,
    and optionally shows a diagnostics preview with fingertip markers.
    """

    def __init__(
        self,
        camera: CameraManager,
        coordinator: AppCoordinator,
        tracker: HandLandmarkerLike | None = None,
        on_stats: StatsListener | None = None,
        gesture_engine: GestureEngine | None = None,
        input_controller: SystemInputController | None = None,
    ) -> None:
        self._camera = camera
        self._coordinator = coordinator
        self._tracker = tracker
        self._owns_tracker = tracker is None
        self._on_stats = on_stats
        self._gesture_engine = gesture_engine or GestureEngine()
        self._input = input_controller or SystemInputController()
        self._lock = Lock()
        self._state = _WorkerState()
        self._stop = Event()
        self._thread: Thread | None = None
        self._preview_enabled = False
        self._last_camera_frame_count = -1

    def _tracker_instance(self) -> HandLandmarkerLike:
        if self._tracker is None:
            self._tracker = MediaPipeHandTracker()
        return self._tracker

    def set_preview_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._preview_enabled = enabled
        if not enabled and cv2 is not None:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass

    def stats(self) -> TrackingStats:
        with self._lock:
            s = self._state
            return TrackingStats(
                generation=s.generation,
                processed_frames=s.processed_frames,
                inference_fps=s.inference_fps,
                hand_detected=s.hand_detected,
                running=s.running,
                last_error=s.last_error,
                keypoint_confidence=s.keypoint_confidence,
            )

    def get_last_hand_frame(self) -> HandFrame | None:
        with self._lock:
            return self._state.last_hand

    @property
    def gesture_engine(self) -> GestureEngine:
        return self._gesture_engine

    @property
    def gesture_mode(self) -> GestureMode:
        return self._gesture_engine.mode

    def start(self, generation: int) -> None:
        self.stop()
        self._stop.clear()
        self._last_camera_frame_count = -1
        with self._lock:
            self._state = _WorkerState(generation=generation, running=True)
        self._thread = Thread(
            target=self._run,
            name="GestureMac-Tracking",
            daemon=True,
        )
        self._thread.start()
        self._emit_stats()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            gen = self._state.generation
            self._state.running = False
            self._state.hand_detected = False
            self._state.last_hand = None
            self._state.generation = gen
        self.set_preview_enabled(False)
        if self._owns_tracker and isinstance(self._tracker, MediaPipeHandTracker):
            self._tracker.close()
            self._tracker = None
        self._gesture_engine.reset()
        self._emit_stats()

    def _emit_stats(self) -> None:
        if self._on_stats:
            self._on_stats(self.stats())

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state.last_error = message
            self._state.running = False
        self._emit_stats()

    def _run(self) -> None:
        try:
            tracker = self._tracker_instance()
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
            return

        fps_start = time.monotonic()
        inferences_in_window = 0

        while not self._stop.is_set():
            snap = self._coordinator.snapshot()
            if snap.state.value != "active":
                time.sleep(0.05)
                continue
            if not self._coordinator.is_current(snap.generation):
                time.sleep(0.05)
                continue

            cam_stats = self._camera.stats()
            if cam_stats.frames_captured == self._last_camera_frame_count:
                time.sleep(0.005)
                continue
            self._last_camera_frame_count = cam_stats.frames_captured

            gen, frame = self._camera.get_latest_frame()
            if frame is None or not self._coordinator.is_current(gen):
                continue

            timestamp = time.monotonic()
            try:
                hand = tracker.detect(frame, timestamp)
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                break

            inferences_in_window += 1
            elapsed = timestamp - fps_start
            if elapsed >= 1.0:
                inference_fps = inferences_in_window / elapsed
                fps_start = timestamp
                inferences_in_window = 0
                with self._lock:
                    self._state.inference_fps = inference_fps
                self._emit_stats()

            conf = mean_keypoint_confidence(hand)
            with self._lock:
                self._state.processed_frames += 1
                self._state.hand_detected = hand is not None
                self._state.last_hand = hand
                self._state.keypoint_confidence = conf
                show_preview = self._preview_enabled
            self._emit_stats()

            if hand is None:
                commands = self._gesture_engine.note_hand_missing(timestamp)
            else:
                commands = self._gesture_engine.process(hand)
            if self._coordinator.is_current(gen):
                try:
                    for command in commands:
                        self._input.execute(command)
                except Exception as exc:  # noqa: BLE001
                    self._set_error(f"Input failed: {exc}")
                    break

            if show_preview and cv2 is not None:
                display = draw_debug_overlay(frame, hand)
                cv2.imshow(PREVIEW_WINDOW, display)
                cv2.waitKey(1)

        with self._lock:
            self._state.running = False
        self._emit_stats()
