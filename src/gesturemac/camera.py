"""Built-in camera capture (OpenCV), latest-frame only."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Protocol

import numpy as np

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None  # type: ignore[assignment]


class VideoCaptureFactory(Protocol):
    def __call__(self, device_index: int) -> object: ...


@dataclass(frozen=True)
class CameraStats:
    generation: int
    frames_captured: int
    fps: float
    running: bool
    last_error: str | None


StatsListener = Callable[[CameraStats], None]


@dataclass
class _SharedState:
    generation: int = 0
    frames_captured: int = 0
    fps: float = 0.0
    running: bool = False
    last_error: str | None = None
    latest_frame: np.ndarray | None = None


class CameraManager:
    """
    Captures from the built-in camera on a background thread.
    Keeps only the latest frame; drops older samples while the loop runs.
    """

    def __init__(
        self,
        device_index: int = 0,
        on_stats: StatsListener | None = None,
        capture_factory: VideoCaptureFactory | None = None,
    ) -> None:
        self._device_index = device_index
        self._on_stats = on_stats
        self._capture_factory = capture_factory or self._default_capture_factory
        self._lock = Lock()
        self._state = _SharedState()
        self._stop = Event()
        self._thread: Thread | None = None
        self._preview_enabled = False
        self._window_name = "GestureMac Diagnostics"

    @staticmethod
    def _default_capture_factory(device_index: int) -> object:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        return cv2.VideoCapture(device_index)

    def set_preview_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._preview_enabled = enabled
        if not enabled and cv2 is not None:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass

    def stats(self) -> CameraStats:
        with self._lock:
            s = self._state
            return CameraStats(
                generation=s.generation,
                frames_captured=s.frames_captured,
                fps=s.fps,
                running=s.running,
                last_error=s.last_error,
            )

    def start(self, generation: int) -> None:
        self.stop()
        self._stop.clear()
        with self._lock:
            self._state = _SharedState(
                generation=generation,
                running=True,
            )
        self._thread = Thread(
            target=self._run,
            name="GestureMac-Camera",
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
            was_running = self._state.running
            gen = self._state.generation
            self._state.running = False
            self._state.latest_frame = None
        if was_running:
            with self._lock:
                self._state.generation = gen
            self._emit_stats()
        self.set_preview_enabled(False)

    def get_latest_frame(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            frame = self._state.latest_frame
            gen = self._state.generation
            if frame is None:
                return gen, None
            return gen, frame.copy()

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state.last_error = message
            self._state.running = False
        self._emit_stats()

    def _emit_stats(self) -> None:
        if self._on_stats:
            self._on_stats(self.stats())

    def _run(self) -> None:
        try:
            cap = self._capture_factory(self._device_index)
        except Exception as exc:  # noqa: BLE001
            self._set_error(str(exc))
            return

        if cv2 is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

        opened = bool(getattr(cap, "isOpened", lambda: False)())
        if not opened:
            self._set_error("Could not open camera device")
            if hasattr(cap, "release"):
                cap.release()
            return

        fps_window_start = time.monotonic()
        frames_in_window = 0

        try:
            while not self._stop.is_set():
                read = getattr(cap, "read", None)
                if read is None:
                    self._set_error("Capture device does not support read()")
                    break
                ok, frame = read()
                if not ok or frame is None:
                    self._set_error("Failed to read camera frame")
                    break

                now = time.monotonic()
                frames_in_window += 1
                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    fps = frames_in_window / elapsed
                    fps_window_start = now
                    frames_in_window = 0
                    with self._lock:
                        self._state.fps = fps
                    self._emit_stats()

                with self._lock:
                    if self._stop.is_set():
                        break
                    self._state.latest_frame = frame
                    self._state.frames_captured += 1
                    show_preview = self._preview_enabled

                if show_preview and cv2 is not None:
                    cv2.imshow(self._window_name, frame)
                    cv2.waitKey(1)
        finally:
            if hasattr(cap, "release"):
                cap.release()
            with self._lock:
                self._state.running = False
            self._emit_stats()
