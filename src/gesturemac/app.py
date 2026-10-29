"""Menu bar UI (rumps)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import rumps

from gesturemac.camera import CameraManager, CameraStats
from gesturemac.coordinator import AppCoordinator, AppState
from gesturemac.notify import notify
from gesturemac.permissions import (
    camera_status_label,
    ensure_camera_for_capture,
    open_privacy_accessibility_settings,
    open_privacy_camera_settings,
    snapshot as permission_snapshot,
)
from gesturemac.gestures.config import GestureConfig
from gesturemac.gestures.engine import GestureEngine, GestureMode
from gesturemac.input.system_input import SystemInputController
from gesturemac.settings import Settings, load, save
from gesturemac.tracking import TrackingStats, TrackingWorker


def _menu_title(state: AppState, *, error: bool = False) -> str:
    if error:
        return "GM!"
    return "GM●" if state is AppState.ACTIVE else "GM○"


def _status_label(state: AppState) -> str:
    return "Status: Active" if state is AppState.ACTIVE else "Status: Paused"


class GestureMacApp(rumps.App):
    def __init__(
        self,
        coordinator: AppCoordinator | None = None,
        settings_path: Path | None = None,
        camera: CameraManager | None = None,
        tracking: TrackingWorker | None = None,
    ) -> None:
        self._coordinator = coordinator or AppCoordinator()
        self._settings_path = settings_path
        self._settings = load(settings_path)
        self._camera = camera or CameraManager(on_stats=self._on_camera_stats)
        self._gesture_engine = GestureEngine(GestureConfig.from_settings(self._settings))
        self._input_controller = SystemInputController()
        self._tracking = tracking or TrackingWorker(
            self._camera,
            self._coordinator,
            on_stats=self._on_tracking_stats,
            gesture_engine=self._gesture_engine,
            input_controller=self._input_controller,
        )
        self._camera_stats = self._camera.stats()
        self._tracking_stats = self._tracking.stats()
        self._session_error: str | None = None
        super().__init__(
            _menu_title(self._coordinator.state),
            quit_button=None,
        )
        self._stats_timer = rumps.Timer(self._refresh_menu_timer, 1.0)
        self._coordinator.add_listener(self._on_coordinator_change)
        self._stats_timer.start()
        self._rebuild_menu()

    def _on_camera_stats(self, stats: CameraStats) -> None:
        self._camera_stats = stats
        if stats.last_error and self._coordinator.state is AppState.ACTIVE:
            self._session_error = stats.last_error

    def _on_tracking_stats(self, stats: TrackingStats) -> None:
        self._tracking_stats = stats
        if stats.last_error and self._coordinator.state is AppState.ACTIVE:
            self._session_error = stats.last_error

    def _sync_diagnostics_preview(self) -> None:
        enabled = (
            self._settings.show_diagnostics
            and self._coordinator.state is AppState.ACTIVE
        )
        self._camera.set_preview_enabled(False)
        self._tracking.set_preview_enabled(enabled)

    def _on_coordinator_change(self, state: AppState, generation: int) -> None:
        if state is AppState.ACTIVE:
            self._session_error = None
            ok, message = ensure_camera_for_capture(request_if_needed=True)
            if not ok:
                self._session_error = message
                self._coordinator.pause()
                notify("GestureMac", "Camera required", message)
            else:
                perms = permission_snapshot()
                if not perms.accessibility_trusted:
                    notify(
                        "GestureMac",
                        "Accessibility",
                        "Enable Accessibility for Terminal/Cursor (or this app) to move the pointer.",
                    )
                self._settings = load(self._settings_path)
                self._gesture_engine.config = GestureConfig.from_settings(self._settings)
                self._gesture_engine.reset()
                self._sync_diagnostics_preview()
                self._camera.start(generation)
                self._tracking.start(generation)
        else:
            self._tracking.stop()
            self._camera.stop()
            self._session_error = None
        self._apply_state(state)

    def _apply_state(self, state: AppState) -> None:
        self.title = _menu_title(state, error=bool(self._session_error))
        self._rebuild_menu()

    def _refresh_menu_timer(self, _timer: rumps.Timer) -> None:
        if self._coordinator.state is AppState.ACTIVE:
            self._rebuild_menu()

    def _rebuild_menu(self) -> None:
        state = self._coordinator.state
        toggle_title = "Pause" if state is AppState.ACTIVE else "Start"
        perms = permission_snapshot()

        settings_menu = rumps.MenuItem("Settings")
        s = self._settings
        settings_menu.add(rumps.MenuItem(f"Pointer sensitivity: {s.pointer_sensitivity:g}"))
        settings_menu.add(
            rumps.MenuItem(
                f"Pointer gain: {s.pointer_min_gain:g}× slow → {s.pointer_max_gain:g}× fast",
            )
        )
        settings_menu.add(rumps.MenuItem(f"Smoothing alpha: {s.smoothing_alpha:g}"))
        settings_menu.add(
            rumps.MenuItem(
                f"Scroll speed: {s.scroll_min_speed:g}–{s.scroll_max_speed:g} px/s "
                f"(dead zone {s.scroll_dead_zone:g})",
            )
        )
        settings_menu.add(
            rumps.MenuItem(
                f"Invert scroll: {'yes' if s.invert_scroll else 'no'}",
                callback=self.toggle_invert_scroll,
            )
        )
        settings_menu.add(rumps.MenuItem(f"Pinch down: {s.pinch_down_threshold:g}"))
        settings_menu.add(rumps.MenuItem(f"Pinch up: {s.pinch_up_threshold:g}"))
        settings_menu.add(rumps.MenuItem(f"Drag hold: {s.drag_hold_seconds:g} s"))
        settings_menu.add(
            rumps.MenuItem(
                f"Diagnostics preview: {'on' if s.show_diagnostics else 'off'}",
                callback=self.toggle_diagnostics,
            )
        )
        settings_menu.add(None)
        settings_menu.add(rumps.MenuItem("Open settings file", callback=self.open_settings_file))
        settings_menu.add(rumps.MenuItem("Reset to defaults", callback=self.reset_settings))

        self.menu.clear()
        self.menu.add(rumps.MenuItem(_status_label(state)))
        if self._session_error:
            self.menu.add(rumps.MenuItem(f"Error: {self._session_error}"))
        self.menu.add(
            rumps.MenuItem(
                f"Camera: {camera_status_label(perms.camera)}",
            )
        )
        self.menu.add(
            rumps.MenuItem(
                f"Accessibility: {'granted' if perms.accessibility_trusted else 'not granted'}",
            )
        )
        if state is AppState.ACTIVE:
            cs = self._camera.stats()
            ts = self._tracking.stats()
            self.menu.add(rumps.MenuItem(f"Frames captured: {cs.frames_captured}"))
            self.menu.add(rumps.MenuItem(f"Capture FPS: {cs.fps:.1f}"))
            self.menu.add(rumps.MenuItem(f"Tracking passes: {ts.processed_frames}"))
            self.menu.add(rumps.MenuItem(f"Hand detected: {'yes' if ts.hand_detected else 'no'}"))
            engine = self._gesture_engine
            mode_label = engine.mode.value
            if engine.mode is GestureMode.SCROLLING:
                mode_label += f" ({engine.scroll_phase.value})"
                offset = engine.scroll_displacement
                if offset is not None:
                    mode_label += f", offset {offset:+.3f}"
            self.menu.add(rumps.MenuItem(f"Gesture mode: {mode_label}"))
            pinch = engine.last_pinch_ratio
            self.menu.add(
                rumps.MenuItem(
                    f"Pinch ratio: {pinch:.2f} (down < {engine.config.pinch_down_threshold:g})"
                    if pinch is not None
                    else "Pinch ratio: –",
                )
            )
            self.menu.add(rumps.MenuItem(f"Clicks sent: {engine.clicks_emitted}"))
            self.menu.add(rumps.MenuItem(f"Drags started: {engine.drags_emitted}"))
            self.menu.add(
                rumps.MenuItem(
                    "Pointer input: "
                    + ("on" if perms.accessibility_trusted else "off (grant Accessibility)"),
                )
            )
            self.menu.add(rumps.MenuItem(f"Tracking FPS: {ts.inference_fps:.1f}"))
            if ts.keypoint_confidence is not None:
                self.menu.add(
                    rumps.MenuItem(
                        f"Tip confidence (avg): {ts.keypoint_confidence:.2f}",
                    )
                )
            if cs.last_error:
                self.menu.add(rumps.MenuItem(f"Capture: {cs.last_error}"))
            if ts.last_error:
                self.menu.add(rumps.MenuItem(f"Tracking: {ts.last_error}"))
        self.menu.add(None)
        self.menu.add(rumps.MenuItem(toggle_title, callback=self.toggle_tracking))
        self.menu.add(rumps.MenuItem("Open Camera Privacy…", callback=self.open_camera_privacy))
        self.menu.add(
            rumps.MenuItem(
                "Open Accessibility Privacy…",
                callback=self.open_accessibility_privacy,
            )
        )
        self.menu.add(settings_menu)
        self.menu.add(None)
        self.menu.add(rumps.MenuItem("Quit", callback=self.quit_app))

    def toggle_tracking(self, _sender: rumps.MenuItem) -> None:
        self._settings = load(self._settings_path)
        self._coordinator.toggle()

    def toggle_diagnostics(self, _sender: rumps.MenuItem) -> None:
        self._settings.show_diagnostics = not self._settings.show_diagnostics
        save(self._settings, self._settings_path)
        self._sync_diagnostics_preview()
        self._rebuild_menu()

    def toggle_invert_scroll(self, _sender: rumps.MenuItem) -> None:
        self._settings.invert_scroll = not self._settings.invert_scroll
        save(self._settings, self._settings_path)
        self._gesture_engine.config.invert_scroll = self._settings.invert_scroll
        self._rebuild_menu()

    def open_camera_privacy(self, _sender: rumps.MenuItem) -> None:
        open_privacy_camera_settings()

    def open_accessibility_privacy(self, _sender: rumps.MenuItem) -> None:
        open_privacy_accessibility_settings()

    def open_settings_file(self, _sender: rumps.MenuItem) -> None:
        path = save(self._settings, self._settings_path)
        subprocess.run(["open", "-R", str(path)], check=False)

    def reset_settings(self, _sender: rumps.MenuItem) -> None:
        self._settings = Settings()
        save(self._settings, self._settings_path)
        self._rebuild_menu()
        notify("GestureMac", "Settings reset", "Defaults restored.")

    def quit_app(self, _sender: rumps.MenuItem) -> None:
        self._stats_timer.stop()
        self._tracking.stop()
        self._camera.stop()
        rumps.quit_application()
