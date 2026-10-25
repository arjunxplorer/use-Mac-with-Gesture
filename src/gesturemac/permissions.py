"""macOS Camera and Accessibility permission helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import IntEnum
from threading import Event
from typing import Callable

try:
    import AVFoundation  # type: ignore[import-untyped]
    from ApplicationServices import (  # type: ignore[import-untyped]
        AXIsProcessTrusted,
        AXIsProcessTrustedWithOptions,
    )
    from Foundation import NSDictionary  # type: ignore[import-untyped]

    _HAS_PYOBJC = True
except ImportError:
    _HAS_PYOBJC = False


class CameraAuthStatus(IntEnum):
    NOT_DETERMINED = 0
    RESTRICTED = 1
    DENIED = 2
    AUTHORIZED = 3
    UNAVAILABLE = -1


@dataclass(frozen=True)
class PermissionSnapshot:
    camera: CameraAuthStatus
    accessibility_trusted: bool

    @property
    def camera_ok(self) -> bool:
        return self.camera is CameraAuthStatus.AUTHORIZED


def _media_type_video() -> object:
    return AVFoundation.AVMediaTypeVideo


def camera_status() -> CameraAuthStatus:
    if not _HAS_PYOBJC:
        return CameraAuthStatus.UNAVAILABLE
    raw = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        _media_type_video()
    )
    try:
        return CameraAuthStatus(int(raw))
    except ValueError:
        return CameraAuthStatus.UNAVAILABLE


def accessibility_trusted(*, prompt: bool = False) -> bool:
    if not _HAS_PYOBJC:
        return False
    if prompt:
        options = NSDictionary.dictionaryWithObject_forKey_(  # type: ignore[attr-defined]
            True,
            "AXTrustedCheckOptionPrompt",
        )
        return bool(AXIsProcessTrustedWithOptions(options))
    return bool(AXIsProcessTrusted())


def request_camera_access(
    timeout: float = 120.0,
    completion: Callable[[bool], None] | None = None,
) -> bool:
    """
    Request camera access if not yet determined. Returns True only when authorized.
    """
    status = camera_status()
    if status is CameraAuthStatus.AUTHORIZED:
        if completion:
            completion(True)
        return True
    if status in (CameraAuthStatus.DENIED, CameraAuthStatus.RESTRICTED):
        if completion:
            completion(False)
        return False
    if not _HAS_PYOBJC:
        if completion:
            completion(False)
        return False

    done = Event()
    result: list[bool] = [False]

    def handler(granted: bool) -> None:
        result[0] = bool(granted)
        if completion:
            completion(result[0])
        done.set()

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        _media_type_video(),
        handler,
    )
    done.wait(timeout=timeout)
    return result[0] and camera_status() is CameraAuthStatus.AUTHORIZED


def ensure_camera_for_capture(*, request_if_needed: bool = True) -> tuple[bool, str]:
    """
    Returns (ok, user-facing message). Does not open System Settings automatically
    except via the Accessibility prompt when explicitly requested elsewhere.
    """
    status = camera_status()
    if status is CameraAuthStatus.AUTHORIZED:
        return True, "Camera authorized"
    if status is CameraAuthStatus.NOT_DETERMINED and request_if_needed:
        if request_camera_access():
            return True, "Camera authorized"
        status = camera_status()
    if status is CameraAuthStatus.DENIED:
        return False, "Camera access denied — enable GestureMac in System Settings → Privacy → Camera"
    if status is CameraAuthStatus.RESTRICTED:
        return False, "Camera access restricted by system policy"
    if status is CameraAuthStatus.UNAVAILABLE:
        return False, "Camera permission APIs unavailable"
    return False, "Camera access not granted"


def open_privacy_camera_settings() -> None:
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"],
        check=False,
    )


def open_privacy_accessibility_settings() -> None:
    subprocess.run(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        ],
        check=False,
    )


def snapshot(*, accessibility_prompt: bool = False) -> PermissionSnapshot:
    return PermissionSnapshot(
        camera=camera_status(),
        accessibility_trusted=accessibility_trusted(prompt=accessibility_prompt),
    )


def camera_status_label(status: CameraAuthStatus) -> str:
    labels = {
        CameraAuthStatus.NOT_DETERMINED: "not requested",
        CameraAuthStatus.RESTRICTED: "restricted",
        CameraAuthStatus.DENIED: "denied",
        CameraAuthStatus.AUTHORIZED: "authorized",
        CameraAuthStatus.UNAVAILABLE: "unavailable",
    }
    return labels.get(status, "unknown")
