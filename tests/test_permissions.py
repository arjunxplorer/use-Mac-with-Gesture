from gesturemac import permissions
from gesturemac.permissions import CameraAuthStatus, camera_status_label


def test_camera_status_label_known() -> None:
    assert camera_status_label(CameraAuthStatus.AUTHORIZED) == "authorized"
    assert camera_status_label(CameraAuthStatus.DENIED) == "denied"


def test_ensure_camera_denied(monkeypatch) -> None:
    monkeypatch.setattr(permissions, "camera_status", lambda: CameraAuthStatus.DENIED)
    ok, msg = permissions.ensure_camera_for_capture(request_if_needed=False)
    assert ok is False
    assert "denied" in msg.lower()


def test_ensure_camera_authorized(monkeypatch) -> None:
    monkeypatch.setattr(
        permissions,
        "camera_status",
        lambda: CameraAuthStatus.AUTHORIZED,
    )
    ok, msg = permissions.ensure_camera_for_capture()
    assert ok is True
    assert "authorized" in msg.lower()
