from gesturemac.gestures.commands import (
    ClickPointer,
    MovePointer,
    PressPointer,
    ReleasePointer,
    ScrollWheel,
)
from gesturemac.input.system_input import NullInputBackend, SystemInputController


def test_move_blocked_without_accessibility(monkeypatch) -> None:
    backend = NullInputBackend()
    controller = SystemInputController(backend=backend)
    monkeypatch.setattr(
        "gesturemac.input.system_input.accessibility_trusted",
        lambda: False,
    )
    assert controller.execute(MovePointer(dx=10, dy=10)) is False
    assert backend.moves == []


def test_move_executes_with_accessibility(monkeypatch) -> None:
    backend = NullInputBackend()
    controller = SystemInputController(backend=backend)
    monkeypatch.setattr(
        "gesturemac.input.system_input.accessibility_trusted",
        lambda: True,
    )
    monkeypatch.setattr(
        "gesturemac.input.system_input.get_cursor_position",
        lambda: (100.0, 100.0),
    )
    monkeypatch.setattr(
        "gesturemac.input.system_input.clamp_to_displays",
        lambda x, y, _displays=None: (x, y),
    )
    assert controller.execute(MovePointer(dx=5, dy=-3)) is True
    assert backend.moves == [(105.0, 97.0)]


def test_click_counts_forwarded(monkeypatch) -> None:
    backend = NullInputBackend()
    controller = SystemInputController(backend=backend)
    monkeypatch.setattr(
        "gesturemac.input.system_input.accessibility_trusted",
        lambda: True,
    )
    controller.execute(ClickPointer(click_count=1))
    controller.execute(ClickPointer(click_count=2))
    assert backend.clicks == [1, 2]


def test_scroll_forwarded_and_gated(monkeypatch) -> None:
    backend = NullInputBackend()
    controller = SystemInputController(backend=backend)
    trusted = {"value": False}
    monkeypatch.setattr(
        "gesturemac.input.system_input.accessibility_trusted",
        lambda: trusted["value"],
    )
    assert controller.execute(ScrollWheel(vertical_pixels=-3)) is False
    trusted["value"] = True
    assert controller.execute(ScrollWheel(vertical_pixels=-3)) is True
    assert backend.scrolls == [-3]


def test_moves_while_pressed_are_drags(monkeypatch) -> None:
    backend = NullInputBackend()
    controller = SystemInputController(backend=backend)
    monkeypatch.setattr("gesturemac.input.system_input.accessibility_trusted", lambda: True)
    monkeypatch.setattr(
        "gesturemac.input.system_input.get_cursor_position",
        lambda: (100.0, 100.0),
    )
    monkeypatch.setattr(
        "gesturemac.input.system_input.clamp_to_displays",
        lambda x, y, _displays=None: (x, y),
    )
    controller.execute(PressPointer())
    controller.execute(PressPointer())
    controller.execute(MovePointer(dx=5, dy=0))
    controller.execute(ReleasePointer())
    controller.execute(ReleasePointer())
    controller.execute(MovePointer(dx=1, dy=0))
    assert backend.presses == 1
    assert backend.releases == 1
    assert backend.drag_moves == [(105.0, 100.0)]
    assert backend.moves == [(101.0, 100.0)]


def test_release_held_button_on_stop(monkeypatch) -> None:
    backend = NullInputBackend()
    controller = SystemInputController(backend=backend)
    monkeypatch.setattr("gesturemac.input.system_input.accessibility_trusted", lambda: True)
    assert controller.release_held_button() is False
    controller.execute(PressPointer())
    assert controller.release_held_button() is True
    assert controller.button_down is False
    assert backend.releases == 1


def test_click_blocked_without_accessibility(monkeypatch) -> None:
    backend = NullInputBackend()
    controller = SystemInputController(backend=backend)
    monkeypatch.setattr(
        "gesturemac.input.system_input.accessibility_trusted",
        lambda: False,
    )
    assert controller.execute(ClickPointer(click_count=1)) is False
    assert backend.clicks == []
