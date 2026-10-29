"""Post synthetic mouse events (Accessibility required)."""

from __future__ import annotations

from typing import Protocol

from gesturemac.gestures.commands import (
    ClickPointer,
    GestureCommand,
    MovePointer,
    PressPointer,
    ReleasePointer,
    ScrollWheel,
)
from gesturemac.input.cursor import clamp_to_displays, get_cursor_position
from gesturemac.permissions import accessibility_trusted


class InputBackend(Protocol):
    def move_pointer(self, x: float, y: float, *, dragging: bool = False) -> None: ...
    def click(self, click_count: int) -> None: ...
    def press(self) -> None: ...
    def release(self) -> None: ...
    def scroll(self, vertical_pixels: int) -> None: ...


def _post_mouse_event(event_type: int, x: float, y: float, click_state: int | None = None) -> None:
    import Quartz  # type: ignore[import-untyped]

    event = Quartz.CGEventCreateMouseEvent(
        None,
        event_type,
        Quartz.CGPoint(x, y),
        Quartz.kCGMouseButtonLeft,
    )
    if click_state is not None:
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_state)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class QuartzInputBackend:
    def move_pointer(self, x: float, y: float, *, dragging: bool = False) -> None:
        import Quartz  # type: ignore[import-untyped]

        # Apps only see a drag if moves arrive as LeftMouseDragged while the button is down.
        move_type = Quartz.kCGEventLeftMouseDragged if dragging else Quartz.kCGEventMouseMoved
        _post_mouse_event(move_type, x, y)

    def click(self, click_count: int) -> None:
        import Quartz  # type: ignore[import-untyped]

        x, y = get_cursor_position()
        state = max(1, min(int(click_count), 2))
        _post_mouse_event(Quartz.kCGEventLeftMouseDown, x, y, state)
        _post_mouse_event(Quartz.kCGEventLeftMouseUp, x, y, state)

    def press(self) -> None:
        import Quartz  # type: ignore[import-untyped]

        x, y = get_cursor_position()
        _post_mouse_event(Quartz.kCGEventLeftMouseDown, x, y, 1)

    def release(self) -> None:
        import Quartz  # type: ignore[import-untyped]

        x, y = get_cursor_position()
        _post_mouse_event(Quartz.kCGEventLeftMouseUp, x, y, 1)

    def scroll(self, vertical_pixels: int) -> None:
        import Quartz  # type: ignore[import-untyped]

        # Variadic in C: PyObjC requires exactly `wheelCount` wheel values after it.
        event = Quartz.CGEventCreateScrollWheelEvent(
            None,
            Quartz.kCGScrollEventUnitPixel,
            1,
            int(vertical_pixels),
        )
        # Continuous (trackpad-style) events skip the wheel acceleration curve that made
        # line steps uneven, and apps render them as smooth scrolling.
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGScrollWheelEventIsContinuous, 1)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class NullInputBackend:
    """Used in tests to avoid moving the real cursor."""

    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.drag_moves: list[tuple[float, float]] = []
        self.clicks: list[int] = []
        self.presses = 0
        self.releases = 0
        self.scrolls: list[int] = []

    def move_pointer(self, x: float, y: float, *, dragging: bool = False) -> None:
        (self.drag_moves if dragging else self.moves).append((x, y))

    def click(self, click_count: int) -> None:
        self.clicks.append(click_count)

    def press(self) -> None:
        self.presses += 1

    def release(self) -> None:
        self.releases += 1

    def scroll(self, vertical_pixels: int) -> None:
        self.scrolls.append(vertical_pixels)


class SystemInputController:
    def __init__(self, backend: InputBackend | None = None) -> None:
        self._backend = backend or QuartzInputBackend()
        self._button_down = False

    @property
    def button_down(self) -> bool:
        return self._button_down

    def execute(self, command: GestureCommand) -> bool:
        if not accessibility_trusted():
            return False
        if isinstance(command, MovePointer):
            x, y = get_cursor_position()
            # Displays are re-read each move so plugging/unplugging a monitor just works.
            nx, ny = clamp_to_displays(x + command.dx, y + command.dy)
            self._backend.move_pointer(nx, ny, dragging=self._button_down)
            return True
        if isinstance(command, ClickPointer):
            self._backend.click(command.click_count)
            return True
        if isinstance(command, PressPointer):
            if not self._button_down:
                self._backend.press()
                self._button_down = True
            return True
        if isinstance(command, ReleasePointer):
            return self.release_held_button()
        if isinstance(command, ScrollWheel):
            self._backend.scroll(command.vertical_pixels)
            return True
        return False

    def release_held_button(self) -> bool:
        """Let go of a held button; safe to call any time (e.g. on pause) so it never sticks."""
        if not self._button_down:
            return False
        self._button_down = False
        self._backend.release()
        return True
