"""Post synthetic mouse events (Accessibility required)."""

from __future__ import annotations

from typing import Protocol

from gesturemac.gestures.commands import ClickPointer, GestureCommand, MovePointer, ScrollWheel
from gesturemac.input.cursor import clamp_to_displays, get_cursor_position
from gesturemac.permissions import accessibility_trusted


class InputBackend(Protocol):
    def move_pointer(self, x: float, y: float) -> None: ...
    def click(self, click_count: int) -> None: ...
    def scroll(self, vertical_pixels: int) -> None: ...


class QuartzInputBackend:
    def move_pointer(self, x: float, y: float) -> None:
        import Quartz  # type: ignore[import-untyped]

        point = Quartz.CGPoint(x, y)
        move_type = Quartz.kCGEventMouseMoved
        event = Quartz.CGEventCreateMouseEvent(None, move_type, point, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def click(self, click_count: int) -> None:
        import Quartz  # type: ignore[import-untyped]

        x, y = get_cursor_position()
        point = Quartz.CGPoint(x, y)
        down = Quartz.kCGEventLeftMouseDown
        up = Quartz.kCGEventLeftMouseUp
        state = max(1, min(int(click_count), 2))
        down_event = Quartz.CGEventCreateMouseEvent(
            None,
            down,
            point,
            Quartz.kCGMouseButtonLeft,
        )
        Quartz.CGEventSetIntegerValueField(
            down_event,
            Quartz.kCGMouseEventClickState,
            state,
        )
        up_event = Quartz.CGEventCreateMouseEvent(
            None,
            up,
            point,
            Quartz.kCGMouseButtonLeft,
        )
        Quartz.CGEventSetIntegerValueField(
            up_event,
            Quartz.kCGMouseEventClickState,
            state,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)

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
        self.clicks: list[int] = []
        self.scrolls: list[int] = []

    def move_pointer(self, x: float, y: float) -> None:
        self.moves.append((x, y))

    def click(self, click_count: int) -> None:
        self.clicks.append(click_count)

    def scroll(self, vertical_pixels: int) -> None:
        self.scrolls.append(vertical_pixels)


class SystemInputController:
    def __init__(self, backend: InputBackend | None = None) -> None:
        self._backend = backend or QuartzInputBackend()

    def execute(self, command: GestureCommand) -> bool:
        if not accessibility_trusted():
            return False
        if isinstance(command, MovePointer):
            x, y = get_cursor_position()
            # Displays are re-read each move so plugging/unplugging a monitor just works.
            nx, ny = clamp_to_displays(x + command.dx, y + command.dy)
            self._backend.move_pointer(nx, ny)
            return True
        if isinstance(command, ClickPointer):
            self._backend.click(command.click_count)
            return True
        if isinstance(command, ScrollWheel):
            self._backend.scroll(command.vertical_pixels)
            return True
        return False
