"""Output commands from the gesture engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class MovePointer:
    dx: float
    dy: float


@dataclass(frozen=True)
class ClickPointer:
    click_count: int = 1


@dataclass(frozen=True)
class PressPointer:
    """Hold the left button down; MovePointer drags until ReleasePointer."""


@dataclass(frozen=True)
class ReleasePointer:
    pass


@dataclass(frozen=True)
class ScrollWheel:
    # Negative scrolls the page down (toward the end of the document), positive scrolls up.
    vertical_pixels: int


GestureCommand = Union[MovePointer, ClickPointer, PressPointer, ReleasePointer, ScrollWheel]
