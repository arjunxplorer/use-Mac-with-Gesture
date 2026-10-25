"""Pointer position helpers (all active displays, global Quartz coordinates)."""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_DISPLAYS = 16


@dataclass(frozen=True)
class ScreenBounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x < self.max_x and self.min_y <= y < self.max_y

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        # Keep one point inside the far edges; max_x/max_y belong to the neighbouring display.
        cx = max(self.min_x, min(self.max_x - 1.0, x))
        cy = max(self.min_y, min(self.max_y - 1.0, y))
        return cx, cy


_FALLBACK = ScreenBounds(min_x=0.0, min_y=0.0, max_x=1920.0, max_y=1080.0)


def _bounds_for(display_id: int) -> ScreenBounds:
    from Quartz import CGDisplayBounds  # type: ignore[import-untyped]

    rect = CGDisplayBounds(display_id)
    return ScreenBounds(
        min_x=float(rect.origin.x),
        min_y=float(rect.origin.y),
        max_x=float(rect.origin.x + rect.size.width),
        max_y=float(rect.origin.y + rect.size.height),
    )


def primary_display_bounds() -> ScreenBounds:
    try:
        from Quartz import CGMainDisplayID  # type: ignore[import-untyped]

        return _bounds_for(CGMainDisplayID())
    except Exception:  # noqa: BLE001
        return _FALLBACK


def active_display_bounds() -> list[ScreenBounds]:
    try:
        from Quartz import CGGetActiveDisplayList  # type: ignore[import-untyped]

        err, ids, count = CGGetActiveDisplayList(MAX_DISPLAYS, None, None)
        if err == 0 and count:
            return [_bounds_for(d) for d in ids[:count]]
    except Exception:  # noqa: BLE001
        pass
    return [primary_display_bounds()]


def get_cursor_position() -> tuple[float, float]:
    import Quartz  # type: ignore[import-untyped]

    event = Quartz.CGEventCreate(None)
    loc = Quartz.CGEventGetLocation(event)
    return float(loc.x), float(loc.y)


def clamp_to_displays(
    x: float,
    y: float,
    displays: list[ScreenBounds] | None = None,
) -> tuple[float, float]:
    """
    Keep the point on a real screen. Displays of different sizes leave gaps in the
    desktop's bounding box, so snap to the nearest display rather than the union.
    """
    screens = displays or active_display_bounds()
    for screen in screens:
        if screen.contains(x, y):
            return x, y
    best = screens[0].clamp(x, y)
    best_dist = math.hypot(best[0] - x, best[1] - y)
    for screen in screens[1:]:
        candidate = screen.clamp(x, y)
        dist = math.hypot(candidate[0] - x, candidate[1] - y)
        if dist < best_dist:
            best, best_dist = candidate, dist
    return best
