"""Safe macOS notifications when running without a bundled Info.plist."""

from __future__ import annotations

import rumps


def notify(title: str, subtitle: str, message: str) -> None:
    try:
        rumps.notification(title, subtitle, message)
    except RuntimeError:
        # Common when launched via `python -m gesturemac` (no CFBundleIdentifier).
        pass
