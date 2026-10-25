# GestureMac

Local macOS menu-bar utility: control the system cursor, clicks, and vertical scrolling with hand gestures from the built-in camera. All processing stays on device.

## Requirements

- macOS 14+ on Apple Silicon
- Python 3.12 (see `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) recommended for installs

## Slice E (current)

When Active:

- **Pointer:** index finger extended, other fingers curled. Moves the cursor relative to where it is (trackpad-style), with acceleration so fast flicks travel farther than slow drags.
- **Reposition ("lift the mouse"):** curl your index finger. The cursor stops instantly; move your hand back to the middle of the camera view and point again to keep going. The cursor moves across all connected displays.
- **Click:** while pointing, touch thumb to index tip, then release. One click fires on release; holding the pinch never repeats. The cursor freezes as the thumb closes in so the click lands where you aimed.
- **Double-click:** two pinches within the macOS double-click interval without moving the cursor.

- **Scroll (joystick):** turn your hand sideways to the camera and stretch the index and middle fingers out horizontally, pressed together (ring and pinky curled). Hold still for about a third of a second: that height becomes the neutral center. Raise the fingers above it to scroll down, lower them to scroll up; the farther from center, the faster. Return to center to stop. Cursor and clicks are disabled while scrolling. If the direction feels backwards, click **Settings → Invert scroll**; it applies immediately.

Requires **Accessibility** for the app you launch from (Terminal/Cursor). The menu shows gesture mode, live pinch ratio, and clicks sent.

### Tuning

Edit the settings JSON (**Settings → Open settings file**), then Pause and Start to apply. Hand motion is measured in *units*: 1 unit is the camera image's height, roughly 45 cm of hand travel at arm's length.

| Setting | Default | Effect |
|---|---|---|
| `pointer_sensitivity` | 1500 | Screen points per unit at 1× gain. Scales everything. |
| `pointer_min_gain` | 0.4 | Gain for slow, precise moves. Lower = finer control. |
| `pointer_max_gain` | 3.5 | Gain for fast flicks. Higher = cross the screen with less hand travel. |
| `pointer_accel_slow_speed` | 0.08 | Hand speed (units/s) at or below which `min_gain` applies. |
| `pointer_accel_fast_speed` | 0.8 | Hand speed (units/s) at or above which `max_gain` applies. Lower = acceleration kicks in sooner. |
| `smoothing_alpha` | 0.35 | Fingertip filter. Higher = less lag, more jitter. |
| `scroll_activation_hold_seconds` | 0.15 | How long the scroll pose is held before the neutral center is captured. |
| `scroll_dead_zone` | 0.03 | Distance (units) around center that scrolls nothing. |
| `scroll_max_range` | 0.15 | Distance past the dead zone at which max speed is reached. |
| `scroll_min_speed` / `scroll_max_speed` | 60 / 1500 | Scroll pixels/s just outside the dead zone / at full range (sent as smooth, trackpad-style events). |
| `scroll_smoothing_alpha` | 0.3 | Fingertip height filter. Higher = more responsive, more jitter. |
| `scroll_finger_separation_max` | 0.5 | Max gap between the two fingertips, as a fraction of palm length. |
| `scroll_finger_parallel_tolerance_degrees` | 25 | Max angle between the two fingers. |
| `scroll_finger_extension_min` | 0.8 | Required finger straightness (1.0 = perfectly straight). |
| `scroll_max_tilt_degrees` | 45 | Max tilt of the fingers away from horizontal. |
| `scroll_release_slack` | 1.5 | Once scrolling, the pose limits loosen by this factor so tracking noise can't interrupt it. |
| `scroll_exit_grace_seconds` | 0.1 | A pose dropout shorter than this pauses scrolling but keeps the center. |
| `pinch_down_threshold` / `pinch_up_threshold` | 0.25 / 0.35 | Pinch ratio for press / release. |

Set `pointer_min_gain` equal to `pointer_max_gain` to turn acceleration off.

On first tracking run, the hand model downloads to `~/Library/Application Support/GestureMac/hand_landmarker.task` (~3 MB).

**Permissions:** While running via `uv run`, macOS may attribute Camera access to **Terminal or Cursor** — grant camera access to that host app, or wait for a packaged `.app` (Slice G).

### Setup

```bash
cd /path/to/use-Mac-with-Gesture
uv sync --extra dev
```

### Run

```bash
uv run python -m gesturemac
```

The app starts **paused**. Use the menu bar item to Start/Pause or open settings.

### Tests

```bash
uv run pytest
```

See [Gesture.md](Gesture.md) for full product spec and implementation slices.
