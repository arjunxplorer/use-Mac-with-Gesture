import json
from pathlib import Path

from gesturemac.settings import Settings, load, save


def test_defaults() -> None:
    s = Settings()
    assert s.pointer_sensitivity == 1500.0
    assert s.smoothing_alpha == 0.35
    assert s.pinch_down_threshold == 0.25
    assert s.show_diagnostics is False


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = Settings(pointer_sensitivity=1200, invert_scroll=True)
    save(original, path)
    loaded = load(path)
    assert loaded.pointer_sensitivity == 1200
    assert loaded.invert_scroll is True


def test_missing_file_uses_defaults(tmp_path: Path) -> None:
    assert load(tmp_path / "missing.json") == Settings().clamped()


def test_corrupt_file_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")
    assert load(path) == Settings().clamped()


def test_clamp_pointer_sensitivity(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"pointer_sensitivity": 99999}), encoding="utf-8")
    loaded = load(path)
    assert loaded.pointer_sensitivity == 5000.0


def test_pinch_up_above_down(tmp_path: Path) -> None:
    s = Settings(pinch_down_threshold=0.5, pinch_up_threshold=0.2).clamped()
    assert s.pinch_up_threshold > s.pinch_down_threshold


def test_scroll_settings_reach_gesture_config(tmp_path: Path) -> None:
    from gesturemac.gestures.config import GestureConfig

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"scroll_dead_zone": 0.05, "scroll_max_speed": 60, "scroll_sensitivity": 150}),
        encoding="utf-8",
    )
    config = GestureConfig.from_settings(load(path))
    assert config.scroll_dead_zone == 0.05
    assert config.scroll_max_speed == 60


def test_scroll_max_speed_not_below_min() -> None:
    s = Settings(scroll_min_speed=20, scroll_max_speed=5).clamped()
    assert s.scroll_max_speed >= s.scroll_min_speed
