from gesturemac.coordinator import AppCoordinator, AppState


def test_starts_paused_generation_zero() -> None:
    c = AppCoordinator()
    assert c.state is AppState.PAUSED
    assert c.generation == 0
    assert c.is_current(0) is False


def test_start_active_and_bumps_generation() -> None:
    c = AppCoordinator()
    snap = c.start()
    assert snap.state is AppState.ACTIVE
    assert snap.generation == 1
    assert c.is_current(1) is True
    assert c.is_current(0) is False


def test_pause_bumps_generation_and_not_current() -> None:
    c = AppCoordinator()
    c.start()
    snap = c.pause()
    assert snap.state is AppState.PAUSED
    assert snap.generation == 2
    assert c.is_current(1) is False
    assert c.is_current(2) is False


def test_toggle() -> None:
    c = AppCoordinator()
    c.toggle()
    assert c.state is AppState.ACTIVE
    c.toggle()
    assert c.state is AppState.PAUSED


def test_idempotent_start_pause() -> None:
    c = AppCoordinator()
    c.start()
    gen = c.generation
    c.start()
    assert c.generation == gen
    c.pause()
    gen2 = c.generation
    c.pause()
    assert c.generation == gen2


def test_listeners_notified() -> None:
    c = AppCoordinator()
    seen: list[tuple[AppState, int]] = []

    def listener(state: AppState, gen: int) -> None:
        seen.append((state, gen))

    c.add_listener(listener)
    c.start()
    c.pause()
    assert seen == [(AppState.ACTIVE, 1), (AppState.PAUSED, 2)]
