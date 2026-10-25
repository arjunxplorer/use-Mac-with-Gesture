"""Application state: paused/active sessions and generation gating."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Callable


class AppState(str, Enum):
    PAUSED = "paused"
    ACTIVE = "active"


Listener = Callable[[AppState, int], None]


@dataclass
class CoordinatorSnapshot:
    state: AppState
    generation: int


class AppCoordinator:
    """
    Owns PAUSED vs ACTIVE and a monotonic generation token.

    Generation increments on every start() and pause() so in-flight work from a
    prior session can be discarded (Slice B+).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._state = AppState.PAUSED
        self._generation = 0
        self._listeners: list[Listener] = []

    @property
    def state(self) -> AppState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def snapshot(self) -> CoordinatorSnapshot:
        with self._lock:
            return CoordinatorSnapshot(state=self._state, generation=self._generation)

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation and self._state is AppState.ACTIVE

    def add_listener(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def _notify(self) -> None:
        state = self._state
        gen = self._generation
        listeners = list(self._listeners)
        for listener in listeners:
            listener(state, gen)

    def start(self) -> CoordinatorSnapshot:
        with self._lock:
            if self._state is AppState.ACTIVE:
                return CoordinatorSnapshot(self._state, self._generation)
            self._generation += 1
            self._state = AppState.ACTIVE
            snap = CoordinatorSnapshot(self._state, self._generation)
        self._notify()
        return snap

    def pause(self) -> CoordinatorSnapshot:
        with self._lock:
            if self._state is AppState.PAUSED:
                return CoordinatorSnapshot(self._state, self._generation)
            self._generation += 1
            self._state = AppState.PAUSED
            snap = CoordinatorSnapshot(self._state, self._generation)
        self._notify()
        return snap

    def toggle(self) -> CoordinatorSnapshot:
        with self._lock:
            active = self._state is AppState.ACTIVE
        if active:
            return self.pause()
        return self.start()
