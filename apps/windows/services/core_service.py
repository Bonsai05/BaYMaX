"""Composition root and stable service boundary for the Windows Core."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from apps.windows.core.events.models import Event
from apps.windows.core.events.service import StateService
from apps.windows.core.tasks.service import TaskService
from apps.windows.storage.database import Database
from apps.windows.storage.repositories import EventRepository, StateRepository, TaskRepository


class WindowsCore:
    """Compose repositories and domain services without requiring AI or networking."""

    def __init__(self, database_path: str | Path, *, device_id: str = "windows-core") -> None:
        self.database = Database(database_path)
        events = EventRepository(self.database)
        self.state = StateService(events, StateRepository(self.database), device_id)
        self.tasks = TaskService(TaskRepository(self.database))

    def push_events(self, events: Iterable[Event]) -> list[Event]:
        """Durably apply a batch; either every event is committed or none are."""
        return self.state.apply_batch(events)

    def pull_events(self, after_sequence: int, *, limit: int = 1000) -> list[Event]:
        return self.state.events_after(after_sequence, limit=limit)

    def recover(self) -> str:
        """Validate SQLite pages after restart; SQLite WAL recovery runs on open."""
        return self.database.integrity_check()

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> WindowsCore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
