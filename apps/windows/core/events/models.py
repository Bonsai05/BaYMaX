"""Typed canonical event and materialized-state records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Event:
    event_id: str
    entity_type: str
    entity_id: str
    event_type: str
    payload: dict[str, Any]
    sequence: int | None = None
    parent_event_id: str | None = None
    base_version: int | None = None
    tombstone: bool = False
    source_device_id: str = "windows-core"
    source_sequence: int | None = None
    idempotency_key: str | None = None
    created_at: float = 0.0


@dataclass(frozen=True)
class StateRecord:
    entity_type: str
    entity_id: str
    version: int
    payload: dict[str, Any] | None
    deleted: bool
    last_event_id: str
    updated_at: float
