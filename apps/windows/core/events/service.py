"""Application service for canonical state and durable synchronization."""

from __future__ import annotations

import time
import uuid
from typing import Any, Iterable

from apps.windows.core.errors import ValidationError
from apps.windows.core.events.models import Event, StateRecord
from apps.windows.storage.repositories import EventRepository, StateRepository


class StateService:
    def __init__(
        self,
        events: EventRepository,
        state: StateRepository,
        device_id: str = "windows-core",
    ) -> None:
        self._events = events
        self._state = state
        self._device_id = device_id

    @staticmethod
    def _validate_text(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name} must be a non-empty string")

    def apply(
        self,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any],
        *,
        base_version: int | None = None,
        event_id: str | None = None,
        parent_event_id: str | None = None,
        idempotency_key: str | None = None,
        source_device_id: str | None = None,
        source_sequence: int | None = None,
    ) -> Event:
        self._validate_text("entity_type", entity_type)
        self._validate_text("entity_id", entity_id)
        if event_id is not None:
            self._validate_text("event_id", event_id)
        if idempotency_key is not None:
            self._validate_text("idempotency_key", idempotency_key)
        if not isinstance(payload, dict):
            raise ValidationError("State payload must be a JSON object")
        if base_version is not None and base_version < 0:
            raise ValidationError("base_version cannot be negative")
        event = Event(
            event_id=event_id or str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            event_type="state.upsert",
            payload=payload,
            parent_event_id=parent_event_id,
            base_version=base_version,
            source_device_id=source_device_id or self._device_id,
            source_sequence=source_sequence,
            idempotency_key=idempotency_key,
            created_at=time.time(),
        )
        return self._events.append_many([event])[0]

    def delete(
        self,
        entity_type: str,
        entity_id: str,
        *,
        base_version: int | None = None,
        event_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Event:
        self._validate_text("entity_type", entity_type)
        self._validate_text("entity_id", entity_id)
        if event_id is not None:
            self._validate_text("event_id", event_id)
        if idempotency_key is not None:
            self._validate_text("idempotency_key", idempotency_key)
        if base_version is not None and base_version < 0:
            raise ValidationError("base_version cannot be negative")
        return self._events.append_many(
            [
                Event(
                    event_id=event_id or str(uuid.uuid4()),
                    entity_type=entity_type,
                    entity_id=entity_id,
                    event_type="state.delete",
                    payload={},
                    base_version=base_version,
                    tombstone=True,
                    source_device_id=self._device_id,
                    idempotency_key=idempotency_key,
                    created_at=time.time(),
                )
            ]
        )[0]

    def apply_batch(self, events: Iterable[Event]) -> list[Event]:
        batch = list(events)
        for event in batch:
            self._validate_text("event_id", event.event_id)
            self._validate_text("entity_type", event.entity_type)
            self._validate_text("entity_id", event.entity_id)
            self._validate_text("event_type", event.event_type)
            self._validate_text("source_device_id", event.source_device_id)
            if event.event_type not in {"state.upsert", "state.delete"}:
                raise ValidationError(f"Unsupported event_type: {event.event_type}")
            if event.tombstone != (event.event_type == "state.delete"):
                raise ValidationError("Only state.delete events may set tombstone")
            if not isinstance(event.payload, dict):
                raise ValidationError("Event payload must be a JSON object")
            if event.base_version is not None and event.base_version < 0:
                raise ValidationError("base_version cannot be negative")
            if event.source_sequence is not None and event.source_sequence < 0:
                raise ValidationError("source_sequence cannot be negative")
            if event.idempotency_key is not None:
                self._validate_text("idempotency_key", event.idempotency_key)
        return self._events.append_many(batch)

    def get(self, entity_type: str, entity_id: str) -> StateRecord | None:
        self._validate_text("entity_type", entity_type)
        self._validate_text("entity_id", entity_id)
        return self._state.get(entity_type, entity_id)

    def list(self, entity_type: str, *, include_deleted: bool = False) -> list[StateRecord]:
        self._validate_text("entity_type", entity_type)
        return self._state.list(entity_type, include_deleted)

    def entity_types(self) -> list[str]:
        return self._state.entity_types()

    def events_after(self, sequence: int, *, limit: int = 1000) -> list[Event]:
        if sequence < 0:
            raise ValidationError("sequence cannot be negative")
        if not 1 <= limit <= 10000:
            raise ValidationError("limit must be between 1 and 10000")
        return self._events.after_sequence(sequence, limit)
