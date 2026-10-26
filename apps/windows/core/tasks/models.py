"""Typed task and checkpoint records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_type: str
    status: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    created_at: float
    updated_at: float
    lease_owner: str | None
    lease_until: float | None
    lease_epoch: int
    idempotency_key: str | None


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: int
    task_id: str
    lease_epoch: int
    payload: dict[str, Any]
    created_at: float
    idempotency_key: str
