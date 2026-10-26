"""Application service enforcing task ownership, fencing, and retry semantics."""

from __future__ import annotations

import time
import uuid
from typing import Any

from apps.windows.core.errors import ValidationError
from apps.windows.core.tasks.models import Checkpoint, TaskRecord
from apps.windows.storage.repositories import TaskRepository


_TERMINAL = {"completed", "failed", "cancelled"}


class TaskService:
    def __init__(self, tasks: TaskRepository) -> None:
        self._tasks = tasks

    @staticmethod
    def _positive_seconds(name: str, value: float) -> None:
        if value <= 0:
            raise ValidationError(f"{name} must be greater than zero")

    def create(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        task_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> TaskRecord:
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValidationError("task_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValidationError("Task payload must be a JSON object")
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            raise ValidationError("task_id must be a non-empty string")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str) or not idempotency_key.strip()
        ):
            raise ValidationError("idempotency_key must be a non-empty string")
        return self._tasks.create(
            task_id or str(uuid.uuid4()),
            task_type,
            payload,
            time.time(),
            idempotency_key,
        )

    def get(self, task_id: str) -> TaskRecord:
        return self._tasks.get(task_id)

    def list(self, status: str | None = None) -> list[TaskRecord]:
        if status is not None and status not in {"pending", "running", *_TERMINAL}:
            raise ValidationError(f"Unknown task status: {status}")
        return self._tasks.list(status)

    def claim_next(self, worker_id: str, *, lease_seconds: float = 30) -> TaskRecord | None:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValidationError("worker_id must be a non-empty string")
        self._positive_seconds("lease_seconds", lease_seconds)
        return self._tasks.claim_next(worker_id, lease_seconds, time.time())

    def renew(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        *,
        lease_seconds: float = 30,
    ) -> TaskRecord:
        self._positive_seconds("lease_seconds", lease_seconds)
        return self._tasks.renew(task_id, worker_id, lease_epoch, lease_seconds, time.time())

    def checkpoint(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> Checkpoint:
        if not isinstance(payload, dict):
            raise ValidationError("Checkpoint payload must be a JSON object")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValidationError("idempotency_key must be a non-empty string")
        return self._tasks.add_checkpoint(
            task_id,
            worker_id,
            lease_epoch,
            payload,
            idempotency_key,
            time.time(),
        )

    def finish(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        idempotency_key: str,
    ) -> TaskRecord:
        if status not in _TERMINAL:
            raise ValidationError(f"status must be one of {', '.join(sorted(_TERMINAL))}")
        if result is not None and not isinstance(result, dict):
            raise ValidationError("Task result must be a JSON object")
        if status == "failed" and not error:
            raise ValidationError("A failed task must include an error message")
        if status != "failed" and error is not None:
            raise ValidationError("Only failed tasks may include an error message")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValidationError("idempotency_key must be a non-empty string")
        return self._tasks.finish(
            task_id,
            worker_id,
            lease_epoch,
            status,
            result,
            error,
            idempotency_key,
            time.time(),
        )

    def checkpoints(self, task_id: str) -> list[Checkpoint]:
        return self._tasks.checkpoints(task_id)

    def history(self, task_id: str) -> list[dict[str, Any]]:
        return self._tasks.history(task_id)
