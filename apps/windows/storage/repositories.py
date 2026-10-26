"""Repositories that keep SQLite access behind a small persistence boundary."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable

from apps.windows.core.errors import ConflictError, NotFoundError
from apps.windows.core.events.models import Event, StateRecord
from apps.windows.core.tasks.models import Checkpoint, TaskRecord
from apps.windows.storage.database import Database


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _event(row: sqlite3.Row) -> Event:
    return Event(
        event_id=row["event_id"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        event_type=row["event_type"],
        payload=json.loads(row["payload_json"]),
        sequence=row["sequence"],
        parent_event_id=row["parent_event_id"],
        base_version=row["base_version"],
        tombstone=bool(row["tombstone"]),
        source_device_id=row["source_device_id"],
        source_sequence=row["source_sequence"],
        idempotency_key=row["idempotency_key"],
        created_at=row["created_at"],
    )


def _task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"],
        task_type=row["task_type"],
        status=row["status"],
        payload=json.loads(row["payload_json"]),
        result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        lease_owner=row["lease_owner"],
        lease_until=row["lease_until"],
        lease_epoch=row["lease_epoch"],
        idempotency_key=row["idempotency_key"],
    )


class EventRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def append_many(self, events: Iterable[Event]) -> list[Event]:
        drafts = list(events)
        if not drafts:
            return []
        committed: list[Event] = []
        with self._database.transaction() as connection:
            for draft in drafts:
                existing = connection.execute(
                    "SELECT * FROM event_log WHERE event_id = ?",
                    (draft.event_id,),
                ).fetchone()
                if existing is None and draft.idempotency_key is not None:
                    existing = connection.execute(
                        "SELECT * FROM event_log WHERE idempotency_key = ?",
                        (draft.idempotency_key,),
                    ).fetchone()
                if existing is not None:
                    same_request = (
                        existing["entity_type"] == draft.entity_type
                        and existing["entity_id"] == draft.entity_id
                        and existing["event_type"] == draft.event_type
                        and existing["payload_json"] == _json(draft.payload)
                        and existing["parent_event_id"] == draft.parent_event_id
                        and existing["base_version"] == draft.base_version
                        and bool(existing["tombstone"]) == draft.tombstone
                        and existing["source_device_id"] == draft.source_device_id
                        and existing["source_sequence"] == draft.source_sequence
                        and existing["idempotency_key"] == draft.idempotency_key
                    )
                    if not same_request:
                        raise ConflictError(
                            f"Event {draft.event_id} or its idempotency key was reused "
                            "with different event data"
                        )
                    committed.append(_event(existing))
                    continue

                current = connection.execute(
                    """
                    SELECT version FROM canonical_state
                    WHERE entity_type = ? AND entity_id = ?
                    """,
                    (draft.entity_type, draft.entity_id),
                ).fetchone()
                version = int(current["version"]) if current else 0
                if draft.base_version is not None and draft.base_version != version:
                    raise ConflictError(
                        f"State version conflict for {draft.entity_type}/{draft.entity_id}: "
                        f"expected {draft.base_version}, current version is {version}"
                    )
                cursor = connection.execute(
                    """
                    INSERT INTO event_log (
                        event_id, entity_type, entity_id, event_type, payload_json,
                        parent_event_id, base_version, tombstone, source_device_id,
                        source_sequence, idempotency_key, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        draft.event_id,
                        draft.entity_type,
                        draft.entity_id,
                        draft.event_type,
                        _json(draft.payload),
                        draft.parent_event_id,
                        draft.base_version,
                        int(draft.tombstone),
                        draft.source_device_id,
                        draft.source_sequence,
                        draft.idempotency_key,
                        draft.created_at,
                    ),
                )
                sequence = int(cursor.lastrowid)
                new_version = version + 1
                connection.execute(
                    """
                    INSERT INTO canonical_state (
                        entity_type, entity_id, version, payload_json, deleted,
                        last_event_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                        version = excluded.version,
                        payload_json = excluded.payload_json,
                        deleted = excluded.deleted,
                        last_event_id = excluded.last_event_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        draft.entity_type,
                        draft.entity_id,
                        new_version,
                        None if draft.tombstone else _json(draft.payload),
                        int(draft.tombstone),
                        draft.event_id,
                        draft.created_at,
                    ),
                )
                committed.append(
                    Event(
                        event_id=draft.event_id,
                        entity_type=draft.entity_type,
                        entity_id=draft.entity_id,
                        event_type=draft.event_type,
                        payload=draft.payload,
                        sequence=sequence,
                        parent_event_id=draft.parent_event_id,
                        base_version=draft.base_version,
                        tombstone=draft.tombstone,
                        source_device_id=draft.source_device_id,
                        source_sequence=draft.source_sequence,
                        idempotency_key=draft.idempotency_key,
                        created_at=draft.created_at,
                    )
                )
        return committed

    def get(self, event_id: str) -> Event:
        with self._database.read() as connection:
            row = connection.execute("SELECT * FROM event_log WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Event {event_id} was not found")
        return _event(row)

    def after_sequence(self, sequence: int, limit: int = 1000) -> list[Event]:
        with self._database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM event_log WHERE sequence > ? ORDER BY sequence LIMIT ?",
                (sequence, limit),
            ).fetchall()
        return [_event(row) for row in rows]


class StateRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def get(self, entity_type: str, entity_id: str) -> StateRecord | None:
        with self._database.read() as connection:
            row = connection.execute(
                """
                SELECT * FROM canonical_state
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, entity_id),
            ).fetchone()
        if row is None:
            return None
        return StateRecord(
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            version=row["version"],
            payload=json.loads(row["payload_json"]) if row["payload_json"] is not None else None,
            deleted=bool(row["deleted"]),
            last_event_id=row["last_event_id"],
            updated_at=row["updated_at"],
        )

    def list(self, entity_type: str, include_deleted: bool = False) -> list[StateRecord]:
        statement = "SELECT * FROM canonical_state WHERE entity_type = ?"
        params: tuple[Any, ...] = (entity_type,)
        if not include_deleted:
            statement += " AND deleted = 0"
        statement += " ORDER BY entity_id"
        with self._database.read() as connection:
            rows = connection.execute(statement, params).fetchall()
        return [
            StateRecord(
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                version=row["version"],
                payload=json.loads(row["payload_json"]) if row["payload_json"] is not None else None,
                deleted=bool(row["deleted"]),
                last_event_id=row["last_event_id"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def entity_types(self) -> list[str]:
        with self._database.read() as connection:
            rows = connection.execute(
                "SELECT DISTINCT entity_type FROM canonical_state ORDER BY entity_type"
            ).fetchall()
        return [str(row["entity_type"]) for row in rows]


class TaskRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _history(
        connection: sqlite3.Connection,
        task_id: str,
        old_status: str | None,
        new_status: str,
        actor: str | None,
        lease_epoch: int,
        details: dict[str, Any],
        now: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO task_history (
                task_id, from_status, to_status, actor, lease_epoch, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, old_status, new_status, actor, lease_epoch, _json(details), now),
        )

    @staticmethod
    def _get(connection: sqlite3.Connection, task_id: str) -> TaskRecord:
        row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Task {task_id} was not found")
        return _task(row)

    def create(
        self,
        task_id: str,
        task_type: str,
        payload: dict[str, Any],
        now: float,
        idempotency_key: str | None,
    ) -> TaskRecord:
        with self._database.transaction() as connection:
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM tasks WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    if existing["task_type"] != task_type or existing["payload_json"] != _json(payload):
                        raise ConflictError("Task idempotency key was reused with different task data")
                    return _task(existing)
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, task_type, status, payload_json, created_at, updated_at,
                    idempotency_key
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?)
                """,
                (task_id, task_type, _json(payload), now, now, idempotency_key),
            )
            self._history(connection, task_id, None, "pending", None, 0, {}, now)
            return self._get(connection, task_id)

    def get(self, task_id: str) -> TaskRecord:
        with self._database.read() as connection:
            return self._get(connection, task_id)

    def list(self, status: str | None = None) -> list[TaskRecord]:
        with self._database.read() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT * FROM tasks ORDER BY updated_at DESC, task_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY updated_at DESC, task_id",
                    (status,),
                ).fetchall()
        return [_task(row) for row in rows]

    def claim_next(self, worker_id: str, lease_seconds: float, now: float) -> TaskRecord | None:
        with self._database.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'pending'
                   OR (status = 'running' AND lease_until <= ?)
                ORDER BY created_at, task_id
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            old_status = row["status"]
            epoch = int(row["lease_epoch"]) + 1
            lease_until = now + lease_seconds
            connection.execute(
                """
                UPDATE tasks SET status = 'running', lease_owner = ?, lease_until = ?,
                    lease_epoch = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (worker_id, lease_until, epoch, now, row["task_id"]),
            )
            self._history(
                connection,
                row["task_id"],
                old_status,
                "running",
                worker_id,
                epoch,
                {"lease_until": lease_until},
                now,
            )
            return self._get(connection, row["task_id"])

    def renew(self, task_id: str, worker_id: str, lease_epoch: int, lease_seconds: float, now: float) -> TaskRecord:
        with self._database.transaction() as connection:
            task = self._get(connection, task_id)
            self._assert_lease(task, worker_id, lease_epoch, now)
            lease_until = now + lease_seconds
            connection.execute(
                "UPDATE tasks SET lease_until = ?, updated_at = ? WHERE task_id = ?",
                (lease_until, now, task_id),
            )
            return self._get(connection, task_id)

    @staticmethod
    def _assert_lease(task: TaskRecord, worker_id: str, lease_epoch: int, now: float) -> None:
        if (
            task.status != "running"
            or task.lease_owner != worker_id
            or task.lease_epoch != lease_epoch
            or task.lease_until is None
            or task.lease_until <= now
        ):
            raise ConflictError(f"Task {task.task_id} lease is expired or fenced by a newer owner")

    def add_checkpoint(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        payload: dict[str, Any],
        idempotency_key: str,
        now: float,
    ) -> Checkpoint:
        with self._database.transaction() as connection:
            task = self._get(connection, task_id)
            self._assert_lease(task, worker_id, lease_epoch, now)
            existing = connection.execute(
                """
                SELECT * FROM task_checkpoints
                WHERE task_id = ? AND idempotency_key = ?
                """,
                (task_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != _json(payload):
                    raise ConflictError("Checkpoint idempotency key was reused with different data")
                return Checkpoint(
                    checkpoint_id=existing["checkpoint_id"],
                    task_id=task_id,
                    lease_epoch=existing["lease_epoch"],
                    payload=json.loads(existing["payload_json"]),
                    created_at=existing["created_at"],
                    idempotency_key=existing["idempotency_key"],
                )
            cursor = connection.execute(
                """
                INSERT INTO task_checkpoints (
                    task_id, lease_epoch, payload_json, created_at, idempotency_key
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, lease_epoch, _json(payload), now, idempotency_key),
            )
            return Checkpoint(
                checkpoint_id=int(cursor.lastrowid),
                task_id=task_id,
                lease_epoch=lease_epoch,
                payload=payload,
                created_at=now,
                idempotency_key=idempotency_key,
            )

    def checkpoints(self, task_id: str) -> list[Checkpoint]:
        with self._database.read() as connection:
            if connection.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone() is None:
                raise NotFoundError(f"Task {task_id} was not found")
            rows = connection.execute(
                "SELECT * FROM task_checkpoints WHERE task_id = ? ORDER BY checkpoint_id",
                (task_id,),
            ).fetchall()
        return [
            Checkpoint(
                checkpoint_id=row["checkpoint_id"],
                task_id=row["task_id"],
                lease_epoch=row["lease_epoch"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
                idempotency_key=row["idempotency_key"],
            )
            for row in rows
        ]

    def finish(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
        idempotency_key: str,
        now: float,
    ) -> TaskRecord:
        request = {
            "worker_id": worker_id,
            "lease_epoch": lease_epoch,
            "status": status,
            "result": result,
            "error": error,
        }
        request_hash = hashlib.sha256(_json(request).encode("utf-8")).hexdigest()
        with self._database.transaction() as connection:
            existing = connection.execute(
                """
                SELECT request_hash FROM task_operations
                WHERE task_id = ? AND idempotency_key = ?
                """,
                (task_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise ConflictError("Task operation idempotency key was reused with different data")
                return self._get(connection, task_id)

            task = self._get(connection, task_id)
            self._assert_lease(task, worker_id, lease_epoch, now)
            connection.execute(
                """
                UPDATE tasks SET status = ?, result_json = ?, error = ?,
                    lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE task_id = ?
                """,
                (status, _json(result) if result is not None else None, error, now, task_id),
            )
            self._history(
                connection,
                task_id,
                task.status,
                status,
                worker_id,
                lease_epoch,
                {"error": error, "result": result},
                now,
            )
            connection.execute(
                """
                INSERT INTO task_operations (task_id, idempotency_key, request_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, idempotency_key, request_hash, now),
            )
            return self._get(connection, task_id)

    def history(self, task_id: str) -> list[dict[str, Any]]:
        with self._database.read() as connection:
            if connection.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone() is None:
                raise NotFoundError(f"Task {task_id} was not found")
            rows = connection.execute(
                "SELECT * FROM task_history WHERE task_id = ? ORDER BY history_id",
                (task_id,),
            ).fetchall()
        return [
            {
                "from_status": row["from_status"],
                "to_status": row["to_status"],
                "actor": row["actor"],
                "lease_epoch": row["lease_epoch"],
                "details": json.loads(row["details_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
