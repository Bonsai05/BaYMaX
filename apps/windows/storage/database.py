"""SQLite database setup, migrations, and transaction management."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_SCHEMA_VERSION = 1


class Database:
    """Own one SQLite connection and serialize access to it across threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {version} is newer than supported "
                    f"version {_SCHEMA_VERSION}"
                )
            if version == 0:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE event_log (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        parent_event_id TEXT,
                        base_version INTEGER,
                        tombstone INTEGER NOT NULL DEFAULT 0 CHECK (tombstone IN (0, 1)),
                        source_device_id TEXT NOT NULL,
                        source_sequence INTEGER,
                        idempotency_key TEXT UNIQUE,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX event_log_entity_sequence
                        ON event_log(entity_type, entity_id, sequence);
                    CREATE INDEX event_log_source_sequence
                        ON event_log(source_device_id, source_sequence);
                    CREATE TABLE canonical_state (
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        version INTEGER NOT NULL CHECK (version > 0),
                        payload_json TEXT,
                        deleted INTEGER NOT NULL CHECK (deleted IN (0, 1)),
                        last_event_id TEXT NOT NULL REFERENCES event_log(event_id),
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (entity_type, entity_id)
                    );
                    CREATE TABLE tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
                        ),
                        payload_json TEXT NOT NULL,
                        result_json TEXT,
                        error TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        lease_owner TEXT,
                        lease_until REAL,
                        lease_epoch INTEGER NOT NULL DEFAULT 0 CHECK (lease_epoch >= 0),
                        idempotency_key TEXT UNIQUE
                    );
                    CREATE INDEX tasks_claim_order ON tasks(status, created_at, task_id);
                    CREATE TABLE task_checkpoints (
                        checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL REFERENCES tasks(task_id),
                        lease_epoch INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        UNIQUE (task_id, idempotency_key)
                    );
                    CREATE TABLE task_operations (
                        task_id TEXT NOT NULL REFERENCES tasks(task_id),
                        idempotency_key TEXT NOT NULL,
                        request_hash TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        PRIMARY KEY (task_id, idempotency_key)
                    );
                    CREATE TABLE task_history (
                        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL REFERENCES tasks(task_id),
                        from_status TEXT,
                        to_status TEXT NOT NULL,
                        actor TEXT,
                        lease_epoch INTEGER NOT NULL,
                        details_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._connection

    def integrity_check(self) -> str:
        with self.read() as connection:
            results = connection.execute("PRAGMA integrity_check").fetchall()
        messages = [str(row[0]) for row in results]
        if messages != ["ok"]:
            raise RuntimeError("SQLite integrity check failed: " + "; ".join(messages))
        return "ok"

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
