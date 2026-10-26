"""Tests for Member 1's Windows Core services and persistence."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from apps.windows.core.errors import ConflictError
from apps.windows.core.events.models import Event
from apps.windows.services import WindowsCore
from apps.windows.ui.demo_data import load_demo_data, seed_demo_data_if_empty
from apps.windows.ui.app import _blend_color


class WindowsCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "baymax.db"
        self.core = WindowsCore(self.database_path, device_id="test-device")

    def tearDown(self) -> None:
        self.core.close()
        self.temp_dir.cleanup()

    def test_state_is_versioned_and_tombstones_are_hidden_by_default(self) -> None:
        first = self.core.state.apply("note", "one", {"text": "first"}, base_version=0)
        self.assertEqual(first.sequence, 1)
        self.assertEqual(self.core.state.get("note", "one").version, 1)

        second = self.core.state.apply("note", "one", {"text": "second"}, base_version=1)
        self.assertEqual(second.sequence, 2)
        with self.assertRaises(ConflictError):
            self.core.state.apply("note", "one", {"text": "stale"}, base_version=1)
        with self.assertRaises(ConflictError):
            self.core.push_events([
                Event(
                    event_id=first.event_id,
                    entity_type="note",
                    entity_id="one",
                    event_type="state.upsert",
                    payload={"text": "different request"},
                    base_version=0,
                    source_device_id="test-device",
                    created_at=time.time(),
                )
            ])

        deleted = self.core.state.delete("note", "one", base_version=2)
        self.assertTrue(deleted.tombstone)
        self.assertIsNone(self.core.state.get("note", "one").payload)
        self.assertEqual(self.core.state.list("note"), [])
        self.assertTrue(self.core.state.list("note", include_deleted=True)[0].deleted)

    def test_sync_batch_is_atomic_and_idempotent(self) -> None:
        first = Event(
            event_id="remote-1",
            entity_type="profile",
            entity_id="main",
            event_type="state.upsert",
            payload={"name": "Baymax"},
            base_version=0,
            source_device_id="android",
            source_sequence=7,
            created_at=time.time(),
        )
        conflicting = Event(
            event_id="remote-2",
            entity_type="profile",
            entity_id="main",
            event_type="state.upsert",
            payload={"name": "stale"},
            base_version=0,
            source_device_id="android",
            source_sequence=8,
            created_at=time.time(),
        )
        with self.assertRaises(ConflictError):
            self.core.push_events([first, conflicting])
        self.assertIsNone(self.core.state.get("profile", "main"))

        committed = self.core.push_events([first])
        self.assertEqual(committed[0].sequence, 1)
        self.assertEqual(self.core.push_events([first])[0].event_id, "remote-1")
        self.assertEqual(self.core.pull_events(0)[0].source_sequence, 7)

    def test_task_checkpoints_fencing_and_finish_idempotency(self) -> None:
        task = self.core.tasks.create(
            "summarize",
            {"text": "hello"},
            idempotency_key="create-1",
        )
        self.assertEqual(task.status, "pending")
        self.assertEqual(
            self.core.tasks.create("summarize", {"text": "hello"}, idempotency_key="create-1").task_id,
            task.task_id,
        )
        claimed = self.core.tasks.claim_next("worker-a", lease_seconds=60)
        self.assertEqual(claimed.task_id, task.task_id)
        checkpoint = self.core.tasks.checkpoint(
            task.task_id,
            "worker-a",
            claimed.lease_epoch,
            {"step": 1},
            idempotency_key="checkpoint-1",
        )
        self.assertEqual(checkpoint.payload, {"step": 1})
        self.assertEqual(
            self.core.tasks.checkpoint(
                task.task_id,
                "worker-a",
                claimed.lease_epoch,
                {"step": 1},
                idempotency_key="checkpoint-1",
            ).checkpoint_id,
            checkpoint.checkpoint_id,
        )

        with self.core.database.transaction() as connection:
            connection.execute(
                "UPDATE tasks SET lease_until = ? WHERE task_id = ?",
                (time.time() - 1, task.task_id),
            )
        reclaimed = self.core.tasks.claim_next("worker-b", lease_seconds=60)
        self.assertGreater(reclaimed.lease_epoch, claimed.lease_epoch)
        with self.assertRaises(ConflictError):
            self.core.tasks.finish(
                task.task_id,
                "worker-a",
                claimed.lease_epoch,
                "completed",
                result={"ok": True},
                idempotency_key="finish-a",
            )

        finished = self.core.tasks.finish(
            task.task_id,
            "worker-b",
            reclaimed.lease_epoch,
            "completed",
            result={"ok": True},
            idempotency_key="finish-b",
        )
        self.assertEqual(finished.status, "completed")
        self.assertEqual(
            self.core.tasks.finish(
                task.task_id,
                "worker-b",
                reclaimed.lease_epoch,
                "completed",
                result={"ok": True},
                idempotency_key="finish-b",
            ).status,
            "completed",
        )
        self.assertEqual([entry["to_status"] for entry in self.core.tasks.history(task.task_id)],
                         ["pending", "running", "running", "completed"])
        self.assertEqual(self.core.tasks.list("completed")[0].task_id, task.task_id)
        self.assertEqual(self.core.tasks.list("pending"), [])

    def test_state_entity_types_include_deleted_entities(self) -> None:
        self.core.state.apply("profile", "main", {"name": "Baymax"}, base_version=0)
        self.core.state.delete("profile", "main", base_version=1)
        self.assertEqual(self.core.state.entity_types(), ["profile"])
        self.assertEqual(self.core.state.list("profile"), [])
        self.assertTrue(self.core.state.list("profile", include_deleted=True)[0].deleted)

    def test_demo_data_is_safe_to_load_more_than_once(self) -> None:
        self.assertEqual(load_demo_data(self.core), (3, 2))
        self.assertEqual(load_demo_data(self.core), (3, 2))
        self.assertEqual(len(self.core.state.list("note")), 3)
        self.assertEqual(len(self.core.tasks.list()), 2)
        self.assertEqual(len(self.core.pull_events(0)), 3)

    def test_first_run_demo_seed_only_runs_for_an_empty_database(self) -> None:
        self.assertTrue(seed_demo_data_if_empty(self.core))
        self.assertEqual(len(self.core.state.list("note")), 3)
        self.assertEqual(len(self.core.tasks.list()), 2)
        self.assertFalse(seed_demo_data_if_empty(self.core))

    def test_first_run_demo_seed_preserves_existing_user_data(self) -> None:
        self.core.state.apply("note", "my-note", {"text": "My own note"}, base_version=0)
        self.assertFalse(seed_demo_data_if_empty(self.core))
        self.assertEqual([record.entity_id for record in self.core.state.list("note")],
                         ["my-note"])
        self.assertEqual(self.core.tasks.list(), [])

    def test_button_animation_color_interpolation(self) -> None:
        self.assertEqual(_blend_color("#000000", "#ffffff", 0), "#000000")
        self.assertEqual(_blend_color("#000000", "#ffffff", 0.5), "#808080")
        self.assertEqual(_blend_color("#000000", "#ffffff", 1), "#ffffff")

    def test_state_survives_reopening_database(self) -> None:
        self.core.state.apply("setting", "theme", {"value": "dark"}, base_version=0)
        self.core.close()
        self.core = WindowsCore(self.database_path, device_id="test-device")
        self.assertEqual(self.core.recover(), "ok")
        self.assertEqual(self.core.state.get("setting", "theme").payload, {"value": "dark"})


if __name__ == "__main__":
    unittest.main()
