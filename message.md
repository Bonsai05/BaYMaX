# Member 1 - Windows Core changes

This file lists the small changes in simple language so it is easy to see what was added.

## What changed

- Added the Windows Core entry point. It connects state, task, storage, and recovery features together.
- Added a local SQLite database. It uses WAL mode so committed work can be recovered after a process or computer stops unexpectedly.
- Added an event log. Each saved change gets an ID and an increasing sequence number.
- Added version checks for state updates. A stale update is rejected instead of silently overwriting newer data.
- Added delete tombstones. Deleted records stay in the history but do not appear in normal state lists.
- Added atomic sync batches. A batch is saved completely, or not saved at all.
- Added task creation and task status history.
- Added worker leases. If a worker stops responding, another worker can claim the task.
- Added lease epochs (fencing). An old worker cannot change a task after it has been reassigned.
- Added task checkpoints so work can resume after a restart.
- Added idempotency keys so safely retried requests do not create duplicate tasks, checkpoints, or task completions.
- Added a database integrity check and a schema version for future migrations.
- Added protobuf message/service contracts for state reads, event synchronization, and task calls.
- Added tests for state, sync, task ownership, retries, and reopening the database.
- Added a Windows Core README with a quick-start example and the important rules.
- Added a Windows desktop interface with an overview, state editor, task manager, and event log.
- Restyled the desktop interface with an original black-and-white look, dark navigation, brighter content panels, and clearer active-page, button, and table states.
- Added a one-click demo-data option with three example notes and two example tasks; it asks first and safely avoids duplicate demo entries.
- Added automatic first-run sample data for a completely empty database; existing user data is left untouched.
- Added subtle grayscale hover and press animations to navigation and action buttons.
- Added friendly empty-state placeholders for state, tasks, and event history.
- Changed the BAYMAX launcher so it opens the desktop window and reports database startup errors visibly.
- Updated the PyInstaller instructions to build a windowed app without a terminal window.

## Files to look at

- `apps/windows/` contains the Windows Core application services.
- `apps/windows/storage/` contains SQLite setup and repository code.
- `apps/windows/core/events/` contains event and canonical-state behavior.
- `apps/windows/core/tasks/` contains task lifecycle and lease behavior.
- `apps/windows/ui/app.py` contains the desktop screens.
- `baymax_windows.py` is the launcher used by Python and PyInstaller.
- `tests/unit/test_windows_core.py` checks the main behavior.

## Not included

- The desktop interface is a local control panel; it does not add AI chat, Android connectivity, or network services.
- Android code, WireGuard, and AI/model internals are outside Member 1's ownership.
- The Python service boundary is ready for a transport adapter; generated gRPC server code is not included.
