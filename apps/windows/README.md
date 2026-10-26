# BAYMAX Windows Core

The Windows Core is Member 1's backend: it stores canonical state, records durable events, and coordinates task ownership. Its Tkinter desktop app provides an overview, state editor, task manager, and event log. Android lifecycle code, WireGuard, and model-provider internals remain outside this app.

## What is implemented

- SQLite persistence with WAL mode, foreign-key checks, full synchronous commits, and a schema-versioned migration.
- An append-only event log with stable event IDs, local sequence numbers, parent/base-version metadata, source-device metadata, idempotency keys, and delete tombstones.
- A canonical state view updated in the same database transaction as each event. State writes support optimistic version checks.
- Atomic event batches for sync: if any event conflicts, the complete batch rolls back.
- A task lifecycle (`pending` → `running` → `completed` / `failed` / `cancelled`) with durable checkpoints, lease renewal, monotonically increasing fencing epochs, and idempotent create/checkpoint/finish operations.
- An integrity-check recovery entry point. Opening SQLite replays its WAL before the application accepts requests.
- A `WindowsCore` composition root that exposes state, tasks, durable sync push/pull, and recovery independently of AI or network availability.
- A native Windows desktop interface for viewing and editing saved state, managing task lifecycles, and inspecting durable events.
- An optional demo-data action that adds repeat-safe example notes and tasks only when the user chooses it.
- Subtle grayscale hover/press button animations and helpful empty-state placeholders.

## Run the desktop app

From the repository root, start the app with Python:

```powershell
python baymax_windows.py
```

To rebuild the standalone Windows executable:

```powershell
python -m PyInstaller --clean --onefile --windowed --name BAYMAX baymax_windows.py
```

The executable is created at `dist\BAYMAX.exe`. The app stores its database at `%LOCALAPPDATA%\BAYMAX\baymax.db`.

On the first launch with a completely empty database, BAYMAX adds three example notes and two example tasks so the screens are populated. Existing databases with saved events or tasks are left unchanged. You can also choose **Load demo data** on the Overview page to add the examples manually; repeating the action does not create duplicates. Empty State, Tasks, and Event log pages show a short hint when they have no records. Buttons animate subtly when hovered or pressed.

## Use the Core in Python

```python
from apps.windows.services import WindowsCore

with WindowsCore("data/baymax.db", device_id="my-windows-device") as baymax:
    event = baymax.state.apply(
        "note",
        "welcome",
        {"text": "Hello from BAYMAX"},
        base_version=0,
    )
    note = baymax.state.get("note", "welcome")
    print(event.sequence, note.payload)
```

Task workers claim one pending task at a time. Every write by a worker must include the task's current `lease_epoch`; a worker whose lease expired or was reassigned is rejected.

## Contracts and running tests

The protobuf files in `protocol/proto/` define the transport contract. The Python services in this package are transport-independent so a gRPC adapter can call them without moving database logic into the network layer. No generated gRPC code or third-party runtime is needed to use the local core. The desktop interface uses Python's built-in Tkinter toolkit.

Run the standard-library test suite from the repository root:

```powershell
C:\Users\Parth\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests -t . -v
```

## Important invariants

1. A committed event and its canonical-state projection are written together.
2. Sequence numbers are assigned by SQLite and are monotonically increasing within this Windows Core database.
3. `base_version` is checked against the current entity version before writing.
4. Tombstones remain in canonical state and in the event log; ordinary state queries hide deleted entries.
5. A task's `lease_epoch` increases each time it is claimed, including after an expired lease. Old epochs cannot checkpoint, renew, or finish.
6. Task checkpoints and state changes survive process restarts because they are committed to SQLite.
