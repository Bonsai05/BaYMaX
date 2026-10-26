"""Explicit, repeatable sample content for the BAYMAX desktop demo."""

from __future__ import annotations

import uuid

from apps.windows.services import WindowsCore


_DEMO_ITEMS = (
    (
        "welcome",
        {
            "title": "Welcome to BAYMAX",
            "text": "Your personal assistant keeps saved information on this device.",
            "category": "getting-started",
            "demo": True,
        },
    ),
    (
        "daily-plan",
        {
            "title": "Today's plan",
            "text": "Review notes, organize tasks, and take a short break.",
            "category": "example",
            "demo": True,
        },
    ),
    (
        "favorite-color",
        {
            "title": "Favorite color",
            "text": "Soft white",
            "category": "example",
            "demo": True,
        },
    ),
)

_DEMO_TASKS = (
    (
        "organize-notes",
        "Organize notes",
        {"description": "Group notes into a few useful categories.", "demo": True},
    ),
    (
        "plan-a-break",
        "Plan a break",
        {"description": "Choose a time for a relaxing break.", "demo": True},
    ),
)


def load_demo_data(core: WindowsCore) -> tuple[int, int]:
    """Add stable sample notes and tasks without duplicating them on repeat clicks."""
    for item_id, payload in _DEMO_ITEMS:
        operation_key = f"baymax-demo-state-{item_id}"
        core.state.apply(
            "note",
            f"demo-{item_id}",
            payload,
            base_version=0,
            event_id=str(uuid.uuid5(uuid.NAMESPACE_URL, operation_key)),
            idempotency_key=operation_key,
        )

    for task_id, task_type, payload in _DEMO_TASKS:
        core.tasks.create(
            task_type,
            payload,
            idempotency_key=f"baymax-demo-task-{task_id}",
        )

    return len(_DEMO_ITEMS), len(_DEMO_TASKS)


def seed_demo_data_if_empty(core: WindowsCore) -> bool:
    """Seed a first-run database without mixing samples into an existing user's data."""
    if core.pull_events(0, limit=1) or core.tasks.list():
        return False
    load_demo_data(core)
    return True
