"""Tkinter desktop interface for the local BAYMAX Windows Core."""

from __future__ import annotations

import json
import os
import socket
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any, Callable
import uuid

from apps.windows.core.errors import CoreError
from apps.windows.core.events.models import StateRecord
from apps.windows.core.tasks.models import TaskRecord
from apps.windows.services import WindowsCore
from apps.windows.ui.demo_data import load_demo_data, seed_demo_data_if_empty


_BACKGROUND = "#f3f3f1"
_PANEL = "#ffffff"
_PANEL_ALT = "#e8e8e6"
_TEXT = "#171717"
_MUTED = "#696967"
_ACCENT = "#171717"
_SIDEBAR = "#111111"
_SIDEBAR_TEXT = "#f5f5f3"


def _blend_color(start: str, end: str, progress: float) -> str:
    start_rgb = tuple(int(start[index:index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index:index + 2], 16) for index in (1, 3, 5))
    channels = tuple(
        round(first + (last - first) * progress)
        for first, last in zip(start_rgb, end_rgb)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


class BaymaxApp:
    def __init__(self, root: tk.Tk, core: WindowsCore) -> None:
        self.root = root
        self.core = core
        self.worker_id = (
            f"desktop-{os.environ.get('USERNAME') or socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        )
        self.selected_task_id: str | None = None
        self.selected_state: StateRecord | None = None
        self._nav_button_states: dict[str, dict[str, Any]] = {}
        self.root.title("BAYMAX | Personal Core")
        self.root.geometry("1120x740")
        self.root.minsize(900, 620)
        self.root.configure(background=_BACKGROUND)
        self._configure_style()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._build_shell()
        self.show_page("Overview")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background=_BACKGROUND)
        style.configure("Panel.TFrame", background=_PANEL)
        style.configure("TLabel", background=_BACKGROUND, foreground=_TEXT)
        style.configure("Panel.TLabel", background=_PANEL, foreground=_TEXT)
        style.configure("Muted.TLabel", background=_BACKGROUND, foreground=_MUTED)
        style.configure("PanelMuted.TLabel", background=_PANEL, foreground=_MUTED)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 24), foreground=_TEXT)
        style.configure("CardValue.TLabel", background=_PANEL, foreground=_ACCENT,
                        font=("Segoe UI Semibold", 23))
        style.configure("CardTitle.TLabel", background=_PANEL, foreground=_MUTED)
        style.configure("Nav.TButton", background=_SIDEBAR, foreground="#bdbdbb",
                        padding=(16, 12), anchor="w", borderwidth=0, relief="flat")
        style.map("Nav.TButton", background=[("active", "#292929")],
                  foreground=[("active", _SIDEBAR_TEXT)])
        style.configure("NavActive.TButton", background="#292929",
                        foreground=_SIDEBAR_TEXT, padding=(16, 12), anchor="w",
                        borderwidth=0, relief="flat")
        style.map("NavActive.TButton", background=[("active", "#353535")])
        style.configure("Accent.TButton", background=_ACCENT, foreground="#ffffff",
                        padding=(14, 9), borderwidth=0, relief="flat")
        style.map("Accent.TButton", background=[("active", "#353535")],
                  foreground=[("active", "#ffffff")])
        style.configure("TButton", padding=(12, 8), background=_PANEL_ALT,
                        foreground=_TEXT, borderwidth=0, relief="flat")
        style.map("TButton", background=[("active", "#d6d6d4")])
        style.configure("TEntry", fieldbackground=_PANEL, foreground=_TEXT,
                        insertcolor=_TEXT, padding=8)
        style.configure("Treeview", background=_PANEL, fieldbackground=_PANEL,
                        foreground=_TEXT, rowheight=34, borderwidth=0)
        style.configure("Treeview.Heading", background=_PANEL_ALT, foreground=_TEXT,
                        padding=(10, 10), font=("Segoe UI Semibold", 9))
        style.map("Treeview", background=[("selected", "#d8d8d6")],
                  foreground=[("selected", _TEXT)])
        style.configure("TNotebook", background=_BACKGROUND, borderwidth=0)
        style.configure("TNotebook.Tab", background=_PANEL, foreground=_MUTED,
                        padding=(12, 8))
        style.map("TNotebook.Tab", selected=[("selected", _PANEL_ALT)],
                  foreground=[("selected", _TEXT)])

    def _build_shell(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        sidebar = tk.Frame(outer, bg=_SIDEBAR, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        brand = tk.Frame(sidebar, bg=_SIDEBAR)
        brand.pack(fill="x", padx=22, pady=(30, 36))
        tk.Label(brand, text="BAYMAX", bg=_SIDEBAR, fg=_SIDEBAR_TEXT,
                 font=("Segoe UI Semibold", 19)).pack(anchor="w")
        tk.Label(brand, text="PERSONAL CORE", bg=_SIDEBAR, fg="#a9a9a7",
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

        self.nav_buttons: dict[str, tk.Button] = {}
        for page in ("Overview", "State", "Tasks", "Event log"):
            button = self._animated_button(
                sidebar,
                f"   {page}",
                lambda selected=page: self.show_page(selected),
                variant="navigation",
            )
            button.configure(anchor="w", padx=12, pady=11)
            button.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[page] = button

        tk.Label(sidebar, text="LOCAL-FIRST  •  PRIVATE", bg=_SIDEBAR, fg="#a9a9a7",
                 font=("Segoe UI", 8)).pack(side="bottom", anchor="w", padx=20, pady=18)

        self.content = ttk.Frame(outer)
        self.content.pack(side="left", fill="both", expand=True, padx=32, pady=28)

    def show_page(self, page: str) -> None:
        for name, state in self._nav_button_states.items():
            state["base"] = "#292929" if name == page else _SIDEBAR
            state["base_foreground"] = _SIDEBAR_TEXT if name == page else "#bdbdbb"
            self._animate_button(self.nav_buttons[name], state, state["base"])
        for child in self.content.winfo_children():
            child.destroy()
        ttk.Label(self.content, text=page, style="Title.TLabel").pack(anchor="w", pady=(0, 5))
        descriptions = {
            "Overview": "Your local BAYMAX Core at a glance.",
            "State": "View and manage information saved on this device.",
            "Tasks": "Create tasks, claim work, save checkpoints, and review progress.",
            "Event log": "Browse the durable history of changes recorded by this Core.",
        }
        ttk.Label(self.content, text=descriptions[page], style="Muted.TLabel"
                  ).pack(anchor="w", pady=(2, 22))
        try:
            {"Overview": self._show_overview, "State": self._show_state,
             "Tasks": self._show_tasks, "Event log": self._show_events}[page]()
        except (CoreError, OSError, ValueError) as exc:
            messagebox.showerror("BAYMAX could not load this page", str(exc), parent=self.root)

    def _animated_button(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], None],
        *,
        variant: str = "secondary",
    ) -> tk.Button:
        palettes = {
            "primary": (_ACCENT, "#393939", "#555555", "#ffffff", "#ffffff"),
            "secondary": (_PANEL_ALT, "#d5d5d3", "#c1c1bf", _TEXT, _TEXT),
            "navigation": (_SIDEBAR, "#292929", "#3a3a3a", "#bdbdbb", _SIDEBAR_TEXT),
        }
        base, hover, pressed, foreground, active_foreground = palettes[variant]
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=base,
            fg=foreground,
            activebackground=hover,
            activeforeground=active_foreground,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            cursor="hand2",
            font=("Segoe UI Semibold", 9),
            padx=14,
            pady=9,
            takefocus=True,
        )
        state: dict[str, Any] = {
            "base": base,
            "hover": hover,
            "pressed": pressed,
            "base_foreground": foreground,
            "hover_foreground": active_foreground,
            "pressed_foreground": active_foreground,
            "current": base,
            "after_id": None,
        }

        button.bind("<Enter>", lambda _event: self._animate_button(button, state, state["hover"]))
        button.bind("<Leave>", lambda _event: self._animate_button(button, state, state["base"]))
        button.bind("<ButtonPress-1>", lambda _event: self._animate_button(button, state, state["pressed"]))
        button.bind("<ButtonRelease-1>", lambda _event: self._animate_button(button, state, state["hover"]))

        def cancel_animation(event: tk.Event) -> None:
            if event.widget is button and state["after_id"] is not None:
                button.after_cancel(state["after_id"])
                state["after_id"] = None

        button.bind("<Destroy>", cancel_animation, add="+")
        if variant == "navigation":
            self._nav_button_states[text.strip()] = state
        return button

    def _animate_button(self, button: tk.Button, state: dict[str, Any], target: str) -> None:
        if state["after_id"] is not None:
            button.after_cancel(state["after_id"])
            state["after_id"] = None
        start = state["current"]
        if target == state["base"]:
            foreground = state["base_foreground"]
        elif target == state["pressed"]:
            foreground = state["pressed_foreground"]
        else:
            foreground = state["hover_foreground"]
        steps = 5

        def animate(step: int) -> None:
            if not button.winfo_exists():
                return
            color = _blend_color(start, target, step / steps)
            button.configure(
                bg=color,
                fg=foreground,
                activebackground=target,
                activeforeground=foreground,
            )
            state["current"] = color
            if step < steps:
                state["after_id"] = button.after(14, animate, step + 1)
            else:
                state["after_id"] = None

        animate(1)

    def _empty_placeholder(self, parent: tk.Widget, tree: ttk.Treeview, text: str) -> None:
        if not tree.get_children():
            tk.Label(
                parent,
                text=text,
                bg=_PANEL,
                fg=_MUTED,
                font=("Segoe UI", 11),
                justify="center",
                padx=24,
                pady=24,
            ).place(relx=0.5, rely=0.5, anchor="center")

    def _card(self, parent: tk.Widget, title: str, value: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Panel.TFrame", padding=(20, 17))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=value, style="CardValue.TLabel").pack(anchor="w", pady=(8, 0))
        return card

    def _show_overview(self) -> None:
        state_count = sum(
            len(self.core.state.list(entity_type))
            for entity_type in self.core.state.entity_types()
        )
        tasks = self.core.tasks.list()
        active = sum(task.status in {"pending", "running"} for task in tasks)
        cards = ttk.Frame(self.content)
        cards.pack(fill="x")
        for title, value in (
            ("SAVED ITEMS", str(state_count)),
            ("ACTIVE TASKS", str(active)),
            ("TOTAL TASKS", str(len(tasks))),
        ):
            self._card(cards, title, value).pack(side="left", fill="x", expand=True,
                                                padx=(0, 12))

        panel = ttk.Frame(self.content, style="Panel.TFrame", padding=22)
        panel.pack(fill="x", pady=(24, 0))
        ttk.Label(panel, text="Your data stays on this device", style="Panel.TLabel",
                  font=("Segoe UI Semibold", 13)).pack(anchor="w")
        ttk.Label(panel, text="BAYMAX stores state and task history in its local SQLite database. "
                  "No AI provider or network connection is required for these features.",
                  style="PanelMuted.TLabel", wraplength=720, justify="left"
                  ).pack(anchor="w", pady=(9, 17))
        self._animated_button(
            panel, "Open saved state", lambda: self.show_page("State"), variant="primary"
        ).pack(side="left")
        self._animated_button(
            panel, "View tasks", lambda: self.show_page("Tasks")
        ).pack(side="left", padx=(10, 0))
        self._animated_button(
            panel, "Load demo data", self._load_demo_data
        ).pack(side="left", padx=(10, 0))

        health = ttk.Frame(self.content, style="Panel.TFrame", padding=18)
        health.pack(fill="x", pady=(16, 0))
        ttk.Label(health, text="DATABASE", style="PanelMuted.TLabel").pack(anchor="w")
        ttk.Label(health, text=self.core.recover(), style="Panel.TLabel",
                  font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(5, 0))

    def _load_demo_data(self) -> None:
        if not messagebox.askyesno(
            "Load demo data",
            "Add three example notes and two example tasks to this local database?",
            parent=self.root,
        ):
            return
        try:
            notes_added, tasks_added = load_demo_data(self.core)
        except CoreError as exc:
            messagebox.showerror("Could not load demo data", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "Demo data ready",
            f"{notes_added} example notes and {tasks_added} example tasks are ready.\n"
            "The demo items are marked in their saved data, and loading them again will not duplicate them.",
            parent=self.root,
        )
        self.show_page("Overview")

    def _tree(
        self,
        parent: tk.Widget,
        columns: tuple[tuple[str, str, int], ...],
    ) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=[column[0] for column in columns],
                            show="headings", selectmode="browse")
        for name, title, width in columns:
            tree.heading(name, text=title)
            tree.column(name, width=width, minwidth=65, anchor="w")
        return tree

    def _selected_iid(self, tree: ttk.Treeview) -> str | None:
        selection = tree.selection()
        return selection[0] if selection else None

    def _show_state(self) -> None:
        toolbar = ttk.Frame(self.content)
        toolbar.pack(fill="x", pady=(0, 12))
        self._animated_button(
            toolbar, "+  Add state", lambda: self._state_dialog(), variant="primary"
        ).pack(side="left")
        self._animated_button(toolbar, "Edit", self._edit_state).pack(side="left", padx=8)
        self._animated_button(toolbar, "Delete", self._delete_state).pack(side="left")
        self._animated_button(
            toolbar, "Refresh", lambda: self.show_page("State")
        ).pack(side="right")

        panel = ttk.Frame(self.content, style="Panel.TFrame", padding=10)
        panel.pack(fill="both", expand=True)
        self.state_tree = self._tree(
            panel,
            (("type", "TYPE", 160), ("id", "ID", 220), ("version", "VERSION", 90),
             ("payload", "SAVED DATA", 450)),
        )
        self.state_tree.pack(fill="both", expand=True)
        self.state_records: dict[str, StateRecord] = {}
        for entity_type in self.core.state.entity_types():
            for record in self.core.state.list(entity_type):
                key = f"{record.entity_type}\x1f{record.entity_id}"
                self.state_records[key] = record
                self.state_tree.insert(
                    "", "end", iid=key,
                    values=(record.entity_type, record.entity_id, record.version,
                            json.dumps(record.payload, ensure_ascii=False, sort_keys=True)),
                )
        self._empty_placeholder(
            panel,
            self.state_tree,
            "Nothing saved yet\nAdd your first item or load demo data from Overview.",
        )
        self.state_tree.bind("<<TreeviewSelect>>", self._select_state)

    def _select_state(self, _event: tk.Event) -> None:
        del _event
        iid = self._selected_iid(self.state_tree)
        self.selected_state = self.state_records.get(iid) if iid else None

    def _state_dialog(self, record: StateRecord | None = None) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit saved state" if record else "Add saved state")
        dialog.configure(background=_BACKGROUND)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("520x430")
        form = ttk.Frame(dialog, padding=22)
        form.pack(fill="both", expand=True)
        ttk.Label(form, text="Edit state" if record else "New state",
                  style="Title.TLabel").pack(anchor="w", pady=(0, 18))

        type_var = tk.StringVar(value=record.entity_type if record else "note")
        id_var = tk.StringVar(value=record.entity_id if record else "")
        for label, variable in (("Type (for example note or setting)", type_var),
                                ("ID", id_var)):
            ttk.Label(form, text=label).pack(anchor="w", pady=(0, 4))
            entry = ttk.Entry(form, textvariable=variable)
            entry.pack(fill="x", pady=(0, 12))
            if record:
                entry.configure(state="disabled")

        ttk.Label(form, text="Data (JSON object)").pack(anchor="w", pady=(0, 4))
        payload_text = tk.Text(form, height=8, bg=_PANEL, fg=_TEXT, insertbackground=_TEXT,
                               relief="flat", padx=10, pady=9, font=("Cascadia Code", 10))
        payload_text.pack(fill="both", expand=True)
        if record:
            payload_text.insert("1.0", json.dumps(record.payload, indent=2, ensure_ascii=False))
        else:
            payload_text.insert("1.0", '{\n  "text": ""\n}')

        def save() -> None:
            entity_type, entity_id = type_var.get().strip(), id_var.get().strip()
            try:
                payload = json.loads(payload_text.get("1.0", "end").strip())
                if not isinstance(payload, dict):
                    raise ValueError("Data must be a JSON object surrounded by { }.")
                if record:
                    self.core.state.apply(
                        entity_type, entity_id, payload, base_version=record.version
                    )
                else:
                    self.core.state.apply(entity_type, entity_id, payload, base_version=0)
            except (json.JSONDecodeError, ValueError, CoreError) as exc:
                messagebox.showerror("Could not save state", str(exc), parent=dialog)
                return
            dialog.destroy()
            self.show_page("State")

        self._animated_button(form, "Save state", save, variant="primary"
                              ).pack(anchor="e", pady=(12, 0))
        dialog.wait_window()

    def _edit_state(self) -> None:
        if self.selected_state is None:
            messagebox.showinfo("Select state", "Select a row to edit.", parent=self.root)
            return
        self._state_dialog(self.selected_state)

    def _delete_state(self) -> None:
        record = self.selected_state
        if record is None:
            messagebox.showinfo("Select state", "Select a row to delete.", parent=self.root)
            return
        if not messagebox.askyesno(
            "Delete saved state",
            f"Delete {record.entity_type} / {record.entity_id}?\n"
            "The deletion is kept in the event history.",
            parent=self.root,
        ):
            return
        try:
            self.core.state.delete(record.entity_type, record.entity_id,
                                   base_version=record.version)
        except CoreError as exc:
            messagebox.showerror("Could not delete state", str(exc), parent=self.root)
            return
        self.selected_state = None
        self.show_page("State")

    def _show_tasks(self) -> None:
        toolbar = ttk.Frame(self.content)
        toolbar.pack(fill="x", pady=(0, 12))
        self._animated_button(
            toolbar, "+  New task", self._new_task_dialog, variant="primary"
        ).pack(side="left")
        self._animated_button(toolbar, "Claim next", self._claim_task
                              ).pack(side="left", padx=8)
        self._animated_button(toolbar, "Checkpoint", self._checkpoint_task
                              ).pack(side="left")
        self._animated_button(toolbar, "Finish task", self._finish_task
                              ).pack(side="left", padx=8)
        self._animated_button(
            toolbar, "Refresh", lambda: self.show_page("Tasks")
        ).pack(side="right")

        panel = ttk.Frame(self.content, style="Panel.TFrame", padding=10)
        panel.pack(fill="both", expand=True)
        self.task_tree = self._tree(
            panel,
            (("id", "TASK ID", 245), ("type", "TYPE", 150), ("status", "STATUS", 110),
             ("owner", "WORKER", 150), ("epoch", "FENCE", 70), ("updated", "UPDATED", 150)),
        )
        self.task_tree.pack(fill="both", expand=True)
        self.task_records: dict[str, TaskRecord] = {}
        for task in self.core.tasks.list():
            self.task_records[task.task_id] = task
            self.task_tree.insert(
                "", "end", iid=task.task_id,
                values=(task.task_id, task.task_type, task.status,
                        task.lease_owner or "—", task.lease_epoch,
                        self._format_time(task.updated_at)),
            )
        self._empty_placeholder(
            panel,
            self.task_tree,
            "No tasks yet\nCreate a task or load demo examples from Overview.",
        )
        self.task_tree.bind("<<TreeviewSelect>>", self._select_task)
        self.task_tree.bind("<Double-1>", self._show_task_details)
        ttk.Label(self.content, text=f"Worker ID: {self.worker_id}  •  Double-click a task for details",
                  style="Muted.TLabel").pack(anchor="w", pady=(9, 0))

    def _format_time(self, timestamp: float) -> str:
        from datetime import datetime

        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")

    def _select_task(self, _event: tk.Event) -> None:
        del _event
        self.selected_task_id = self._selected_iid(self.task_tree)

    def _new_task_dialog(self) -> None:
        task_type = simpledialog.askstring(
            "New task", "Task type:", initialvalue="general", parent=self.root
        )
        if task_type is None:
            return
        payload_text = simpledialog.askstring(
            "New task", "Task data as a JSON object:", initialvalue="{}",
            parent=self.root,
        )
        if payload_text is None:
            return
        try:
            payload = json.loads(payload_text)
            if not isinstance(payload, dict):
                raise ValueError("Task data must be a JSON object.")
            self.core.tasks.create(task_type, payload)
        except (json.JSONDecodeError, ValueError, CoreError) as exc:
            messagebox.showerror("Could not create task", str(exc), parent=self.root)
            return
        self.show_page("Tasks")

    def _selected_task(self) -> TaskRecord | None:
        task_id = self.selected_task_id
        if task_id is None:
            messagebox.showinfo("Select a task", "Select a task row first.", parent=self.root)
            return None
        try:
            return self.core.tasks.get(task_id)
        except CoreError as exc:
            messagebox.showerror("Could not load task", str(exc), parent=self.root)
            return None

    def _claim_task(self) -> None:
        try:
            task = self.core.tasks.claim_next(self.worker_id, lease_seconds=300)
        except CoreError as exc:
            messagebox.showerror("Could not claim task", str(exc), parent=self.root)
            return
        if task is None:
            messagebox.showinfo("No task available", "There are no pending tasks to claim.",
                                parent=self.root)
        else:
            messagebox.showinfo("Task claimed", f"Claimed {task.task_id} for 5 minutes.",
                                parent=self.root)
        self.show_page("Tasks")

    def _checkpoint_task(self) -> None:
        task = self._selected_task()
        if task is None:
            return
        if task.status != "running" or task.lease_owner != self.worker_id:
            messagebox.showinfo("Task not owned",
                                "Claim this task on this computer before saving a checkpoint.",
                                parent=self.root)
            return
        value = simpledialog.askstring(
            "Save checkpoint", "Checkpoint data as a JSON object:", initialvalue="{}",
            parent=self.root,
        )
        if value is None:
            return
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict):
                raise ValueError("Checkpoint data must be a JSON object.")
            self.core.tasks.checkpoint(
                task.task_id, self.worker_id, task.lease_epoch, payload,
                idempotency_key=f"desktop-checkpoint-{os.urandom(12).hex()}",
            )
        except (json.JSONDecodeError, ValueError, CoreError) as exc:
            messagebox.showerror("Could not save checkpoint", str(exc), parent=self.root)
            return
        messagebox.showinfo("Checkpoint saved", "The checkpoint was saved to the database.",
                            parent=self.root)
        self.show_page("Tasks")

    def _finish_task(self) -> None:
        task = self._selected_task()
        if task is None:
            return
        if task.status != "running" or task.lease_owner != self.worker_id:
            messagebox.showinfo("Task not owned",
                                "Only the current worker can finish a running task.",
                                parent=self.root)
            return
        status = simpledialog.askstring(
            "Finish task", "Enter completed, failed, or cancelled:",
            initialvalue="completed", parent=self.root,
        )
        if status is None:
            return
        status = status.strip().lower()
        if status not in {"completed", "failed", "cancelled"}:
            messagebox.showerror("Invalid status", "Choose completed, failed, or cancelled.",
                                 parent=self.root)
            return
        result: dict[str, Any] | None = None
        error: str | None = None
        if status == "failed":
            error = simpledialog.askstring("Failure reason", "Describe why this task failed:",
                                           parent=self.root)
            if not error:
                return
        elif status == "completed":
            result_text = simpledialog.askstring(
                "Task result", "Result as a JSON object:", initialvalue="{}",
                parent=self.root,
            )
            if result_text is None:
                return
            try:
                result = json.loads(result_text)
                if not isinstance(result, dict):
                    raise ValueError("Result must be a JSON object.")
            except (json.JSONDecodeError, ValueError) as exc:
                messagebox.showerror("Invalid result", str(exc), parent=self.root)
                return
        try:
            self.core.tasks.finish(
                task.task_id, self.worker_id, task.lease_epoch, status,
                result=result, error=error,
                idempotency_key=f"desktop-finish-{os.urandom(12).hex()}",
            )
        except CoreError as exc:
            messagebox.showerror("Could not finish task", str(exc), parent=self.root)
            return
        self.show_page("Tasks")

    def _show_task_details(self, _event: tk.Event) -> None:
        del _event
        task = self._selected_task()
        if task is None:
            return
        details = {
            "task": {
                "id": task.task_id,
                "type": task.task_type,
                "status": task.status,
                "payload": task.payload,
                "result": task.result,
                "error": task.error,
                "worker": task.lease_owner,
                "lease_epoch": task.lease_epoch,
                "lease_until": task.lease_until,
            },
            "checkpoints": [
                {"lease_epoch": checkpoint.lease_epoch, "payload": checkpoint.payload,
                 "created_at": self._format_time(checkpoint.created_at)}
                for checkpoint in self.core.tasks.checkpoints(task.task_id)
            ],
            "history": self.core.tasks.history(task.task_id),
        }
        self._show_json_window(f"Task {task.task_id}", details)

    def _show_events(self) -> None:
        panel = ttk.Frame(self.content, style="Panel.TFrame", padding=10)
        panel.pack(fill="both", expand=True)
        tree = self._tree(
            panel,
            (("sequence", "SEQ", 80), ("event", "EVENT", 240),
             ("entity", "ENTITY", 240), ("device", "SOURCE DEVICE", 150),
             ("tombstone", "DELETED", 80)),
        )
        tree.pack(fill="both", expand=True)
        events = self.core.pull_events(0, limit=10000)
        event_map = {}
        for event in events:
            tree.insert(
                "", "end", iid=event.event_id,
                values=(event.sequence, event.event_type,
                        f"{event.entity_type} / {event.entity_id}",
                        event.source_device_id, "Yes" if event.tombstone else "No"),
            )
            event_map[event.event_id] = event
        self._empty_placeholder(
            panel,
            tree,
            "No events yet\nSaved state changes will appear here.",
        )

        details = tk.Text(panel, height=8, bg=_PANEL_ALT, fg=_TEXT, insertbackground=_TEXT,
                          relief="flat", padx=10, pady=9, font=("Cascadia Code", 9),
                          state="disabled")
        details.pack(fill="x", pady=(10, 0))

        def show_event(_event: tk.Event) -> None:
            del _event
            selected = tree.selection()
            if not selected:
                return
            value = event_map[selected[0]]
            details.configure(state="normal")
            details.delete("1.0", "end")
            details.insert("1.0", json.dumps(value.__dict__, indent=2, ensure_ascii=False))
            details.configure(state="disabled")

        tree.bind("<<TreeviewSelect>>", show_event)

    def _show_json_window(self, title: str, value: dict[str, Any]) -> None:
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("720x560")
        text = tk.Text(window, bg=_PANEL, fg=_TEXT, insertbackground=_TEXT,
                       relief="flat", padx=16, pady=16, font=("Cascadia Code", 10))
        text.pack(fill="both", expand=True, padx=14, pady=14)
        text.insert("1.0", json.dumps(value, indent=2, ensure_ascii=False))
        text.configure(state="disabled")

    def _close(self) -> None:
        self.core.close()
        self.root.destroy()


def run_desktop_app(core: WindowsCore) -> None:
    seed_demo_data_if_empty(core)
    root = tk.Tk()
    BaymaxApp(root, core)
    root.mainloop()
