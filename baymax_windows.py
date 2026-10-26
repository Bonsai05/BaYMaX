"""Start the BAYMAX Windows desktop application."""

import os
from pathlib import Path
import sqlite3
import tkinter as tk
from tkinter import messagebox

from apps.windows.services import WindowsCore
from apps.windows.ui.app import run_desktop_app


def main() -> int:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "BAYMAX could not start",
            "Windows did not provide a LOCALAPPDATA folder for this user.",
            parent=root,
        )
        root.destroy()
        return 1

    data_folder = Path(local_app_data) / "BAYMAX"
    database_path = data_folder / "baymax.db"
    try:
        core = WindowsCore(database_path)
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("BAYMAX could not open its database", str(exc), parent=root)
        root.destroy()
        return 1

    try:
        run_desktop_app(core)
    finally:
        core.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
