"""Reference desktop runner that reuses the shared pipeline service layer.

The current implementation uses Tkinter only as a lightweight built-in
adapter. The pipeline, reporting, and worker boundaries are intentionally
kept framework-agnostic so this module can be replaced by a Qt frontend
later without changing the core services.
"""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import threading

from .pipeline import run_pipeline_from_file
from .reporting import ProgressEvent, QueueReporter


class GuiApplication:
    """Small reference desktop runner suitable for a later Qt migration."""

    def __init__(self, root, initial_config: Path | None = None) -> None:
        self.root = root
        self.root.title("PhoPS")
        self.root.geometry("900x560")
        self.event_queue: Queue[ProgressEvent] = Queue()
        self.running = False
        self.worker: threading.Thread | None = None

        import tkinter as tk
        from tkinter import filedialog

        self._filedialog = filedialog
        self.config_var = tk.StringVar(value=str(initial_config or Path("config.yaml")))
        self.status_var = tk.StringVar(value="Idle")

        outer = tk.Frame(root, padx=12, pady=12)
        outer.pack(fill="both", expand=True)

        controls = tk.Frame(outer)
        controls.pack(fill="x")

        tk.Label(controls, text="Config").pack(side="left")
        tk.Entry(controls, textvariable=self.config_var, width=70).pack(side="left", fill="x", expand=True, padx=(8, 8))
        tk.Button(controls, text="Browse", command=self.browse_config).pack(side="left")
        self.run_button = tk.Button(controls, text="Run Pipeline", command=self.start_pipeline)
        self.run_button.pack(side="left", padx=(8, 0))

        status = tk.Frame(outer, pady=8)
        status.pack(fill="x")
        tk.Label(status, textvariable=self.status_var).pack(side="left")

        self.log_widget = tk.Text(outer, wrap="word")
        self.log_widget.pack(fill="both", expand=True)

        root.after(150, self.poll_events)

    def browse_config(self) -> None:
        selected = self._filedialog.askopenfilename(
            title="Select PhoPS config",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            self.config_var.set(selected)

    def start_pipeline(self) -> None:
        if self.running:
            return
        self.running = True
        self.run_button.config(state="disabled")
        self.status_var.set("Running")
        self.log_widget.delete("1.0", "end")
        config_path = Path(self.config_var.get()).expanduser()
        self.worker = threading.Thread(target=self._run_worker, args=(config_path,), daemon=True)
        self.worker.start()

    def _run_worker(self, config_path: Path) -> None:
        try:
            summary = run_pipeline_from_file(config_path, reporter=QueueReporter(self.event_queue))
            self.event_queue.put(
                ProgressEvent(
                    level="info",
                    message=(
                        f"Done. total={summary.total_files} solved={summary.solved_files} "
                        f"measured={summary.measured_files} skipped={summary.skipped_files}"
                    ),
                    stage="gui",
                )
            )
        except Exception as exc:
            self.event_queue.put(ProgressEvent(level="error", message=str(exc), stage="gui"))
        finally:
            self.event_queue.put(ProgressEvent(level="info", message="__COMPLETE__", stage="gui"))

    def poll_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except Empty:
                break
            if event.message == "__COMPLETE__":
                self.running = False
                self.run_button.config(state="normal")
                self.status_var.set("Idle")
                continue
            self.log_widget.insert("end", f"[{event.level.upper()}] {event.message}\n")
            self.log_widget.see("end")
        self.root.after(150, self.poll_events)


def launch_gui(config_path: str | Path | None = None) -> int:
    """Launch the current reference desktop application."""

    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python installation.") from exc

    root = tk.Tk()
    GuiApplication(root, initial_config=Path(config_path) if config_path else None)
    root.mainloop()
    return 0
