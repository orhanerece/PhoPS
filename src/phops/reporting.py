"""Reporting helpers shared by CLI and GUI."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
import logging
from queue import Queue
import re
import shutil
import sys
from typing import Any, Protocol
import warnings


@dataclass
class ProgressEvent:
    """Structured progress event."""

    level: str
    message: str
    stage: str = "general"
    payload: dict[str, Any] = field(default_factory=dict)


class ProgressReporter(Protocol):
    """Interface for progress sinks."""

    def emit(self, event: ProgressEvent) -> None:
        """Handle a progress event."""


class NullReporter:
    """Reporter that discards all events."""

    def emit(self, event: ProgressEvent) -> None:
        del event


class LoggingReporter:
    """Reporter backed by the standard logging package."""

    LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("phops")

    def emit(self, event: ProgressEvent) -> None:
        level = self.LEVEL_MAP.get(event.level.lower(), logging.INFO)
        self.logger.log(level, event.message)


FRAME_STAGES = ("astrometry", "photometry", "target", "plot")
FRAME_STAGE_TOTAL = len(FRAME_STAGES)
FRAME_STAGE_INDEX = {stage: index for index, stage in enumerate(FRAME_STAGES, start=1)}
STAGE_META = {
    "general": {"label": "general", "short": "INFO", "simple_badge": "·", "rich_badge": "·", "simple_color": "37", "rich_style": "white"},
    "pipeline": {"label": "pipeline", "short": "RUN", "simple_badge": "◆", "rich_badge": "🛰", "simple_color": "36;1", "rich_style": "bold cyan"},
    "astrometry": {"label": "astrometry", "short": "ASTRO", "simple_badge": "◎", "rich_badge": "🧭", "simple_color": "34;1", "rich_style": "bold blue"},
    "photometry": {"label": "photometry", "short": "PHOTO", "simple_badge": "◉", "rich_badge": "📷", "simple_color": "32;1", "rich_style": "bold green"},
    "target": {"label": "target", "short": "TARGET", "simple_badge": "⌖", "rich_badge": "🎯", "simple_color": "35;1", "rich_style": "bold magenta"},
    "plot": {"label": "plot", "short": "PLOT", "simple_badge": "▣", "rich_badge": "📈", "simple_color": "37;1", "rich_style": "bold white"},
    "gui": {"label": "gui", "short": "GUI", "simple_badge": "□", "rich_badge": "□", "simple_color": "36", "rich_style": "cyan"},
}
LEVEL_META = {
    "debug": {"badge": "…", "simple_color": "2", "rich_style": "dim"},
    "info": {"badge": "ℹ", "simple_color": "36", "rich_style": "cyan"},
    "warning": {"badge": "⚠", "simple_color": "33;1", "rich_style": "yellow"},
    "error": {"badge": "✖", "simple_color": "31;1", "rich_style": "bold red"},
    "success": {"badge": "✅", "simple_color": "32;1", "rich_style": "bold green"},
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def normalize_stage(stage: str) -> str:
    """Map unknown stages to the generic presentation metadata."""

    return stage if stage in STAGE_META else "general"


def level_key_for_event(event: ProgressEvent) -> str:
    """Return the display level, promoting successful completion to a dedicated status."""

    event_type = str(event.payload.get("event", ""))
    if event_type == "run_complete" and event.level.lower() == "info":
        return "success"
    level = event.level.lower()
    return level if level in LEVEL_META else "info"


def frame_phase_state(stage: str, event_type: str, frame_status: str | None = None) -> tuple[int, str]:
    """Return the current-frame phase index and label for an event."""

    if event_type == "frame_start":
        return 0, "queued"
    if event_type in {"frame_end", "run_complete"}:
        return FRAME_STAGE_TOTAL, frame_status or "complete"
    stage_key = normalize_stage(stage)
    if stage_key in FRAME_STAGE_INDEX:
        return FRAME_STAGE_INDEX[stage_key], STAGE_META[stage_key]["label"]
    return 0, "queued"


@dataclass
class TerminalProgressState:
    """Shared progress state used by terminal renderers."""

    total_files: int = 0
    current_index: int = 0
    overall_completed: int = 0
    current_file: str = "-"
    measured_files: int = 0
    skipped_files: int = 0
    solved_files: int = 0
    current_stage: str = "pipeline"
    current_action: str = "waiting"
    last_warning: str = "-"
    frame_phase_completed: int = 0
    frame_phase_label: str = "queued"

    def apply(self, event: ProgressEvent) -> str:
        """Merge an event into the current rendering state."""

        event_type = str(event.payload.get("event", ""))
        total_files = event.payload.get("total_files")
        file_name = event.payload.get("file_name")
        current_index = event.payload.get("current_index")
        frame_status = event.payload.get("frame_status")

        if isinstance(total_files, int) and total_files > 0:
            self.total_files = total_files
        if isinstance(file_name, str) and file_name:
            self.current_file = file_name
        if isinstance(current_index, int) and current_index > 0:
            self.current_index = current_index

        stage_key = normalize_stage(event.stage)
        if stage_key != "general" or event_type in {"run_summary", "run_start", "frame_start", "frame_end", "run_complete"}:
            self.current_stage = stage_key
        if event.level.lower() not in {"warning", "error"}:
            self.current_action = event.message

        if event_type == "run_start":
            self.overall_completed = 0
            self.frame_phase_completed = 0
            self.frame_phase_label = "queued"
            self.last_warning = "-"
            return event_type

        if event_type == "frame_start":
            self.overall_completed = max(self.current_index - 1, 0)
            self.frame_phase_completed = 0
            self.frame_phase_label = "queued"
            return event_type

        if event.level.lower() in {"warning", "error"}:
            self.last_warning = event.message

        phase_completed, phase_label = frame_phase_state(self.current_stage, event_type, frame_status)
        if event_type == "frame_end":
            if frame_status == "measured":
                self.measured_files += 1
            elif frame_status == "skipped":
                self.skipped_files += 1
            if bool(event.payload.get("solved")):
                self.solved_files += 1
            self.overall_completed = self.current_index
            self.frame_phase_completed = phase_completed
            self.frame_phase_label = phase_label
            return event_type

        if event_type == "run_complete":
            self.measured_files = int(event.payload.get("measured_files", self.measured_files))
            self.skipped_files = int(event.payload.get("skipped_files", self.skipped_files))
            self.solved_files = int(event.payload.get("solved_files", self.solved_files))
            self.overall_completed = self.total_files or self.current_index
            self.current_index = self.total_files or self.current_index
            self.frame_phase_completed = phase_completed
            self.frame_phase_label = phase_label
            return event_type

        if phase_completed > 0:
            self.frame_phase_completed = max(self.frame_phase_completed, phase_completed)
            self.frame_phase_label = phase_label
        return event_type


@contextmanager
def capture_python_warnings(reporter: ProgressReporter | None):
    """Redirect Python warnings into the structured progress channel."""

    seen_messages: set[str] = set()
    original_showwarning = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None) -> None:
        del filename, lineno, file, line
        warning_text = f"{category.__name__}: {message}"
        if warning_text in seen_messages:
            return
        seen_messages.add(warning_text)
        report(reporter, "warning", warning_text, stage="general", event="python_warning")

    warnings.showwarning = _showwarning
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            yield
    finally:
        warnings.showwarning = original_showwarning


class SimpleTerminalProgressReporter:
    """Reporter that renders a lightweight two-line live progress view."""

    def __init__(self) -> None:
        self.stream = sys.stderr
        self.state = TerminalProgressState()
        self._live_active = False

    def close(self) -> None:
        if self._live_active:
            self.stream.write("\n")
            self.stream.flush()
            self._live_active = False

    def _color(self, text: str, color_code: str) -> str:
        if not self.stream.isatty():
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    def _terminal_width(self) -> int:
        return max(shutil.get_terminal_size(fallback=(120, 24)).columns, 72)

    def _visible_length(self, text: str) -> int:
        return len(ANSI_RE.sub("", text))

    def _fit_width(self, text: str, width: int) -> str:
        visible_length = self._visible_length(text)
        if visible_length >= width:
            return text
        return text + (" " * (width - visible_length))

    def _stage_badge(self, stage: str) -> str:
        meta = STAGE_META[normalize_stage(stage)]
        return self._color(f"{meta['simple_badge']} {meta['short']:<6}", meta["simple_color"])

    def _level_badge(self, event: ProgressEvent) -> str:
        meta = LEVEL_META[level_key_for_event(event)]
        return self._color(meta["badge"], meta["simple_color"])

    def _counter_text(self) -> str:
        count_width = max(len(str(self.state.total_files or 0)), 1)
        done_label = self._color("done", LEVEL_META["info"]["simple_color"])
        ok_label = self._color("ok", LEVEL_META["success"]["simple_color"])
        skip_label = self._color("skip", LEVEL_META["warning"]["simple_color"])
        left_label = self._color("left", STAGE_META["pipeline"]["simple_color"])
        done = self.state.overall_completed
        remaining = max((self.state.total_files or 0) - self.state.overall_completed, 0)
        return (
            f"{done_label} {done:>{count_width}} "
            f"{ok_label} {self.state.measured_files:>{count_width}} "
            f"{skip_label} {self.state.skipped_files:>{count_width}} "
            f"{left_label} {remaining:>{count_width}}"
        )

    def _phase_bar(self, width: int) -> str:
        completed = min(self.state.frame_phase_completed, FRAME_STAGE_TOTAL)
        fraction = completed / FRAME_STAGE_TOTAL if FRAME_STAGE_TOTAL else 0.0
        filled = int(round(fraction * width))
        return f"{'=' * filled}{'-' * (width - filled)}"

    def _render_lines(self) -> tuple[str, str]:
        terminal_width = self._terminal_width()
        bar_width = min(max(terminal_width // 10, 10), 18)
        phase_bar_width = 16
        count_width = max(len(str(self.state.total_files or 0)), 1)
        total = self.state.total_files or 1
        completed = min(self.state.overall_completed, total)
        fraction = completed / total
        filled = int(round(fraction * bar_width))
        bar = f"{'#' * filled}{'-' * (bar_width - filled)}"
        percent = f"{fraction * 100:5.1f}%"
        stage_label = self._stage_badge(self.state.current_stage)
        counters = self._counter_text()
        prefix_one = f"{stage_label} [{bar}] {percent} {completed:>{count_width}}/{total:<{count_width}} "
        middle_one = f" | {counters} | ⚠ "
        fixed_one = self._visible_length(prefix_one) + self._visible_length(middle_one)
        available_one = max(terminal_width - fixed_one, 24)
        file_width = max(12, min(26, int(available_one * 0.48)))
        warning_width = max(10, available_one - file_width)
        file_label = f"{self._truncate(self.state.current_file, file_width):<{file_width}}"
        warning_label = f"{self._truncate(self.state.last_warning, warning_width):<{warning_width}}"
        line_one = prefix_one + file_label + middle_one + warning_label
        phase_label = self._truncate(self.state.frame_phase_label, 18)
        prefix_two = (
            f"  ↳ frame [{self._phase_bar(phase_bar_width)}] {self.state.frame_phase_completed}/{FRAME_STAGE_TOTAL} "
            f"{phase_label:<8} | action: "
        )
        available_two = max(terminal_width - self._visible_length(prefix_two), 12)
        current_action = f"{self._truncate(self.state.current_action, available_two):<{available_two}}"
        line_two = prefix_two + current_action
        width = terminal_width
        return self._fit_width(line_one, width), self._fit_width(line_two, width)

    def _render_live(self) -> None:
        line_one, line_two = self._render_lines()
        if self._live_active and self.stream.isatty():
            self.stream.write(f"\r\033[1A\033[2K{line_one}\r\n\033[2K{line_two}")
        else:
            self.stream.write(f"{line_one}\n{line_two}")
            self._live_active = True
        self.stream.flush()

    def _suspend_live(self) -> None:
        if not self._live_active:
            return
        if self.stream.isatty():
            self.stream.write("\r\033[1A\033[2K\r\n\033[2K")
        else:
            self.stream.write("\n")
        self.stream.flush()
        self._live_active = False

    def _print_event(self, event: ProgressEvent) -> None:
        self._suspend_live()
        self.stream.write(f"{self._level_badge(event)} {event.message}\n")
        self.stream.flush()

    def _print_summary(self, event: ProgressEvent) -> None:
        self._suspend_live()
        summary_items = event.payload.get("summary_items")
        title = self._color("Run summary", "1;36")
        if isinstance(summary_items, list) and summary_items:
            label_width = max(len(str(label)) for label, _ in summary_items)
            self.stream.write(f"{title}\n")
            for label, value in summary_items:
                self.stream.write(f"  {str(label).ljust(label_width)} : {value}\n")
            self.stream.flush()
            return
        self.stream.write(f"{title}\n{event.message}\n")
        self.stream.flush()

    def emit(self, event: ProgressEvent) -> None:
        event_type = self.state.apply(event)

        if event_type == "run_summary":
            self._print_summary(event)
            return

        if event.level == "error":
            self._print_event(event)
            self._render_live()
        elif event_type == "run_complete":
            self._print_event(event)
        else:
            self._render_live()


class RichProgressReporter:
    """Reporter that renders a two-layer live terminal progress view."""

    def __init__(self) -> None:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self.console = Console(stderr=True)
        self.state = TerminalProgressState()
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("{task.fields[stage_label]}", justify="left"),
            BarColumn(bar_width=24),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[file_label]}", justify="left"),
            TextColumn("[dim]{task.fields[detail_label]}[/dim]", justify="left"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        self.progress.start()
        self.overall_task_id = self.progress.add_task(
            "overall",
            total=1,
            completed=0,
            stage_label=self._stage_badge("pipeline"),
            file_label="0/? idle",
            detail_label="done 0 ok 0 skip 0 left 0 | ⚠ -",
        )
        self.frame_task_id = self.progress.add_task(
            "frame",
            total=FRAME_STAGE_TOTAL,
            completed=0,
            stage_label=self._stage_badge("pipeline"),
            file_label=f"0/{FRAME_STAGE_TOTAL} queued",
            detail_label="action: waiting",
        )

    def close(self) -> None:
        self.progress.stop()

    def _truncate(self, text: str, limit: int = 64) -> str:
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    def _stage_badge(self, stage: str) -> str:
        meta = STAGE_META[normalize_stage(stage)]
        return f"[{meta['rich_style']}]{meta['rich_badge']} {meta['short']:<6}[/{meta['rich_style']}]"

    def _level_badge(self, event: ProgressEvent) -> tuple[str, str]:
        meta = LEVEL_META[level_key_for_event(event)]
        return meta["badge"], meta["rich_style"]

    def _refresh_overall(self) -> None:
        count_width = max(len(str(self.state.total_files or 0)), 1)
        warning_label = self._truncate(self.state.last_warning, 28)
        done = self.state.overall_completed
        remaining = max((self.state.total_files or 0) - self.state.overall_completed, 0)
        detail_label = (
            f"done {done:>{count_width}} "
            f"ok {self.state.measured_files:>{count_width}} "
            f"skip {self.state.skipped_files:>{count_width}} "
            f"left {remaining:>{count_width}} | ⚠ {warning_label}"
        )
        self.progress.update(
            self.overall_task_id,
            total=self.state.total_files or 1,
            completed=min(self.state.overall_completed, self.state.total_files or 1),
            stage_label=self._stage_badge(self.state.current_stage),
            file_label=(
                f"{self.state.current_index:>{count_width}}/"
                f"{self.state.total_files or '?':<{count_width}} {self._truncate(self.state.current_file, 24)}"
            ),
            detail_label=detail_label,
        )

    def _refresh_frame(self) -> None:
        self.progress.update(
            self.frame_task_id,
            total=FRAME_STAGE_TOTAL,
            completed=min(self.state.frame_phase_completed, FRAME_STAGE_TOTAL),
            stage_label=self._stage_badge(self.state.current_stage),
            file_label=f"{self.state.frame_phase_completed}/{FRAME_STAGE_TOTAL} {self._truncate(self.state.frame_phase_label, 18)}",
            detail_label=f"action: {self._truncate(self.state.current_action, 60)}",
        )

    def _print_event(self, event: ProgressEvent) -> None:
        badge, style = self._level_badge(event)
        self.progress.console.print(f"[{style}]{badge}[/{style}] {event.message}")

    def _print_summary(self, event: ProgressEvent) -> None:
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        summary_items = event.payload.get("summary_items")
        if not isinstance(summary_items, list) or not summary_items:
            self.progress.console.print(Panel.fit(event.message, title="Run Summary", border_style="cyan"))
            return

        label_width = max(len(str(label)) for label, _ in summary_items)
        lines: list[Text] = []
        for label, value in summary_items:
            line = Text()
            line.append(str(label).ljust(label_width), style="bold cyan")
            line.append(" : ", style="dim")
            line.append(str(value))
            lines.append(line)

        self.progress.console.print(Panel(Group(*lines), title="Run Summary", border_style="cyan"))

    def emit(self, event: ProgressEvent) -> None:
        event_type = self.state.apply(event)

        if event_type == "run_summary":
            self._print_summary(event)
            return

        self._refresh_overall()
        self._refresh_frame()

        if event_type == "run_complete":
            self._print_event(event)
            return

        if event.level == "error":
            self._print_event(event)


class QueueReporter:
    """Reporter used by the GUI to receive background progress."""

    def __init__(self, queue: Queue[ProgressEvent]) -> None:
        self.queue = queue

    def emit(self, event: ProgressEvent) -> None:
        self.queue.put(event)


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the project logger."""

    logger = logging.getLogger("phops")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)
    return logger


def create_terminal_reporter(verbose: bool = False) -> ProgressReporter:
    """Create the most suitable terminal reporter for the current environment."""

    if verbose or not sys.stderr.isatty():
        return LoggingReporter(configure_logging(verbose=verbose))
    if importlib.util.find_spec("rich") is None:
        return SimpleTerminalProgressReporter()
    return RichProgressReporter()


def report(
    reporter: ProgressReporter | None,
    level: str,
    message: str,
    stage: str = "general",
    **payload: Any,
) -> None:
    """Emit a progress event if a reporter is available."""

    active_reporter = reporter or NullReporter()
    active_reporter.emit(ProgressEvent(level=level, message=message, stage=stage, payload=payload))
