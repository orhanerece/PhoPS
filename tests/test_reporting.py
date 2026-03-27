import warnings

from phops.reporting import (
    ProgressEvent,
    TerminalProgressState,
    capture_python_warnings,
    frame_phase_state,
    level_key_for_event,
)


class CollectingReporter:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def test_frame_phase_state_maps_stage_flow() -> None:
    assert frame_phase_state("pipeline", "frame_start") == (0, "queued")
    assert frame_phase_state("astrometry", "") == (1, "astrometry")
    assert frame_phase_state("photometry", "") == (2, "photometry")
    assert frame_phase_state("target", "") == (3, "target")
    assert frame_phase_state("plot", "") == (4, "plot")
    assert frame_phase_state("pipeline", "frame_end", "measured") == (4, "measured")


def test_level_key_for_event_marks_successful_completion() -> None:
    event = ProgressEvent(level="info", message="done", stage="pipeline", payload={"event": "run_complete"})
    assert level_key_for_event(event) == "success"


def test_terminal_progress_state_tracks_warning_and_frame_counts() -> None:
    state = TerminalProgressState()

    state.apply(ProgressEvent(level="info", message="Processing 2 FITS files.", stage="pipeline", payload={"event": "run_start", "total_files": 2}))
    state.apply(
        ProgressEvent(
            level="info",
            message="Starting frame frame01.fits",
            stage="pipeline",
            payload={"event": "frame_start", "total_files": 2, "current_index": 1, "file_name": "frame01.fits"},
        )
    )
    state.apply(ProgressEvent(level="info", message="Solving astrometry for frame01.fits", stage="astrometry"))
    state.apply(
        ProgressEvent(
            level="warning",
            message="FITSFixedWarning: changed DATE-OBS",
            stage="general",
            payload={"event": "python_warning"},
        )
    )
    state.apply(
        ProgressEvent(
            level="warning",
            message="frame01.fits: not enough matched stars for calibration.",
            stage="pipeline",
            payload={"event": "frame_end", "current_index": 1, "file_name": "frame01.fits", "frame_status": "skipped", "solved": True},
        )
    )

    assert state.total_files == 2
    assert state.overall_completed == 1
    assert state.current_file == "frame01.fits"
    assert state.current_action == "Solving astrometry for frame01.fits"
    assert state.last_warning == "frame01.fits: not enough matched stars for calibration."
    assert state.current_stage == "pipeline"
    assert state.frame_phase_completed == 4
    assert state.frame_phase_label == "skipped"
    assert state.skipped_files == 1
    assert state.solved_files == 1


def test_capture_python_warnings_deduplicates_messages() -> None:
    reporter = CollectingReporter()

    with capture_python_warnings(reporter):
        warnings.warn("repeated warning", UserWarning)
        warnings.warn("repeated warning", UserWarning)

    assert len(reporter.events) == 1
    assert reporter.events[0].level == "warning"
    assert reporter.events[0].message == "UserWarning: repeated warning"


def test_python_warning_does_not_override_current_stage_or_action() -> None:
    state = TerminalProgressState(total_files=10, current_stage="astrometry", current_action="Solving astrometry")

    state.apply(ProgressEvent(level="warning", message="FITSFixedWarning: header fix", stage="general", payload={"event": "python_warning"}))

    assert state.current_stage == "astrometry"
    assert state.current_action == "Solving astrometry"
    assert state.last_warning == "FITSFixedWarning: header fix"
