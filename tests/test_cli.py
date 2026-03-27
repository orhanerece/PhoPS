from pathlib import Path

from phops.cli import _prompt_for_existing_run_action
from phops.pipeline import ExistingRunState


def test_prompt_for_existing_run_action_defaults_to_resume(monkeypatch, capsys) -> None:
    state = ExistingRunState(
        total_input_files=10,
        completed_frames={"frame01.fits", "frame02.fits"},
        artifact_paths=[Path("/tmp/photometry.csv")],
    )
    monkeypatch.setattr("builtins.input", lambda _: "")

    choice = _prompt_for_existing_run_action(state)

    captured = capsys.readouterr()
    assert choice == "resume"
    assert "2 completed, 8 pending" in captured.out


def test_prompt_for_existing_run_action_accepts_restart(monkeypatch) -> None:
    state = ExistingRunState(
        total_input_files=4,
        completed_frames=set(),
        artifact_paths=[Path("/tmp/astrometry.csv")],
    )
    monkeypatch.setattr("builtins.input", lambda _: "restart")

    assert _prompt_for_existing_run_action(state) == "restart"
