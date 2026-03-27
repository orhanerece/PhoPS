"""Command line interface for PhoPS."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from .config import load_config
from .errors import PhopsError
from .reporting import (
    LoggingReporter,
    NullReporter,
    capture_python_warnings,
    configure_logging,
    create_terminal_reporter,
)


def _write_example_config(destination: Path) -> None:
    template = resources.files("phops").joinpath("templates/example_config.yaml").read_text(encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template, encoding="utf-8")


def _prompt_for_existing_run_action(existing_state) -> str:
    artifact_lines = []
    for path in existing_state.artifact_paths[:4]:
        if path.is_dir():
            try:
                count = sum(1 for _ in path.iterdir())
            except OSError:
                count = 0
            artifact_lines.append(f"  - {path} ({count} files)")
        else:
            artifact_lines.append(f"  - {path}")
    if len(existing_state.artifact_paths) > 4:
        artifact_lines.append(f"  - ... and {len(existing_state.artifact_paths) - 4} more")

    print("Existing PhoPS run artifacts were found.")
    if artifact_lines:
        print("\n".join(artifact_lines))
    print(
        "Resume will keep existing outputs and skip frames already present in the checkpoint/photometry table "
        f"({len(existing_state.completed_frames)} completed, {existing_state.pending_files} pending)."
    )

    default_choice = "resume" if existing_state.pending_files > 0 else "restart"
    prompt = f"Choose [resume/restart/cancel] (default: {default_choice}): "
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"", default_choice}:
            return default_choice
        if answer in {"resume", "restart", "cancel"}:
            return answer
        print("Please answer with 'resume', 'restart', or 'cancel'.")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(prog="phops", description="Photometry and astrometry pipeline for point sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline.")
    run_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    run_mode_group = run_parser.add_mutually_exclusive_group()
    run_mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs and skip frames already measured.",
    )
    run_mode_group.add_argument(
        "--restart",
        action="store_true",
        help="Discard previous run outputs and start from scratch.",
    )
    run_mode_group.add_argument("--no-overwrite", action="store_true", help=argparse.SUPPRESS)
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    plot_parser = subparsers.add_parser("plot-photometry", help="Render the photometry light curve from an existing CSV.")
    plot_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    plot_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    validate_parser = subparsers.add_parser("validate-config", help="Validate the YAML configuration.")
    validate_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )

    init_parser = subparsers.add_parser("init-config", help="Write an example configuration file.")
    init_parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=Path("config.yaml"),
        help="Where to write the example configuration.",
    )

    gui_parser = subparsers.add_parser("gui", help="Launch the current reference desktop runner.")
    gui_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )

    return parser


def _run_command(args: argparse.Namespace) -> int:
    from .pipeline import PipelineRunner, inspect_existing_run

    config = load_config(args.config)
    existing_state = inspect_existing_run(config)
    resume_requested = bool(args.resume or args.no_overwrite)
    restart_requested = bool(args.restart)

    if resume_requested:
        overwrite = False
        resume = True
    elif restart_requested:
        overwrite = True
        resume = False
    elif existing_state.has_artifacts and sys.stdin.isatty():
        choice = _prompt_for_existing_run_action(existing_state)
        if choice == "cancel":
            raise PhopsError("Run cancelled by user.")
        overwrite = choice == "restart"
        resume = choice == "resume"
    else:
        overwrite = True
        resume = False

    reporter = create_terminal_reporter(verbose=bool(args.verbose))
    try:
        runner = PipelineRunner(config=config, reporter=reporter)
        with capture_python_warnings(reporter):
            summary = runner.run(overwrite=overwrite, resume=resume)
    finally:
        close_method = getattr(reporter, "close", None)
        if callable(close_method):
            close_method()

    if isinstance(reporter, LoggingReporter):
        reporter.logger.info(
            "Completed: total=%s solved=%s measured=%s skipped=%s",
            summary.total_files,
            summary.solved_files,
            summary.measured_files,
            summary.skipped_files,
        )
        reporter.logger.info("Photometry CSV: %s", summary.photometry_csv)
        reporter.logger.info("Astrometry CSV: %s", summary.astrometry_csv)
    else:
        print(f"Photometry CSV: {summary.photometry_csv}")
        print(f"Astrometry CSV: {summary.astrometry_csv}")
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    load_config(args.config)
    print(f"Configuration is valid: {args.config}")
    return 0


def _plot_photometry(args: argparse.Namespace) -> int:
    from .plotting import plot_photometry_light_curve

    config = load_config(args.config)
    photometry_csv = config.paths.photometry_csv_path
    if not photometry_csv.exists():
        raise PhopsError(f"Photometry CSV not found: {photometry_csv}")

    reporter = LoggingReporter(configure_logging(verbose=bool(args.verbose))) if args.verbose else NullReporter()
    output_pdf = config.paths.plot_dir / "light_curve.pdf" if config.plots.light_curve_pdf else None
    result = plot_photometry_light_curve(
        photometry_csv,
        output_png=config.paths.plot_dir / "light_curve.png",
        output_pdf=output_pdf,
        config=config,
        reporter=reporter,
    )

    if result is None:
        raise PhopsError(f"Could not generate the light-curve plot from {photometry_csv}.")

    print(f"Light-curve PNG: {result.png_path}")
    if result.pdf_path is not None:
        print(f"Light-curve PDF: {result.pdf_path}")
    return 0


def _init_config(args: argparse.Namespace) -> int:
    destination = Path(args.destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    _write_example_config(destination)
    print(f"Example configuration written to {destination}")
    return 0


def _launch_gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    return launch_gui(args.config)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "validate-config":
            return _validate_config(args)
        if args.command == "plot-photometry":
            return _plot_photometry(args)
        if args.command == "init-config":
            return _init_config(args)
        if args.command == "gui":
            return _launch_gui(args)
    except (PhopsError, FileExistsError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    return 0
