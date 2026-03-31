"""Pipeline orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .astrometry import AstrometrySolver
from .config import AppConfig, load_config
from .errors import PhopsError, PipelineError, TargetResolutionError
from .photometry import Photometry
from .plotting import plot_astrometry_residuals, plot_photometry_light_curve
from .reporting import NullReporter, ProgressReporter, capture_python_warnings, report
from .target import TargetManager
from .utils import append_rows_to_csv, calculate_residuals, describe_input_selector, iter_input_files, load_fits_image


@dataclass
class PipelineSummary:
    """High level pipeline execution summary."""

    total_files: int
    solved_files: int
    measured_files: int
    skipped_files: int
    photometry_csv: Path
    astrometry_csv: Path
    generated_plots: list[Path] = field(default_factory=list)


@dataclass
class ExistingRunState:
    """Existing outputs that can be used for a resumed run."""

    total_input_files: int
    completed_frames: set[str] = field(default_factory=set)
    artifact_paths: list[Path] = field(default_factory=list)

    @property
    def pending_files(self) -> int:
        return max(self.total_input_files - len(self.completed_frames), 0)

    @property
    def has_artifacts(self) -> bool:
        return bool(self.artifact_paths)

    @property
    def can_resume(self) -> bool:
        return bool(self.completed_frames)


def _load_checkpoint_frames(state_path: Path, input_dir: Path) -> set[str]:
    if not state_path.exists():
        return set()

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    if not isinstance(payload, dict):
        return set()

    stored_input_dir = payload.get("input_dir")
    if isinstance(stored_input_dir, str) and stored_input_dir and stored_input_dir != str(input_dir.resolve()):
        return set()

    raw_frames = payload.get("completed_frames", [])
    if not isinstance(raw_frames, list):
        return set()
    return {str(item) for item in raw_frames if isinstance(item, str) and item}


def _read_completed_frames_from_photometry(path: Path) -> set[str]:
    if not path.exists():
        return set()

    try:
        frame = pd.read_csv(path)
    except Exception:
        return set()

    if frame.empty or "filename" not in frame.columns:
        return set()
    return {str(value) for value in frame["filename"].dropna().astype(str)}


def _deduplicate_photometry_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()

    try:
        frame = pd.read_csv(path)
    except Exception:
        return set()

    if frame.empty or "filename" not in frame.columns:
        return set()

    frame["filename"] = frame["filename"].astype(str)
    deduplicated = frame.drop_duplicates(subset="filename", keep="last")
    if len(deduplicated) != len(frame):
        deduplicated.to_csv(path, index=False)
    return {str(value) for value in deduplicated["filename"].dropna()}


def _prune_astrometry_csv(path: Path, completed_frames: set[str]) -> None:
    if not path.exists():
        return
    if not completed_frames:
        path.unlink(missing_ok=True)
        return

    try:
        frame = pd.read_csv(path)
    except Exception:
        path.unlink(missing_ok=True)
        return

    if frame.empty or "filename" not in frame.columns:
        path.unlink(missing_ok=True)
        return

    filtered = frame[frame["filename"].astype(str).isin(completed_frames)].copy()
    if filtered.empty:
        path.unlink(missing_ok=True)
        return
    filtered.to_csv(path, index=False)


def _write_run_state(state_path: Path, input_dir: Path, completed_frames: set[str]) -> None:
    payload = {
        "version": 1,
        "input_dir": str(input_dir.resolve()),
        "completed_frames": sorted(completed_frames),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def inspect_existing_run(config: AppConfig) -> ExistingRunState:
    """Inspect known output artifacts and resumable frame state."""

    input_files = iter_input_files(config.paths.input_dir, config.paths.file_extension)
    input_names = {path.name for path in input_files}
    completed_frames = _load_checkpoint_frames(
        config.paths.run_state_path, config.paths.input_dir
    ) | _read_completed_frames_from_photometry(config.paths.photometry_csv_path)
    completed_frames &= input_names

    artifact_paths: list[Path] = []
    for candidate in (
        config.paths.photometry_csv_path,
        config.paths.astrometry_csv_path,
        config.paths.run_state_path,
    ):
        if candidate.exists():
            artifact_paths.append(candidate)

    for candidate in (config.paths.plot_dir, config.paths.cutout_dir):
        if candidate is not None and candidate.exists() and any(candidate.iterdir()):
            artifact_paths.append(candidate)

    return ExistingRunState(
        total_input_files=len(input_files),
        completed_frames=completed_frames,
        artifact_paths=artifact_paths,
    )


def build_run_summary_items(
    config: AppConfig,
    total_files: int,
    overwrite: bool,
    existing_state: ExistingRunState | None = None,
) -> list[tuple[str, str]]:
    """Build a compact pre-run summary for terminal and GUI reporters."""

    if config.photometry.mode == "asteroid":
        target_value = f"{config.photometry.target_id} | filter={config.photometry.filter}"
    else:
        coords = config.photometry.coords or (0.0, 0.0)
        target_value = f"RA={coords[0]:.6f} deg Dec={coords[1]:.6f} deg | filter={config.photometry.filter}"

    enabled_plots: list[str] = []
    if config.plots.plot_astrometry:
        enabled_plots.append("astrometry")
    if config.plots.plot_image:
        enabled_plots.append("image")
    if config.plots.plot_light_curve:
        enabled_plots.append("light-curve")
    plot_modes = ", ".join(enabled_plots) if enabled_plots else "disabled"

    total_input_files = existing_state.total_input_files if existing_state else total_files
    input_value = f"{config.paths.input_dir} ({total_input_files} {describe_input_selector(config.paths.file_extension)} files)"
    if existing_state is not None and not overwrite:
        input_value += f" | {len(existing_state.completed_frames)} completed | {total_files} pending"

    items = [
        ("Config", str(config.source_path)),
        ("Input", input_value),
        ("Mode", config.photometry.mode),
        ("Target", target_value),
        ("Run mode", "restart" if overwrite else "resume"),
        ("Astrometry", config.astrometry.solve_mode),
        ("Photometry CSV", str(config.paths.photometry_csv_path)),
        ("Astrometry CSV", str(config.paths.astrometry_csv_path)),
        ("Plots", f"{config.paths.plot_dir} ({plot_modes})"),
        ("Cutouts", str(config.paths.cutout_dir)),
        ("Overwrite", "yes" if overwrite else "no"),
    ]
    if existing_state is not None and not overwrite:
        items.insert(
            5,
            (
                "Resume state",
                f"{len(existing_state.completed_frames)} completed | {total_files} pending",
            ),
        )
    return items


def format_run_summary(items: Iterable[tuple[str, str]]) -> str:
    """Render a human-friendly pre-run summary block."""

    item_list = list(items)
    if not item_list:
        return "Run summary"
    label_width = max(len(label) for label, _ in item_list)
    body = "\n".join(f"  {label.ljust(label_width)} : {value}" for label, value in item_list)
    return f"Run summary\n{body}"


class PipelineRunner:
    """Run the PhoPS processing pipeline."""

    def __init__(
        self,
        config: AppConfig,
        reporter: ProgressReporter | None = None,
        astrometry_solver: AstrometrySolver | None = None,
        photometry: Photometry | None = None,
        target_manager: TargetManager | None = None,
    ) -> None:
        self.config = config
        self.reporter = reporter or NullReporter()
        self.astrometry_solver = astrometry_solver or AstrometrySolver(config, reporter=self.reporter)
        self.photometry = photometry or Photometry(config, reporter=self.reporter)
        self.target_manager = target_manager or TargetManager(config, reporter=self.reporter)

    def run(self, overwrite: bool = True, resume: bool = False) -> PipelineSummary:
        """Run the pipeline and return a summary."""

        self.config.ensure_runtime_dirs()
        input_files = iter_input_files(self.config.paths.input_dir, self.config.paths.file_extension)
        if not input_files:
            raise PipelineError(
                f"No input files matching '{describe_input_selector(self.config.paths.file_extension)}' were found in "
                f"{self.config.paths.input_dir}."
            )

        photometry_csv = self.config.paths.photometry_csv_path
        astrometry_csv = self.config.paths.astrometry_csv_path
        run_state_path = self.config.paths.run_state_path
        existing_state = inspect_existing_run(self.config) if resume else None
        completed_frames: set[str] = set()
        if overwrite:
            photometry_csv.unlink(missing_ok=True)
            astrometry_csv.unlink(missing_ok=True)
            run_state_path.unlink(missing_ok=True)
        elif resume:
            completed_frames = _deduplicate_photometry_csv(photometry_csv)
            completed_frames |= _load_checkpoint_frames(run_state_path, self.config.paths.input_dir) & {path.name for path in input_files}
            _prune_astrometry_csv(astrometry_csv, completed_frames)
            _write_run_state(run_state_path, self.config.paths.input_dir, completed_frames)

        files_to_process = [path for path in input_files if path.name not in completed_frames] if resume else input_files

        solved_files = 0
        measured_files = 0
        skipped_files = 0
        generated_plots: list[Path] = []
        summary_items = build_run_summary_items(
            self.config,
            len(files_to_process),
            overwrite=overwrite,
            existing_state=existing_state,
        )
        report(
            self.reporter,
            "info",
            format_run_summary(summary_items),
            stage="pipeline",
            event="run_summary",
            total_files=len(files_to_process),
            summary_items=summary_items,
        )
        if files_to_process:
            report(
                self.reporter,
                "info",
                f"Processing {len(files_to_process)} FITS files.",
                stage="pipeline",
                event="run_start",
                total_files=len(files_to_process),
            )
        else:
            report(
                self.reporter,
                "info",
                "No pending frames remain; refreshing summary outputs from the existing CSV files.",
                stage="pipeline",
            )

        for current_index, image_path in enumerate(files_to_process, start=1):
            report(
                self.reporter,
                "info",
                f"Starting frame {image_path.name}",
                stage="pipeline",
                event="frame_start",
                total_files=len(files_to_process),
                current_index=current_index,
                file_name=image_path.name,
            )
            solved_this_frame = False
            try:
                solved_fits = self.astrometry_solver.solve(image_path)
                solved_files += 1
                solved_this_frame = True
            except PhopsError as exc:
                skipped_files += 1
                report(
                    self.reporter,
                    "warning",
                    f"{image_path.name}: {exc}",
                    stage="pipeline",
                    event="frame_end",
                    total_files=len(files_to_process),
                    current_index=current_index,
                    file_name=image_path.name,
                    frame_status="skipped",
                    solved=False,
                )
                continue

            try:
                data, header, wcs = load_fits_image(solved_fits)

                ra_img, dec_img, observation_time = self.astrometry_solver.extract_field_coordinates(
                    header,
                    image_path.name,
                    wcs=wcs,
                )
                gaia_patch = self.astrometry_solver.catalog_patch_path(ra_img, dec_img, observation_time)
                if not gaia_patch.exists():
                    gaia_patch = self.astrometry_solver.ensure_reference_patch(ra_img, dec_img, observation_time)

                matched_stars, all_detected = self.photometry.get_clean_gaia_matches(solved_fits, gaia_patch)
                if matched_stars is None or len(matched_stars) < 5 or all_detected is None:
                    skipped_files += 1
                    report(
                        self.reporter,
                        "warning",
                        f"{image_path.name}: not enough matched stars for calibration.",
                        stage="pipeline",
                        event="frame_end",
                        total_files=len(files_to_process),
                        current_index=current_index,
                        file_name=image_path.name,
                        frame_status="skipped",
                        solved=solved_this_frame,
                    )
                    continue

                dra_corr, ddec, gmag = calculate_residuals(matched_stars)
                astrometry_rows = [
                    {
                        "filename": image_path.name,
                        "dra_corr": dra_item,
                        "ddec": ddec_item,
                        "gmag": gmag_item,
                    }
                    for dra_item, ddec_item, gmag_item in zip(dra_corr, ddec, gmag)
                ]
                append_rows_to_csv(astrometry_csv, astrometry_rows)

                transformed_stars = self.photometry.transform_gaia_to_filter(matched_stars)
                measured_stars, median_fwhm = self.photometry.perform_aperture_photometry(
                    data,
                    transformed_stars,
                    all_detected,
                )
                zeropoint_plot = self.config.paths.plot_dir / f"{image_path.stem}_zeropoint.png"
                zp_function, zp_scatter, _, zp_average = self.photometry.calculate_zeropoint_model(
                    measured_stars,
                    save_plot=True,
                    output_path=zeropoint_plot,
                )
                generated_plots.append(zeropoint_plot)

                target_info = self.target_manager.resolve(header)
                report(
                    self.reporter,
                    "info",
                    f"Measuring target aperture for {image_path.name}",
                    stage="target",
                )
                target_result = self.photometry.measure_target(
                    data,
                    wcs,
                    target_info,
                    zp_function,
                    median_fwhm,
                    all_detected,
                    image_path.name,
                    zp_average,
                )
                if target_result is None:
                    skipped_files += 1
                    report(
                        self.reporter,
                        "warning",
                        f"{image_path.name}: target measurement failed.",
                        stage="pipeline",
                        event="frame_end",
                        total_files=len(input_files),
                        current_index=current_index,
                        file_name=image_path.name,
                        frame_status="skipped",
                        solved=solved_this_frame,
                    )
                    continue

                result_row = {
                    "filename": image_path.name,
                    "jd": target_info.jd,
                    "mag_inst": target_result["mag_inst"],
                    "mag_calib": target_result["mag_calib"],
                    "snr": target_result["snr"],
                    "mag_err": target_result["err"],
                    "x_target": target_result["x"],
                    "y_target": target_result["y"],
                    "bg": target_result["BG"],
                    "zp": target_result["zp"],
                    "zp_scatter": zp_scatter,
                    "fwhm": median_fwhm,
                }
                if target_info.r is not None and target_info.delta is not None and target_info.alpha is not None:
                    result_row["reduced_mag"] = target_result["mag_calib"] - 5 * np.log10(target_info.r * target_info.delta)
                    result_row["r_au"] = target_info.r
                    result_row["delta_au"] = target_info.delta
                    result_row["alpha"] = target_info.alpha

                append_rows_to_csv(photometry_csv, [result_row])
                measured_files += 1
                completed_frames.add(image_path.name)
                _write_run_state(run_state_path, self.config.paths.input_dir, completed_frames)
                report(
                    self.reporter,
                    "info",
                    f"{image_path.name}: target magnitude {target_result['mag_calib']:.3f} mag",
                    stage="pipeline",
                    event="frame_end",
                    total_files=len(files_to_process),
                    current_index=current_index,
                    file_name=image_path.name,
                    frame_status="measured",
                    solved=solved_this_frame,
                )
            except TargetResolutionError as exc:
                skipped_files += 1
                report(
                    self.reporter,
                    "warning",
                    f"{image_path.name}: {exc}",
                    stage="pipeline",
                    event="frame_end",
                    total_files=len(files_to_process),
                    current_index=current_index,
                    file_name=image_path.name,
                    frame_status="skipped",
                    solved=solved_this_frame,
                )
            except Exception as exc:
                skipped_files += 1
                report(
                    self.reporter,
                    "warning",
                    f"{image_path.name}: unexpected error: {exc}",
                    stage="pipeline",
                    event="frame_end",
                    total_files=len(files_to_process),
                    current_index=current_index,
                    file_name=image_path.name,
                    frame_status="skipped",
                    solved=solved_this_frame,
                )

        if self.config.plots.plot_astrometry and astrometry_csv.exists():
            astrometry_plot = self.config.paths.plot_dir / "astrometry_residuals.png"
            plot_astrometry_residuals(
                astrometry_csv,
                output_path=astrometry_plot,
                reporter=self.reporter,
            )
            generated_plots.append(astrometry_plot)

        if self.config.plots.plot_light_curve and photometry_csv.exists():
            light_curve_png = self.config.paths.plot_dir / "light_curve.png"
            light_curve_pdf = self.config.paths.plot_dir / "light_curve.pdf" if self.config.plots.light_curve_pdf else None
            light_curve_result = plot_photometry_light_curve(
                photometry_csv,
                output_png=light_curve_png,
                output_pdf=light_curve_pdf,
                config=self.config,
                reporter=self.reporter,
            )
            if light_curve_result is not None:
                generated_plots.append(light_curve_result.png_path)
                if light_curve_result.pdf_path is not None:
                    generated_plots.append(light_curve_result.pdf_path)

        report(
            self.reporter,
            "info",
            (f"Completed run: total={len(files_to_process)} solved={solved_files} measured={measured_files} skipped={skipped_files}"),
            stage="pipeline",
            event="run_complete",
            total_files=len(files_to_process),
            solved_files=solved_files,
            measured_files=measured_files,
            skipped_files=skipped_files,
        )

        return PipelineSummary(
            total_files=len(files_to_process),
            solved_files=solved_files,
            measured_files=measured_files,
            skipped_files=skipped_files,
            photometry_csv=photometry_csv,
            astrometry_csv=astrometry_csv,
            generated_plots=generated_plots,
        )


def run_pipeline_from_file(
    config_path: str | Path,
    reporter: ProgressReporter | None = None,
    overwrite: bool = True,
    resume: bool = False,
) -> PipelineSummary:
    """Load configuration from disk and run the pipeline."""

    config = load_config(config_path)
    runner = PipelineRunner(config=config, reporter=reporter)
    with capture_python_warnings(reporter):
        return runner.run(overwrite=overwrite, resume=resume)
