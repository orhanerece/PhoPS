"""Pipeline orchestration."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .astrometry import AstrometrySolver
from .config import AppConfig, load_config
from .errors import PhopsError, PipelineError, TargetResolutionError
from .photometry import Photometry, select_threshold_by_inlier_knee
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


def _as_float(value: object) -> float:
    if np.ma.is_masked(value):
        return np.nan
    if hasattr(value, "value"):
        value = value.value
    return float(value)


def _as_csv_value(value: object) -> object:
    if np.ma.is_masked(value):
        return ""
    if hasattr(value, "value"):
        value = value.value
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _observation_jd(observation_time: object) -> float:
    if hasattr(observation_time, "jd"):
        return float(observation_time.jd)
    return float(observation_time)


def _quadrature_sigma(formal_sigma: float, zeropoint_sigma: float) -> float:
    if not (np.isfinite(formal_sigma) and np.isfinite(zeropoint_sigma)):
        return np.nan
    return float(np.sqrt(formal_sigma**2 + zeropoint_sigma**2))


def _finite_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _threshold_grid(config: AppConfig) -> np.ndarray:
    grid = config.photometry.ransac_threshold_grid
    return np.arange(grid.start, grid.stop + 0.5 * grid.step, grid.step)


def _target_radius_from_result(data: np.ndarray, target_result: dict[str, float]) -> float:
    if "radius" in target_result:
        return float(target_result["radius"])
    return float(np.sqrt((target_result["x"] - data.shape[1] / 2) ** 2 + (target_result["y"] - data.shape[0] / 2) ** 2))


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_analysis_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_json_safe(payload), sort_keys=False), encoding="utf-8")


def _reference_star_timeseries_rows(
    *,
    filename: str,
    jd: float,
    measured_stars,
    zp_function,
    zp_average: float,
    zeropoint_mode: str,
    zp_errors: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if zp_errors is None:
        zp_errors = np.full(len(measured_stars), np.nan, dtype=float)
    for index, star in enumerate(measured_stars):
        standard_mag = _as_float(star["standard_mag"])
        inst_mag = _as_float(star["inst_mag"])
        mag_err = _as_float(star["mag_err"])
        radius = _as_float(star["r_dist"])
        gaia_gmag = _as_float(star["gaia_gmag"])
        if not (
            np.isfinite(standard_mag)
            and np.isfinite(inst_mag)
            and np.isfinite(mag_err)
            and np.isfinite(radius)
            and np.isfinite(gaia_gmag)
        ):
            continue
        if "zp_valid" in measured_stars.colnames and not bool(star["zp_valid"]):
            continue

        zeropoint = float(zp_average) if zeropoint_mode == "average" else float(zp_function(radius))
        sigma_total = _quadrature_sigma(mag_err, float(zp_errors[index]))
        rows.append(
            {
                "filename": filename,
                "jd": jd,
                "source_id": _as_csv_value(star["source_id"]) if "source_id" in measured_stars.colnames else "",
                "gaia_gmag": gaia_gmag,
                "bp_rp": _as_float(star["bp_rp"]) if "bp_rp" in measured_stars.colnames else np.nan,
                "x": _as_float(star["x_precise"] if "x_precise" in measured_stars.colnames else star["xcentroid"]),
                "y": _as_float(star["y_precise"] if "y_precise" in measured_stars.colnames else star["ycentroid"]),
                "mag_inst": inst_mag,
                "zp": zeropoint,
                "mag_calib": float(inst_mag + zeropoint),
                "mag_err": mag_err,
                "sigma_total": sigma_total,
                "zp_inlier": bool(star["zp_inlier"]) if "zp_inlier" in measured_stars.colnames else "",
                "standard_mag": standard_mag,
                "snr": _as_float(star["snr"]) if "snr" in measured_stars.colnames else np.nan,
            }
        )
    return rows


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
        reference_star_timeseries_csv = self.config.paths.reference_star_timeseries_csv_path
        analysis_summary_path = self.config.paths.solve_dir / self.config.photometry.analysis_summary_filename
        run_state_path = self.config.paths.run_state_path
        existing_state = inspect_existing_run(self.config) if resume else None
        completed_frames: set[str] = set()
        if overwrite:
            photometry_csv.unlink(missing_ok=True)
            astrometry_csv.unlink(missing_ok=True)
            if self.config.photometry.export_reference_star_timeseries:
                reference_star_timeseries_csv.unlink(missing_ok=True)
            if self.config.photometry.write_analysis_summary:
                analysis_summary_path.unlink(missing_ok=True)
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
        run_started_at = datetime.now(timezone.utc)
        run_timer = time.perf_counter()
        rng = np.random.default_rng(self.config.photometry.zp_bootstrap_random_seed)
        selected_threshold = (
            float(self.config.photometry.ransac_threshold)
            if self.config.photometry.ransac_threshold_mode == "fixed"
            else None
        )
        selected_from_image: str | None = None
        threshold_selection: dict[str, object] = {}
        if self.config.photometry.ransac_threshold_mode == "fixed":
            threshold_selection = {
                "status": "fixed",
                "selected_threshold": selected_threshold,
                "fallback_used": False,
            }
        per_image_summary: list[dict[str, object]] = []
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
            frame_timer = time.perf_counter()
            frame_summary: dict[str, object] = {
                "filename": image_path.name,
                "status": "started",
                "warnings": [],
            }
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
                frame_summary.update(
                    {
                        "status": "skipped",
                        "runtime_seconds": time.perf_counter() - frame_timer,
                        "warnings": [str(exc)],
                    }
                )
                per_image_summary.append(frame_summary)
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
                frame_summary["n_detected_sources"] = int(len(all_detected)) if all_detected is not None else 0
                frame_summary["n_matched_reference_stars"] = int(len(matched_stars)) if matched_stars is not None else 0
                if matched_stars is None or len(matched_stars) < 5 or all_detected is None:
                    skipped_files += 1
                    frame_summary.update(
                        {
                            "status": "skipped",
                            "runtime_seconds": time.perf_counter() - frame_timer,
                            "warnings": ["not enough matched stars for calibration"],
                        }
                    )
                    per_image_summary.append(frame_summary)
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
                if self.config.photometry.ransac_threshold_mode == "auto":
                    should_select = selected_threshold is None or not self.config.photometry.ransac_auto_reuse_for_sequence
                    if should_select:
                        scan = self.photometry.scan_ransac_thresholds(measured_stars, _threshold_grid(self.config))
                        selected_threshold, threshold_selection = select_threshold_by_inlier_knee(
                            scan["thresholds"],
                            scan["n_inliers"],
                            min_inliers=self.config.photometry.ransac_auto_min_inliers,
                            fallback_threshold=self.config.photometry.ransac_auto_fallback_threshold,
                        )
                        selected_from_image = image_path.name
                        threshold_selection = {
                            **threshold_selection,
                            "selected_threshold": selected_threshold,
                            "fallback_used": threshold_selection.get("status") == "fallback",
                            "fallback_reason": threshold_selection.get("reason"),
                        }
                        for key in ("thresholds", "n_inliers"):
                            threshold_selection.setdefault(key, scan[key])

                zeropoint_plot = self.config.paths.plot_dir / f"{image_path.stem}_zeropoint.png"
                zp_function, zp_scatter, _, zp_average = self.photometry.calculate_zeropoint_model(
                    measured_stars,
                    save_plot=True,
                    output_path=zeropoint_plot,
                    ransac_threshold=selected_threshold,
                )
                generated_plots.append(zeropoint_plot)
                zp_diagnostics = measured_stars.meta.get("zeropoint_diagnostics", {})

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
                    frame_summary.update(
                        {
                            "status": "skipped",
                            "runtime_seconds": time.perf_counter() - frame_timer,
                            "selected_threshold": selected_threshold,
                            "n_valid_reference_stars": zp_diagnostics.get("n_valid_reference_stars"),
                            "n_ransac_inliers": zp_diagnostics.get("n_ransac_inliers"),
                            "zp_slope": zp_diagnostics.get("zp_slope"),
                            "zp_intercept": zp_diagnostics.get("zp_intercept"),
                            "zp_scatter": zp_scatter,
                            "warnings": ["target measurement failed"],
                        }
                    )
                    per_image_summary.append(frame_summary)
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

                target_radius = _target_radius_from_result(data, target_result)
                reference_radii = np.asarray(measured_stars["r_dist"], dtype=float)
                object_radii = np.concatenate(([target_radius], reference_radii))
                zp_errors, zp_error_summary = self.photometry.bootstrap_zeropoint_uncertainty(
                    measured_stars,
                    object_radii,
                    rng,
                )
                target_zp_error = float(zp_errors[0])
                reference_zp_errors = zp_errors[1:]
                target_sigma_total = _quadrature_sigma(float(target_result["err"]), target_zp_error)
                reference_formal_errors = np.asarray(measured_stars["mag_err"], dtype=float)
                reference_sigma_total = [
                    _quadrature_sigma(formal_error, zp_error)
                    for formal_error, zp_error in zip(reference_formal_errors, reference_zp_errors)
                ]
                all_sigma_total = [target_sigma_total, *reference_sigma_total]
                sigma_stats = _finite_stats(all_sigma_total)
                if zp_error_summary.get("warning"):
                    frame_summary["warnings"] = [*frame_summary.get("warnings", []), zp_error_summary["warning"]]

                if self.config.photometry.export_reference_star_timeseries:
                    reference_rows = _reference_star_timeseries_rows(
                        filename=image_path.name,
                        jd=_observation_jd(observation_time),
                        measured_stars=measured_stars,
                        zp_function=zp_function,
                        zp_average=zp_average,
                        zeropoint_mode=self.config.photometry.zeropoint,
                        zp_errors=reference_zp_errors,
                    )
                    append_rows_to_csv(reference_star_timeseries_csv, reference_rows)

                result_row = {
                    "filename": image_path.name,
                    "jd": target_info.jd,
                    "mag_inst": target_result["mag_inst"],
                    "mag_calib": target_result["mag_calib"],
                    "snr": target_result["snr"],
                    "mag_err": target_result["err"],
                    "sigma_total": target_sigma_total,
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
                frame_summary.update(
                    {
                        "status": "measured",
                        "runtime_seconds": time.perf_counter() - frame_timer,
                        "selected_threshold": selected_threshold,
                        "n_valid_reference_stars": zp_diagnostics.get("n_valid_reference_stars"),
                        "n_ransac_inliers": zp_diagnostics.get("n_ransac_inliers"),
                        "zp_slope": zp_diagnostics.get("zp_slope"),
                        "zp_intercept": zp_diagnostics.get("zp_intercept"),
                        "zp_scatter": zp_scatter,
                        "zp_error_method_used": zp_error_summary.get("zp_error_method_used"),
                        "zp_bootstrap_error_median": zp_error_summary.get("zp_bootstrap_error_median"),
                        "zp_bootstrap_error_min": zp_error_summary.get("zp_bootstrap_error_min"),
                        "zp_bootstrap_error_max": zp_error_summary.get("zp_bootstrap_error_max"),
                        "sigma_total_median": sigma_stats["median"],
                        "sigma_total_min": sigma_stats["min"],
                        "sigma_total_max": sigma_stats["max"],
                    }
                )
                per_image_summary.append(frame_summary)
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
                frame_summary.update(
                    {
                        "status": "skipped",
                        "runtime_seconds": time.perf_counter() - frame_timer,
                        "warnings": [str(exc)],
                    }
                )
                per_image_summary.append(frame_summary)
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
                frame_summary.update(
                    {
                        "status": "skipped",
                        "runtime_seconds": time.perf_counter() - frame_timer,
                        "warnings": [f"unexpected error: {exc}"],
                    }
                )
                per_image_summary.append(frame_summary)
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

        if self.config.photometry.write_analysis_summary:
            finished_at = datetime.now(timezone.utc)
            summary_payload = {
                "run": {
                    "started_at": run_started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "total_runtime_seconds": time.perf_counter() - run_timer,
                    "number_of_images_input": len(input_files),
                    "number_of_images_processed": measured_files,
                    "number_of_images_failed": skipped_files,
                },
                "configuration": {
                    "threshold_mode": self.config.photometry.ransac_threshold_mode,
                    "fixed_threshold": self.config.photometry.ransac_threshold,
                    "threshold_grid": {
                        "start": self.config.photometry.ransac_threshold_grid.start,
                        "stop": self.config.photometry.ransac_threshold_grid.stop,
                        "step": self.config.photometry.ransac_threshold_grid.step,
                    },
                    "auto_method": self.config.photometry.ransac_auto_method,
                    "selected_threshold": selected_threshold,
                    "selected_from_image": selected_from_image,
                    "bootstrap_iterations": self.config.photometry.zp_bootstrap_iterations,
                    "bootstrap_random_seed": self.config.photometry.zp_bootstrap_random_seed,
                    "zp_error_method": self.config.photometry.zp_error_method,
                },
                "threshold_selection": threshold_selection,
                "per_image": per_image_summary,
            }
            _write_analysis_summary(analysis_summary_path, summary_payload)

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
