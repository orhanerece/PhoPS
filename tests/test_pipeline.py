from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table

from phops.astrometry import AstrometrySolver
from phops.config import load_config
from phops.errors import AstrometrySolveError
from phops.pipeline import (
    PipelineRunner,
    build_run_summary_items,
    format_run_summary,
    inspect_existing_run,
)
from phops.target import TargetInfo


class FakeAstrometrySolver:
    def __init__(self, config) -> None:
        self.config = config
        self.patch_path = config.paths.temp_dir / "gaia_reference_test.fits"

    def _write_patch(self) -> Path:
        self.patch_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.patch_path.exists():
            Table(
                {
                    "ra": [10.0, 10.1, 10.2, 10.3, 10.4],
                    "dec": [20.0, 20.1, 20.2, 20.3, 20.4],
                    "phot_g_mean_mag": [12.0, 12.2, 12.4, 12.6, 12.8],
                    "bp_rp": [0.5, 0.6, 0.7, 0.8, 0.9],
                    "pmra": [1, 1, 1, 1, 1],
                    "pmdec": [1, 1, 1, 1, 1],
                }
            ).write(self.patch_path, overwrite=True)
        return self.patch_path

    def solve(self, image_path: Path) -> Path:
        output_path = self.config.paths.solve_dir / f"{image_path.stem}_new.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = fits.getdata(image_path)
        header = fits.getheader(image_path)
        fits.writeto(output_path, data=data, header=header, overwrite=True)
        self._write_patch()
        return output_path

    def extract_field_coordinates(self, header, image_name: str = "<header>"):
        del image_name
        return 10.0, 20.0, 2461125.5

    def catalog_patch_path(self, ra: float, dec: float, observation_time) -> Path:
        del ra, dec, observation_time
        return self._write_patch()

    def ensure_reference_patch(self, ra: float, dec: float, observation_time) -> Path:
        del ra, dec, observation_time
        return self._write_patch()


class FakePhotometry:
    def __init__(self, config) -> None:
        self.config = config

    def get_clean_gaia_matches(self, image_path: Path, gaia_patch: Path):
        del image_path, gaia_patch
        matched = Table(
            {
                "xcentroid": [10, 20, 30, 40, 50],
                "ycentroid": [12, 22, 32, 42, 52],
                "img_ra": [10.0, 10.1, 10.2, 10.3, 10.4],
                "img_dec": [20.0, 20.1, 20.2, 20.3, 20.4],
                "gaia_ra": [10.0, 10.1, 10.2, 10.3, 10.4],
                "gaia_dec": [20.0, 20.1, 20.2, 20.3, 20.4],
                "gaia_gmag": [12.0, 12.2, 12.4, 12.6, 12.8],
                "bp_rp": [0.5, 0.6, 0.7, 0.8, 0.9],
                "r_dist": [10, 20, 30, 40, 50],
            }
        )
        all_detected = Table({"xcentroid": [60.0], "ycentroid": [60.0]})
        return matched, all_detected

    def transform_gaia_to_filter(self, matched_table: Table) -> Table:
        matched_table["standard_mag"] = [12.0, 12.2, 12.4, 12.6, 12.8]
        return matched_table

    def perform_aperture_photometry(self, data, matched_table: Table, image_sources: Table):
        del data, image_sources
        matched_table["inst_mag"] = [10.0, 10.1, 10.2, 10.3, 10.4]
        return matched_table, 3.0

    def calculate_zeropoint_model(self, matched_table: Table, save_plot: bool = True, output_path: Path | None = None):
        del matched_table, save_plot
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake plot", encoding="utf-8")
        return np.poly1d([0.0, 25.0]), 0.02, 0.0, 25.0

    def measure_target(self, data, wcs, target, zp_function, median_fwhm, all_detected, filename: str, zp_average: float):
        del data, wcs, target, zp_function, median_fwhm, all_detected, filename, zp_average
        return {
            "mag_inst": 10.0,
            "mag_calib": 15.0,
            "snr": 100.0,
            "err": 0.01,
            "x": 50.0,
            "y": 60.0,
            "BG": 1.0,
            "zp": 25.0,
        }


class FakeTargetManager:
    def resolve(self, header) -> TargetInfo:
        del header
        return TargetInfo(ra=10.0, dec=20.0, jd=2460000.5, r=None, delta=None, alpha=None)


def test_run_summary_lists_key_output_paths() -> None:
    config = load_config(Path("examples/example_config.yaml"))

    summary_items = build_run_summary_items(config, total_files=42, overwrite=False)
    summary_text = format_run_summary(summary_items)

    assert "Run summary" in summary_text
    assert str(config.paths.photometry_csv_path) in summary_text
    assert str(config.paths.astrometry_csv_path) in summary_text
    assert str(config.paths.plot_dir) in summary_text
    assert str(config.paths.cutout_dir) in summary_text
    assert "Overwrite" in summary_text
    assert "no" in summary_text


def test_pipeline_runner_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
fits_keywords:
  ra_key: "OBJCTRA"
  dec_key: "OBJCTDEC"
  date_key: "DATE-OBS"
  exposure_key: "EXPTIME"
  jd_key: "JD"
astrometry:
  radius: 0.5
  quad_scales: [0, 2]
  cache_tolerance: 0.1
instrument:
  pixel_scale: 0.62
  gain: 1.0
  read_noise: 5.0
source_detection:
  fwhm_guess: 5.0
  threshold_sigma: 3.0
  min_area: 5
  edge_margin: 10
matching:
  isolation_radius_arcsec: 0.2
  match_constraint_arcsec: 1.0
photometry:
  mode: "star"
  coords: [10.0, 20.0]
  filter: "R"
  aperture_method: "fixed_pixel"
  aperture: 4
  annulus_inner: 6
  annulus_outer: 8
  zeropoint: "fit"
paths:
  input_dir: "input"
  temp_dir: "temp"
  index_dir: "indexes"
  solve_dir: "output"
  output_photometry: "photometry.csv"
  output_astrometry: "astrometry.csv"
  file_extension: "fits"
plots:
  plot_astrometry: false
  plot_image: false
  image_scale: "pixel"
catalog: "gaiadr3.gaia_source"
        """.strip(),
        encoding="utf-8",
    )

    header = fits.Header()
    header["OBJCTRA"] = "10:00:00"
    header["OBJCTDEC"] = "20:00:00"
    header["DATE-OBS"] = "2026-03-26T00:00:00"
    header["MJD-OBS"] = 61125.0
    header["EXPTIME"] = 30.0
    header["JD"] = 2461125.5
    fits.writeto(input_dir / "frame01.fits", data=np.ones((80, 80)), header=header, overwrite=True)

    config = load_config(config_path)
    runner = PipelineRunner(
        config=config,
        astrometry_solver=FakeAstrometrySolver(config),
        photometry=FakePhotometry(config),
        target_manager=FakeTargetManager(),
    )
    summary = runner.run()

    assert summary.total_files == 1
    assert summary.measured_files == 1
    assert (output_dir / "photometry.csv").exists()
    assert (output_dir / "astrometry.csv").exists()

    photometry_frame = pd.read_csv(output_dir / "photometry.csv")
    astrometry_frame = pd.read_csv(output_dir / "astrometry.csv")
    assert len(photometry_frame) == 1
    assert len(astrometry_frame) == 5


def test_astrometry_solver_existing_wcs_returns_original_frame(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
fits_keywords:
  ra_key: "CRVAL1"
  dec_key: "CRVAL2"
  date_key: "DATE-OBS"
  exposure_key: "EXPTIME"
  jd_key: "JD"
astrometry:
  solve_mode: "existing_wcs"
  radius: 0.5
  quad_scales: [0, 2]
  cache_tolerance: 0.1
instrument:
  pixel_scale: 0.62
  gain: 1.0
  read_noise: 5.0
source_detection:
  fwhm_guess: 5.0
  threshold_sigma: 3.0
  min_area: 5
  edge_margin: 10
matching:
  isolation_radius_arcsec: 0.2
  match_constraint_arcsec: 1.0
photometry:
  mode: "star"
  coords: [10.0, 20.0]
  filter: "R"
  aperture_method: "fixed_pixel"
  aperture: 4
  annulus_inner: 6
  annulus_outer: 8
  zeropoint: "fit"
paths:
  input_dir: "input"
  temp_dir: "temp"
  index_dir: "indexes"
  solve_dir: "output"
  output_photometry: "photometry.csv"
  output_astrometry: "astrometry.csv"
  file_extension: "fits"
plots:
  plot_astrometry: false
  plot_image: false
  image_scale: "pixel"
catalog: "gaiadr3.gaia_source"
        """.strip(),
        encoding="utf-8",
    )

    header = fits.Header()
    header["DATE-OBS"] = "2026-03-26T00:00:00"
    header["MJD-OBS"] = 61125.0
    header["EXPTIME"] = 30.0
    header["JD"] = 2461125.5
    header["CRPIX1"] = 40.0
    header["CRPIX2"] = 40.0
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CD1_1"] = -0.0002
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.0002
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    frame_path = input_dir / "frame01.fits"
    fits.writeto(frame_path, data=np.ones((80, 80)), header=header, overwrite=True)

    config = load_config(config_path)
    solver = AstrometrySolver(config)

    assert solver.solve(frame_path) == frame_path.resolve()


def test_astrometry_solver_existing_wcs_requires_celestial_wcs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
fits_keywords:
  ra_key: "CRVAL1"
  dec_key: "CRVAL2"
  date_key: "DATE-OBS"
  exposure_key: "EXPTIME"
  jd_key: "JD"
astrometry:
  solve_mode: "existing_wcs"
  radius: 0.5
  quad_scales: [0, 2]
  cache_tolerance: 0.1
instrument:
  pixel_scale: 0.62
  gain: 1.0
  read_noise: 5.0
source_detection:
  fwhm_guess: 5.0
  threshold_sigma: 3.0
  min_area: 5
  edge_margin: 10
matching:
  isolation_radius_arcsec: 0.2
  match_constraint_arcsec: 1.0
photometry:
  mode: "star"
  coords: [10.0, 20.0]
  filter: "R"
  aperture_method: "fixed_pixel"
  aperture: 4
  annulus_inner: 6
  annulus_outer: 8
  zeropoint: "fit"
paths:
  input_dir: "input"
  temp_dir: "temp"
  index_dir: "indexes"
  solve_dir: "output"
  output_photometry: "photometry.csv"
  output_astrometry: "astrometry.csv"
  file_extension: "fits"
plots:
  plot_astrometry: false
  plot_image: false
  image_scale: "pixel"
catalog: "gaiadr3.gaia_source"
        """.strip(),
        encoding="utf-8",
    )

    header = fits.Header()
    header["DATE-OBS"] = "2026-03-26T00:00:00"
    header["MJD-OBS"] = 61125.0
    header["EXPTIME"] = 30.0
    header["JD"] = 2461125.5
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    frame_path = input_dir / "frame01.fits"
    fits.writeto(frame_path, data=np.ones((80, 80)), header=header, overwrite=True)

    config = load_config(config_path)
    solver = AstrometrySolver(config)

    with pytest.raises(AstrometrySolveError, match="existing_wcs"):
        solver.solve(frame_path)


def test_pipeline_runner_resume_skips_completed_frames_and_prunes_partial_astrometry(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
fits_keywords:
  ra_key: "OBJCTRA"
  dec_key: "OBJCTDEC"
  date_key: "DATE-OBS"
  exposure_key: "EXPTIME"
  jd_key: "JD"
astrometry:
  radius: 0.5
  quad_scales: [0, 2]
  cache_tolerance: 0.1
instrument:
  pixel_scale: 0.62
  gain: 1.0
  read_noise: 5.0
source_detection:
  fwhm_guess: 5.0
  threshold_sigma: 3.0
  min_area: 5
  edge_margin: 10
matching:
  isolation_radius_arcsec: 0.2
  match_constraint_arcsec: 1.0
photometry:
  mode: "star"
  coords: [10.0, 20.0]
  filter: "R"
  aperture_method: "fixed_pixel"
  aperture: 4
  annulus_inner: 6
  annulus_outer: 8
  zeropoint: "fit"
paths:
  input_dir: "input"
  temp_dir: "temp"
  index_dir: "indexes"
  solve_dir: "output"
  output_photometry: "photometry.csv"
  output_astrometry: "astrometry.csv"
  file_extension: "fits"
plots:
  plot_astrometry: false
  plot_image: false
  plot_light_curve: false
  image_scale: "pixel"
catalog: "gaiadr3.gaia_source"
        """.strip(),
        encoding="utf-8",
    )

    header = fits.Header()
    header["OBJCTRA"] = "10:00:00"
    header["OBJCTDEC"] = "20:00:00"
    header["DATE-OBS"] = "2026-03-26T00:00:00"
    header["MJD-OBS"] = 61125.0
    header["EXPTIME"] = 30.0
    header["JD"] = 2461125.5
    fits.writeto(input_dir / "frame01.fits", data=np.ones((80, 80)), header=header, overwrite=True)
    fits.writeto(input_dir / "frame02.fits", data=np.ones((80, 80)), header=header, overwrite=True)

    config = load_config(config_path)
    config.ensure_runtime_dirs()

    photometry_seed = pd.DataFrame(
        [
            {
                "filename": "frame01.fits",
                "jd": 2460000.5,
                "mag_inst": 10.0,
                "mag_calib": 15.0,
                "snr": 100.0,
                "mag_err": 0.01,
                "x_target": 50.0,
                "y_target": 60.0,
                "bg": 1.0,
                "zp": 25.0,
                "zp_scatter": 0.02,
                "fwhm": 3.0,
            }
        ]
    )
    photometry_seed.to_csv(config.paths.photometry_csv_path, index=False)

    astrometry_seed = pd.DataFrame(
        [
            {"filename": "frame01.fits", "dra_corr": 0.1, "ddec": 0.1, "gmag": 12.0},
            {"filename": "frame01.fits", "dra_corr": 0.1, "ddec": 0.1, "gmag": 12.1},
            {"filename": "frame01.fits", "dra_corr": 0.1, "ddec": 0.1, "gmag": 12.2},
            {"filename": "frame01.fits", "dra_corr": 0.1, "ddec": 0.1, "gmag": 12.3},
            {"filename": "frame01.fits", "dra_corr": 0.1, "ddec": 0.1, "gmag": 12.4},
            {"filename": "frame02.fits", "dra_corr": 0.2, "ddec": 0.2, "gmag": 13.0},
            {"filename": "frame02.fits", "dra_corr": 0.2, "ddec": 0.2, "gmag": 13.1},
        ]
    )
    astrometry_seed.to_csv(config.paths.astrometry_csv_path, index=False)
    config.paths.run_state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "input_dir": str(config.paths.input_dir.resolve()),
                "completed_frames": ["frame01.fits"],
            }
        ),
        encoding="utf-8",
    )

    existing_state = inspect_existing_run(config)
    assert existing_state.can_resume
    assert existing_state.pending_files == 1

    runner = PipelineRunner(
        config=config,
        astrometry_solver=FakeAstrometrySolver(config),
        photometry=FakePhotometry(config),
        target_manager=FakeTargetManager(),
    )
    summary = runner.run(overwrite=False, resume=True)

    assert summary.total_files == 1
    assert summary.measured_files == 1

    photometry_frame = pd.read_csv(config.paths.photometry_csv_path)
    assert sorted(photometry_frame["filename"].tolist()) == ["frame01.fits", "frame02.fits"]

    astrometry_frame = pd.read_csv(config.paths.astrometry_csv_path)
    assert sorted(astrometry_frame["filename"].unique().tolist()) == ["frame01.fits", "frame02.fits"]
    assert len(astrometry_frame) == 10

    state_payload = json.loads(config.paths.run_state_path.read_text(encoding="utf-8"))
    assert state_payload["completed_frames"] == ["frame01.fits", "frame02.fits"]
