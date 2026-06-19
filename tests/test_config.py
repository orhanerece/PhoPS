from pathlib import Path

import pytest

from phops.errors import ConfigurationError
from phops.config import load_config


def test_load_example_config() -> None:
    config = load_config(Path("examples/example_config.yaml"))
    assert config.photometry.mode == "asteroid"
    assert config.photometry.export_reference_star_timeseries is False
    assert config.photometry.ransac_threshold_mode == "auto"
    assert config.photometry.ransac_threshold == 0.10
    assert config.photometry.zp_error_method == "bootstrap"
    assert config.photometry.zp_bootstrap_iterations == 1000
    assert config.photometry.write_analysis_summary is True
    assert config.paths.output_photometry == "photometry.csv"
    assert config.paths.output_reference_star_timeseries == "reference_star_timeseries.csv"
    assert config.paths.solve_dir.name == "output"


def test_load_hourangle_coords(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
fits_keywords:
  ra_key: "RA"
  dec_key: "DEC"
  date_key: "DATE-OBS"
  exposure_key: "EXPTIME"
  jd_key: "JD"
astrometry:
  solve_mode: "solve"
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
  coords: ["07 57 16.4841", "+35 46 55.826"]
  coords_unit: "hourangle_deg"
  filter: "I"
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
    config = load_config(config_path)
    assert round(config.photometry.coords[0], 6) == 119.318684
    assert round(config.photometry.coords[1], 6) == 35.782174


def test_invalid_ransac_threshold_mode_raises_clear_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
fits_keywords:
  ra_key: "RA"
  dec_key: "DEC"
  date_key: "DATE-OBS"
  exposure_key: "EXPTIME"
  jd_key: "JD"
instrument:
  pixel_scale: 0.62
photometry:
  mode: "star"
  coords: [10.0, 20.0]
  ransac_threshold_mode: "adaptive"
paths:
  input_dir: "input"
  temp_dir: "temp"
  index_dir: "indexes"
  solve_dir: "output"
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="ransac_threshold_mode"):
        load_config(config_path)
