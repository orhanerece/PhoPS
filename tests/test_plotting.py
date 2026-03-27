from pathlib import Path

import pandas as pd

from phops.config import load_config
from phops.plotting import _light_curve_y_limits, plot_photometry_light_curve


def _write_config(config_path: Path) -> None:
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
  plot_light_curve: true
  image_scale: "pixel"
  light_curve_x_axis: "relative_seconds"
  light_curve_aux_panels: true
  light_curve_pdf: true
  light_curve_event_window: [15, 45]
  light_curve_event_unit: "relative_seconds"
catalog: "gaiadr3.gaia_source"
        """.strip(),
        encoding="utf-8",
    )


def test_plot_photometry_light_curve_creates_png_and_pdf(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    config = load_config(config_path)
    config.ensure_runtime_dirs()

    photometry_csv = config.paths.photometry_csv_path
    photometry_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "filename": [f"frame{i:02d}.fits" for i in range(8)],
            "jd": [2460000.5 + (i * 10.0 / 86400.0) for i in range(8)],
            "mag_calib": [12.1, 12.11, 12.09, 12.45, 12.1, 12.08, 12.09, 12.1],
            "mag_err": [0.02, 0.02, 0.02, 0.05, 0.02, 0.02, 0.02, 0.02],
            "snr": [120, 118, 119, 40, 121, 123, 122, 120],
            "zp_scatter": [0.01, 0.011, 0.011, 0.04, 0.01, 0.011, 0.01, 0.01],
            "fwhm": [3.1, 3.0, 3.0, 5.2, 3.1, 3.0, 3.1, 3.0],
        }
    )
    frame.to_csv(photometry_csv, index=False)

    png_path = config.paths.plot_dir / "light_curve.png"
    pdf_path = config.paths.plot_dir / "light_curve.pdf"
    result = plot_photometry_light_curve(
        photometry_csv,
        output_png=png_path,
        output_pdf=pdf_path,
        config=config,
    )

    assert result is not None
    assert png_path.exists()
    assert pdf_path.exists()


def test_light_curve_y_limits_use_three_sigma_window() -> None:
    values = pd.Series([12.10, 12.11, 12.09, 12.10, 12.12, 12.08, 12.10, 12.60])
    outliers = pd.Series([False, False, False, False, False, False, False, True])

    y_limits = _light_curve_y_limits(values.to_numpy(), outlier_mask=outliers.to_numpy())

    assert y_limits is not None
    lower, upper = y_limits
    assert lower < 12.09
    assert upper > 12.11
    assert upper < 12.60
