from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from phops.config import load_config
from phops.photometry import Photometry, select_threshold_by_inlier_knee


def _write_bootstrap_config(tmp_path: Path, *, min_inliers: int = 3) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
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
  zp_bootstrap_iterations: 200
  zp_bootstrap_random_seed: 123
  zp_bootstrap_min_inliers: {min_inliers}
paths:
  input_dir: "input"
  temp_dir: "temp"
  index_dir: "indexes"
  solve_dir: "output"
plots:
  plot_astrometry: false
  plot_image: false
  plot_light_curve: false
        """.strip(),
        encoding="utf-8",
    )
    return config_path


def _bootstrap_table() -> Table:
    radius = np.asarray([0, 10, 20, 30, 40, 50], dtype=float)
    inst_mag = np.asarray([10, 10, 10, 10, 10, 10], dtype=float)
    return Table(
        {
            "r_dist": radius,
            "inst_mag": inst_mag,
            "standard_mag": inst_mag + 20.0 + 0.01 * radius + np.asarray([0.00, 0.02, -0.01, 0.03, -0.02, 0.01]),
            "zp_inlier": [True, True, True, True, True, True],
        }
    )


def test_select_threshold_by_inlier_knee_returns_expected_region() -> None:
    thresholds = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    n_inliers = np.array([30, 50, 70, 85, 95, 100, 102, 103])

    selected, diagnostics = select_threshold_by_inlier_knee(thresholds, n_inliers, min_inliers=30)

    assert selected == 0.04
    assert diagnostics["status"] == "ok"
    assert diagnostics["n_inliers_at_knee"] == 85


def test_select_threshold_by_inlier_knee_monotonises_counts() -> None:
    thresholds = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    n_inliers = np.array([30, 50, 49, 85, 84, 100])

    _, diagnostics = select_threshold_by_inlier_knee(thresholds, n_inliers, min_inliers=30)

    assert diagnostics["n_inliers_monotonic"] == [30, 50, 50, 85, 85, 100]


def test_select_threshold_by_inlier_knee_degenerate_curve_falls_back() -> None:
    selected, diagnostics = select_threshold_by_inlier_knee(
        np.array([0.01, 0.02, 0.03, 0.04]),
        np.array([50, 50, 50, 50]),
        fallback_threshold=0.10,
    )

    assert selected == 0.10
    assert diagnostics["status"] == "fallback"
    assert diagnostics["reason"] == "degenerate_inlier_curve"


def test_bootstrap_zeropoint_uncertainty_is_reproducible_and_radius_specific(tmp_path: Path) -> None:
    config = load_config(_write_bootstrap_config(tmp_path, min_inliers=3))
    photometry = Photometry(config)
    table = _bootstrap_table()
    radii = np.asarray([5.0, 45.0])

    first, first_summary = photometry.bootstrap_zeropoint_uncertainty(
        table,
        radii,
        np.random.default_rng(config.photometry.zp_bootstrap_random_seed),
    )
    second, second_summary = photometry.bootstrap_zeropoint_uncertainty(
        table,
        radii,
        np.random.default_rng(config.photometry.zp_bootstrap_random_seed),
    )

    assert first_summary["zp_error_method_used"] == "bootstrap"
    assert second_summary["zp_error_method_used"] == "bootstrap"
    assert np.all(np.isfinite(first))
    assert np.allclose(first, second)
    assert not np.isclose(first[0], first[1])


def test_bootstrap_zeropoint_uncertainty_handles_insufficient_inliers(tmp_path: Path) -> None:
    config = load_config(_write_bootstrap_config(tmp_path, min_inliers=30))
    photometry = Photometry(config)

    errors, summary = photometry.bootstrap_zeropoint_uncertainty(
        _bootstrap_table(),
        np.asarray([5.0, 45.0]),
        np.random.default_rng(config.photometry.zp_bootstrap_random_seed),
    )

    assert summary["zp_error_method_used"] == "bootstrap_failed"
    assert np.isnan(errors).all()
