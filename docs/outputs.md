# Outputs

<p align="center">
  <img src="assets/phops-logo.png" alt="PhoPS logo" width="180">
</p>

PhoPS writes three output categories.

## Tables
- `paths.output_photometry`
  Target photometry per processed frame. The `mag_err` column is the formal aperture-photometry uncertainty, and `sigma_total` is the quadratic sum of `mag_err` and the bootstrap zero-point uncertainty.
- `paths.output_astrometry`
  Residual table for Gaia matched stars.
- `paths.output_reference_star_timeseries`
  Optional per-frame calibrated photometry for valid reference stars when `photometry.export_reference_star_timeseries` is enabled. It includes `sigma_total` for each reference star/frame when bootstrap uncertainty is available.
- `photometry.analysis_summary_filename`
  YAML summary with threshold mode, selected RANSAC threshold, inlier counts, zero-point scatter, bootstrap diagnostics, and per-image uncertainty statistics when `photometry.write_analysis_summary` is enabled.
- `output/.phops-run-state.json`
  Hidden checkpoint file used by `phops run` to offer `resume` and skip frames that were already measured before an interruption.

## Plots
- `output/plots/<frame>_zeropoint.png`
  Zeropoint fit diagnostic.
- `output/plots/astrometry_residuals.png`
  Residual summary across frames.
- `output/plots/<frame>_sources_<mode>.png`
  Detection and match overview in pixel or WCS space.
- `output/plots/light_curve.png`
  Publication-oriented scientific light curve with calibrated magnitude, 3-sigma y-range, error bars, and optional quality panels.
- `output/plots/light_curve.pdf`
  Vector-export companion for publication or reports when PDF output is enabled.

## Cutouts
- `output/cutouts/cutout_target_<frame>.png`
  Saved target cutout around the measured source.
