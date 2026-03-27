# Outputs

<p align="center">
  <img src="assets/phops-logo.png" alt="PhoPS logo" width="180">
</p>

PhoPS writes three output categories.

## Tables
- `paths.output_photometry`
  Target photometry per processed frame.
- `paths.output_astrometry`
  Residual table for Gaia matched stars.
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
