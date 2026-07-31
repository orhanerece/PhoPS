# Configuration

<p align="center">
  <img src="assets/phops-logo.png" alt="PhoPS logo" width="180">
</p>

PhoPS uses a YAML config file. Start from `examples/example_config.yaml` or generate one with:

```bash
phops init-config config.yaml
```

## Important Sections
- `fits_keywords`
  FITS header keys for RA, Dec, date, exposure, and optional JD.
- `astrometry`
  Astrometric solve behavior, Gaia patch settings, and cache control.
- `instrument`
  Pixel scale, gain, read noise, and saturation level.
- `photometry`
  Target mode, filter, aperture settings, and zeropoint behavior.
- `photometry.ephemeris_provider`
  Optional asteroid ephemeris service selection: `jpl` or `miriade`.
- `photometry.export_reference_star_timeseries`
  Optional export of calibrated per-frame photometry for valid reference stars used in calibration.
- `photometry.ransac_threshold_mode`
  `fixed` uses `photometry.ransac_threshold` for every frame; `auto` selects one threshold from the first valid frame and reuses it for the sequence.
- `photometry.zp_error_method`
  `bootstrap` estimates the zero-point model uncertainty from the final RANSAC inlier reference-star set.
- `paths`
  Input directory, working directories, output filenames, and optional plot/cutout folders.
- `plots`
  Enables or disables diagnostic plotting.

## Mode Selection
- `photometry.mode: asteroid`
  Requires `photometry.target_id`
- `photometry.mode: star`
  Requires `photometry.coords` and `photometry.coords_unit`

## Ephemeris Provider
- `photometry.ephemeris_provider: jpl`
  Uses JPL Horizons to resolve the configured asteroid target. This is the default.
- `photometry.ephemeris_provider: miriade`
  Uses IMCCE Miriade ephemcc to resolve the configured asteroid target. This uses the configured `observatory.observatory_code` and is intended as a fast alternative for ordinary Solar System object target photometry.

Example:

```yaml
photometry:
  mode: "asteroid"
  target_id: "19184"
  ephemeris_provider: "miriade"
```

SkyBot cone-search is a separate IMCCE service for identifying all Solar System objects inside a field of view. It is not used for configured target ephemerides.

## Target Coordinate Units
- `photometry.coords_unit: deg`
  Treats both coordinate values as degrees.
- `photometry.coords_unit: hourangle_deg`
  Treats the first value as RA in hour angle and the second value as Dec in degrees.

Examples:

```yaml
photometry:
  mode: "star"
  coords: [119.31868375, 35.78217389]
  coords_unit: "deg"
```

```yaml
photometry:
  mode: "star"
  coords: ["07 57 16.4841", "+35 46 55.826"]
  coords_unit: "hourangle_deg"
```

For occultation work, `star` mode is usually the better default. In many campaigns the exposure time is tuned to the target star, and the occulting body is too faint to measure reliably in individual frames even for positive events.

## Photometric Calibration Uncertainty
PhoPS supports fixed and automatic RANSAC thresholds for the radial zero-point fit. In fixed mode, `photometry.ransac_threshold` is used for every image. In auto mode, PhoPS evaluates `photometry.ransac_threshold_grid` on the first valid image, counts RANSAC inliers at each threshold, monotonises the inlier-count curve, and selects the knee using `photometry.ransac_auto_method: "inlier_knee"`. The selected threshold is reused for subsequent images when `photometry.ransac_auto_reuse_for_sequence` is true.

The zero-point uncertainty is estimated with bootstrap resampling when `photometry.zp_error_method: "bootstrap"`. For each image, PhoPS resamples the final RANSAC inlier reference stars, refits the radial zero-point model, and evaluates the bootstrap models at each measured object radius. The reported total uncertainty is written to the final `mag_err` column:

```text
mag_err = sqrt(sigma_formal^2 + sigma_ZP_boot^2)
```

The formal aperture-photometry uncertainty remains available internally for this calculation, and `snr` remains based on the photometric measurement. The separate `sigma_total` column is not written to final photometry tables. `zp_scatter` is a diagnostic residual scatter of the final zero-point fit and is not added directly to the reported uncertainty.

The RANSAC threshold can be selected automatically from the first valid image of a sequence by analysing the saturation behaviour of the inlier count as a function of threshold. The threshold corresponding to the knee of the monotonised inlier-count curve is adopted and reused for all subsequent images. For each image, the final inlier reference-star sample is bootstrapped to estimate the zero-point uncertainty at the position of each measured source. The final total uncertainty is the quadratic sum of the formal aperture-photometry error and the bootstrap zero-point uncertainty.

Relevant options:

```yaml
photometry:
  ransac_threshold_mode: "auto"
  ransac_threshold: 0.10
  ransac_threshold_grid:
    start: 0.01
    stop: 0.20
    step: 0.01
  ransac_auto_method: "inlier_knee"
  ransac_auto_min_inliers: 30
  ransac_auto_fallback_threshold: 0.10
  ransac_auto_reuse_for_sequence: true
  zp_error_method: "bootstrap"
  zp_bootstrap_iterations: 1000
  zp_bootstrap_random_seed: 42
  zp_bootstrap_min_inliers: 30
  write_analysis_summary: true
  analysis_summary_filename: "phops_analysis_summary.yaml"
```

## Astrometry Solve Mode
- `astrometry.solve_mode: solve`
  Default mode. PhoPS runs `solve-field` and produces a solved FITS file.
- `astrometry.solve_mode: existing_wcs`
  PhoPS skips `solve-field` and uses the input FITS directly. This requires a real celestial WCS in the FITS header, not only approximate center coordinates such as `CRVAL1` and `CRVAL2`.

## Path Behavior
- Relative paths are resolved against the config file location.
- Only runtime directories are created automatically.
- Output CSV files are always written inside `paths.solve_dir`.
- `paths.output_reference_star_timeseries` controls the optional reference-star CSV filename.
- `photometry.analysis_summary_filename` controls the optional calibration summary filename inside `paths.solve_dir`.
- `paths.file_extension` can be a simple suffix like `fits` or a glob pattern like `*_flc.fits`.
- When `paths.file_extension` is set to a FITS-family value (`fits`, `fit`, `fts`, and `.gz` variants), PhoPS matches all common FITS filename variants automatically.
- PhoPS also keeps a hidden `.phops-run-state.json` checkpoint inside `paths.solve_dir` so interrupted runs can be resumed safely.

## Light-Curve Plotting
- `plots.plot_light_curve: true`
  Generates `light_curve.png` and optionally `light_curve.pdf` from `photometry.csv`.
- `plots.light_curve_x_axis: relative_seconds`
  Uses seconds relative to the first measurement on the x-axis.
- `plots.light_curve_x_axis: jd`
  Uses the Julian Date directly on the x-axis.
- `plots.light_curve_aux_panels: true`
  Adds quality-control panels for `snr`, `fwhm`, and `zp_scatter` when those columns exist.
- `plots.light_curve_pdf: true`
  Writes a PDF companion file next to the PNG output.
- `plots.light_curve_event_window`
  Optional two-value sequence used for shaded event-window highlighting.
- `plots.light_curve_event_unit: relative_seconds`
  Interprets the event-window values as seconds relative to the first frame.
- `plots.light_curve_event_unit: jd`
  Interprets the event-window values as Julian Dates.

The generated light-curve figure uses a publication-oriented scientific style by default: boxed axes, inward ticks, serif typography, clipped magnitude limits, error bars, and auxiliary quality panels when available.
