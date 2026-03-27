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
- `paths`
  Input directory, working directories, output filenames, and optional plot/cutout folders.
- `plots`
  Enables or disables diagnostic plotting.

## Mode Selection
- `photometry.mode: asteroid`
  Requires `photometry.target_id`
- `photometry.mode: star`
  Requires `photometry.coords` and `photometry.coords_unit`

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

## Astrometry Solve Mode
- `astrometry.solve_mode: solve`
  Default mode. PhoPS runs `solve-field` and produces a solved FITS file.
- `astrometry.solve_mode: existing_wcs`
  PhoPS skips `solve-field` and uses the input FITS directly. This requires a real celestial WCS in the FITS header, not only approximate center coordinates such as `CRVAL1` and `CRVAL2`.

## Path Behavior
- Relative paths are resolved against the config file location.
- Only runtime directories are created automatically.
- Output CSV files are always written inside `paths.solve_dir`.
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
