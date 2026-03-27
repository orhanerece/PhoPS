# PhoPS

<p align="center">
  <img src="docs/assets/phops-logo.png" alt="PhoPS logo" width="320">
</p>

PhoPS is a Python package for point-source photometry and astrometry on FITS images. The repository is now structured as an installable package with a shared service layer, a CLI, and a desktop UI adapter layer that can support a future Qt frontend.

## What It Provides
- Astrometric solving through `astrometry.net`
- Gaia-based reference patch creation
- Field-star photometric calibration
- Target photometry for asteroid or fixed-star mode
- CSV outputs and diagnostic plots
- Publication-style scientific light-curve plots from `photometry.csv`
- Shared core usable from terminal or desktop UI

## External Requirements
PhoPS depends on Python packages and a few system tools:

- Python 3.10+
- `solve-field`
- `hpsplit`
- `build-astrometry-index`
- Internet access for Gaia and JPL Horizons queries during runtime

Those three binaries come from astrometry.net and must be available on `PATH`.

## Installation
```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start
Create a config file:

```bash
phops init-config config.yaml
```

Validate it:

```bash
phops validate-config -c config.yaml
```

Run the full pipeline:

```bash
phops run -c config.yaml
```

If PhoPS detects an earlier interrupted or completed run in the same output folder, it asks whether you want to `resume`, `restart`, or `cancel`. By default, the terminal runner first shows a compact run summary with the key input and output paths, then starts a two-layer live progress view with the current frame, stage, percentage, running counts, current action, and last warning. Use `-v` if you want plain detailed logs instead.

Regenerate the photometry light-curve plot from an existing CSV:

```bash
phops plot-photometry -c config.yaml
```

Launch the current desktop runner:

```bash
phops gui -c config.yaml
```

## Short Usage Notes
- Put your FITS files into the folder defined by `paths.input_dir`.
- Set `photometry.mode` to `asteroid` or `star`.
- In `asteroid` mode, set `photometry.target_id`.
- In `star` mode, set `photometry.coords` and declare `photometry.coords_unit`.
- `photometry.coords_unit: deg` means both values are degrees.
- `photometry.coords_unit: hourangle_deg` means RA is hour angle and Dec is degrees.
- Use `astrometry.solve_mode: existing_wcs` only when the FITS files already contain a full celestial WCS.
- For occultation datasets, `photometry.mode: star` is usually the correct choice because exposure time is set for the target star and the asteroid is often too faint for reliable frame-by-frame photometry.
- Outputs are written under `paths.solve_dir`, with plots and cutouts in subdirectories.
- Resume mode keeps a small checkpoint file inside `paths.solve_dir` so already measured frames can be skipped cleanly after an interruption.

## Commands
- `phops run -c config.yaml`
  Runs the full astrometry + photometry pipeline.
- `phops run -c config.yaml --resume`
  Resumes from existing outputs without reprocessing frames already measured.
- `phops run -c config.yaml --restart`
  Clears the current run outputs and starts from scratch without prompting.
- `phops validate-config -c config.yaml`
  Parses and validates the YAML config without processing data.
- `phops init-config path/to/config.yaml`
  Writes a ready-to-edit example config.
- `phops gui -c config.yaml`
  Opens the current reference desktop runner. The service layer is kept separate so this can be replaced by a Qt frontend later.
- `phops plot-photometry -c config.yaml`
  Renders `light_curve.png` and, if enabled, `light_curve.pdf` from the current photometry table.

## Project Layout
```text
src/phops/           installable package
tests/               automated tests
docs/                project documentation
examples/            example config
config.yaml          editable local config
```

## Outputs
- `photometry.csv`: calibrated target photometry
- `astrometry.csv`: matched-source residuals
- `output/plots/`: zeropoint and astrometry plots
- `output/plots/light_curve.png`: calibrated light curve
- `output/plots/light_curve.pdf`: optional publication-ready vector export
- `output/cutouts/`: measured target cutouts
- `output/.phops-run-state.json`: hidden resume checkpoint used by the CLI

See [docs/configuration.md](docs/configuration.md), [docs/outputs.md](docs/outputs.md), and [docs/architecture.md](docs/architecture.md) for details.

## License
PhoPS is licensed under GNU GPL v3. See `LICENSE`.
