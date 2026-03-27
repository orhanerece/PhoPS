# Architecture

<p align="center">
  <img src="assets/phops-logo.png" alt="PhoPS logo" width="180">
</p>

PhoPS is split into reusable layers:

## Core Service Layer
- `src/phops/pipeline.py`
  Orchestrates one full run across all FITS files.
- `src/phops/astrometry.py`
  Handles Gaia patch generation and astrometry.net integration.
- `src/phops/photometry.py`
  Handles source matching, calibration, and target measurement.
- `src/phops/target.py`
  Resolves asteroid or fixed-star coordinates.

## Interface Layer
- `src/phops/cli.py`
  Terminal entry point.
- `src/phops/gui.py`
  Reference desktop adapter. It currently uses Tkinter only as a lightweight example; the pipeline and reporting layers are intentionally structured so a future Qt frontend can replace it without touching the core services.
- `src/phops/reporting.py`
  Shared progress/event channel used by both.

## Why This Shape
- The pipeline can be called from CLI, GUI, or future APIs without duplicating logic.
- Progress reporting is structured, so GUI or TUI frontends can react to the same events.
- Config loading is validated once and then passed around as typed objects.
- Plotting and filesystem writes are isolated enough to replace later if a richer UI is added.
