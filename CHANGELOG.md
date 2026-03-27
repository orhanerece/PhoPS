# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [0.1.0] - 2026-03-26
### Added
- Modern `src/` package layout with installable `phops` package.
- Shared service layer for pipeline execution, CLI, and GUI use.
- YAML configuration loader with validation and path normalization.
- CLI commands for `run`, `validate-config`, `init-config`, and `gui`.
- Reference desktop UI adapter backed by the same pipeline service and structured for a future Qt frontend.
- `astrometry.solve_mode` with `existing_wcs` support for pre-solved FITS workflows.
- Pre-run CLI summary for the main input/output paths, followed by a richer two-layer terminal progress view with clearer colors, status badges, current action, and last-warning context.
- Scientific photometry light-curve plotting with error bars, auxiliary quality panels, optional event-window shading, and a dedicated `plot-photometry` CLI command.
- Publication-oriented light-curve styling with a more scientific figure layout, clipped magnitude range, and cleaner paper-ready exports.
- Interactive `resume / restart / cancel` handling for existing run outputs, backed by a hidden run-state checkpoint.
- Structured reporting, error types, and better external dependency checks.
- Documentation set covering setup, architecture, configuration, and outputs.
- Example configuration, CI workflow, and automated tests.

### Changed
- Reworked pipeline orchestration to process every frame instead of stopping after the first successful measurement.
- Fixed several runtime edge cases around missing detections, zeropoint fallback, path handling, and observation time parsing.
- Removed hardcoded plotting assumptions so plots depend on image shape rather than fixed sensor dimensions.
- Switched the project license to GNU GPL v3.
