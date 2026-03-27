"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from astropy.coordinates import SkyCoord
from astropy import units as u
import yaml

from .errors import ConfigurationError

PlotScale = Literal["pixel", "wcs"]
LightCurveXAxis = Literal["relative_seconds", "jd"]
PhotometryMode = Literal["asteroid", "star"]
ApertureMethod = Literal["fixed_pixel", "fixed_arcsec", "fwhm_factor"]
ZeroPointMode = Literal["fit", "average"]
AstrometryMode = Literal["solve", "existing_wcs"]
CoordinateUnit = Literal["deg", "hourangle_deg"]


def _require_mapping(section: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"'{section}' must be a mapping in the YAML configuration.")
    return value


def _resolve_path(base_dir: Path, raw_value: str | Path) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_coords(raw_coords: Any, coords_unit: str) -> tuple[float, float] | None:
    if raw_coords in (None, "", []):
        return None
    if coords_unit not in {"deg", "hourangle_deg"}:
        raise ConfigurationError("'photometry.coords_unit' must be either 'deg' or 'hourangle_deg'.")

    unit = (u.deg, u.deg) if coords_unit == "deg" else (u.hourangle, u.deg)

    try:
        if isinstance(raw_coords, str):
            coord = SkyCoord(raw_coords, unit=unit, frame="icrs")
            return float(coord.ra.deg), float(coord.dec.deg)

        if isinstance(raw_coords, (list, tuple)) and len(raw_coords) == 2:
            coord = SkyCoord(raw_coords[0], raw_coords[1], unit=unit, frame="icrs")
            return float(coord.ra.deg), float(coord.dec.deg)
        raise ConfigurationError(
            "'photometry.coords' must be either a two-item sequence or a single coordinate string."
        )
    except ValueError as exc:
        raise ConfigurationError(
            "Could not parse 'photometry.coords' using the declared 'photometry.coords_unit'."
        ) from exc


def _parse_optional_float_pair(section: str, value: Any) -> tuple[float, float] | None:
    if value in (None, "", []):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigurationError(f"'{section}' must be a two-item sequence when provided.")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{section}' must contain numeric values.") from exc


@dataclass
class FitsKeywordsConfig:
    ra_key: str = "RA"
    dec_key: str = "DEC"
    date_key: str = "DATE-OBS"
    exposure_key: str = "EXPTIME"
    jd_key: str = "JD"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "FitsKeywordsConfig":
        return cls(
            ra_key=str(mapping.get("ra_key", "RA")),
            dec_key=str(mapping.get("dec_key", "DEC")),
            date_key=str(mapping.get("date_key", "DATE-OBS")),
            exposure_key=str(mapping.get("exposure_key", "EXPTIME")),
            jd_key=str(mapping.get("jd_key", "JD")),
        )


@dataclass
class AstrometryConfig:
    solve_mode: AstrometryMode = "solve"
    radius: float = 0.5
    quad_scales: list[int] = field(default_factory=lambda: [0, 2, 4, 6])
    cache_tolerance: float = 0.1
    epoch_bucket_precision: int = 1

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "AstrometryConfig":
        raw_scales = mapping.get("quad_scales", [0, 2, 4, 6])
        return cls(
            solve_mode=str(mapping.get("solve_mode", "solve")),
            radius=float(mapping.get("radius", 0.5)),
            quad_scales=[int(item) for item in raw_scales],
            cache_tolerance=float(mapping.get("cache_tolerance", 0.1)),
            epoch_bucket_precision=int(mapping.get("epoch_bucket_precision", 1)),
        )

    def validate(self) -> None:
        if self.solve_mode not in {"solve", "existing_wcs"}:
            raise ConfigurationError("'astrometry.solve_mode' must be either 'solve' or 'existing_wcs'.")


@dataclass
class InstrumentConfig:
    pixel_scale: float
    gain: float
    read_noise: float
    saturation_level: float = 60000.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "InstrumentConfig":
        try:
            return cls(
                pixel_scale=float(mapping["pixel_scale"]),
                gain=float(mapping.get("gain", 1.0)),
                read_noise=float(mapping.get("read_noise", 5.0)),
                saturation_level=float(mapping.get("saturation_level", 60000.0)),
            )
        except KeyError as exc:
            raise ConfigurationError("'instrument.pixel_scale' is required.") from exc


@dataclass
class ObservatoryConfig:
    observatory_code: str = "500"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ObservatoryConfig":
        return cls(observatory_code=str(mapping.get("observatory_code", "500")))


@dataclass
class SourceDetectionConfig:
    fwhm_guess: float = 5.0
    threshold_sigma: float = 3.0
    min_area: int = 5
    edge_margin: int = 10

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SourceDetectionConfig":
        return cls(
            fwhm_guess=float(mapping.get("fwhm_guess", 5.0)),
            threshold_sigma=float(mapping.get("threshold_sigma", 3.0)),
            min_area=int(mapping.get("min_area", 5)),
            edge_margin=int(mapping.get("edge_margin", 10)),
        )


@dataclass
class MatchingConfig:
    isolation_radius_arcsec: float = 0.2
    match_constraint_arcsec: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "MatchingConfig":
        return cls(
            isolation_radius_arcsec=float(mapping.get("isolation_radius_arcsec", 0.2)),
            match_constraint_arcsec=float(mapping.get("match_constraint_arcsec", 1.0)),
        )


@dataclass
class PhotometryConfig:
    mode: PhotometryMode = "asteroid"
    target_id: str | None = None
    coords: tuple[float, float] | None = None
    coords_unit: CoordinateUnit = "deg"
    filter: str = "R"
    aperture_method: ApertureMethod = "fixed_arcsec"
    aperture: float = 4.0
    annulus_inner: float = 7.0
    annulus_outer: float = 9.0
    zeropoint: ZeroPointMode = "fit"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "PhotometryConfig":
        coords_unit = str(mapping.get("coords_unit", "deg"))
        return cls(
            mode=str(mapping.get("mode", "asteroid")),
            target_id=None if mapping.get("target_id") in (None, "") else str(mapping.get("target_id")),
            coords=_parse_coords(mapping.get("coords"), coords_unit),
            coords_unit=coords_unit,
            filter=str(mapping.get("filter", "R")),
            aperture_method=str(mapping.get("aperture_method", "fixed_arcsec")),
            aperture=float(mapping.get("aperture", 4.0)),
            annulus_inner=float(mapping.get("annulus_inner", 7.0)),
            annulus_outer=float(mapping.get("annulus_outer", 9.0)),
            zeropoint=str(mapping.get("zeropoint", "fit")),
        )

    def validate(self) -> None:
        if self.mode not in {"asteroid", "star"}:
            raise ConfigurationError("'photometry.mode' must be either 'asteroid' or 'star'.")
        if self.coords_unit not in {"deg", "hourangle_deg"}:
            raise ConfigurationError("'photometry.coords_unit' must be either 'deg' or 'hourangle_deg'.")
        if self.aperture_method not in {"fixed_pixel", "fixed_arcsec", "fwhm_factor"}:
            raise ConfigurationError(
                "'photometry.aperture_method' must be one of: fixed_pixel, fixed_arcsec, fwhm_factor."
            )
        if self.zeropoint not in {"fit", "average"}:
            raise ConfigurationError("'photometry.zeropoint' must be either 'fit' or 'average'.")
        if self.mode == "asteroid" and not self.target_id:
            raise ConfigurationError("'photometry.target_id' is required when mode is 'asteroid'.")
        if self.mode == "star" and self.coords is None:
            raise ConfigurationError("'photometry.coords' is required when mode is 'star'.")


@dataclass
class PathsConfig:
    base_dir: Path
    input_dir: Path
    temp_dir: Path
    index_dir: Path
    solve_dir: Path
    file_extension: str = "fits"
    output_photometry: str = "photometry.csv"
    output_astrometry: str = "astrometry.csv"
    plot_dir: Path | None = None
    cutout_dir: Path | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any], base_dir: Path) -> "PathsConfig":
        solve_dir = _resolve_path(base_dir, mapping.get("solve_dir", "output"))
        plot_dir_value = mapping.get("plot_dir")
        cutout_dir_value = mapping.get("cutout_dir")
        return cls(
            base_dir=base_dir,
            input_dir=_resolve_path(base_dir, mapping.get("input_dir", "input")),
            temp_dir=_resolve_path(base_dir, mapping.get("temp_dir", "data/temp")),
            index_dir=_resolve_path(base_dir, mapping.get("index_dir", "data/indexes")),
            solve_dir=solve_dir,
            file_extension=str(mapping.get("file_extension", "fits")).lstrip("."),
            output_photometry=str(mapping.get("output_photometry", "photometry.csv")),
            output_astrometry=str(mapping.get("output_astrometry", "astrometry.csv")),
            plot_dir=_resolve_path(base_dir, plot_dir_value) if plot_dir_value else solve_dir / "plots",
            cutout_dir=_resolve_path(base_dir, cutout_dir_value) if cutout_dir_value else solve_dir / "cutouts",
        )

    @property
    def photometry_csv_path(self) -> Path:
        return self.solve_dir / self.output_photometry

    @property
    def astrometry_csv_path(self) -> Path:
        return self.solve_dir / self.output_astrometry

    @property
    def run_state_path(self) -> Path:
        return self.solve_dir / ".phops-run-state.json"

    def ensure_runtime_dirs(self) -> None:
        for path in (self.temp_dir, self.index_dir, self.solve_dir, self.plot_dir, self.cutout_dir):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class PlotsConfig:
    plot_astrometry: bool = True
    plot_image: bool = True
    plot_light_curve: bool = True
    image_scale: PlotScale = "pixel"
    light_curve_x_axis: LightCurveXAxis = "relative_seconds"
    light_curve_aux_panels: bool = True
    light_curve_pdf: bool = True
    light_curve_event_window: tuple[float, float] | None = None
    light_curve_event_unit: LightCurveXAxis = "relative_seconds"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "PlotsConfig":
        return cls(
            plot_astrometry=bool(mapping.get("plot_astrometry", True)),
            plot_image=bool(mapping.get("plot_image", True)),
            plot_light_curve=bool(mapping.get("plot_light_curve", True)),
            image_scale=str(mapping.get("image_scale", "pixel")),
            light_curve_x_axis=str(mapping.get("light_curve_x_axis", "relative_seconds")),
            light_curve_aux_panels=bool(mapping.get("light_curve_aux_panels", True)),
            light_curve_pdf=bool(mapping.get("light_curve_pdf", True)),
            light_curve_event_window=_parse_optional_float_pair(
                "plots.light_curve_event_window",
                mapping.get("light_curve_event_window"),
            ),
            light_curve_event_unit=str(mapping.get("light_curve_event_unit", "relative_seconds")),
        )

    def validate(self) -> None:
        if self.image_scale not in {"pixel", "wcs"}:
            raise ConfigurationError("'plots.image_scale' must be either 'pixel' or 'wcs'.")
        if self.light_curve_x_axis not in {"relative_seconds", "jd"}:
            raise ConfigurationError("'plots.light_curve_x_axis' must be either 'relative_seconds' or 'jd'.")
        if self.light_curve_event_unit not in {"relative_seconds", "jd"}:
            raise ConfigurationError("'plots.light_curve_event_unit' must be either 'relative_seconds' or 'jd'.")
        if self.light_curve_event_window is not None and self.light_curve_event_window[0] >= self.light_curve_event_window[1]:
            raise ConfigurationError("'plots.light_curve_event_window' must be in ascending order.")


@dataclass
class AppConfig:
    source_path: Path
    fits_keywords: FitsKeywordsConfig
    astrometry: AstrometryConfig
    instrument: InstrumentConfig
    observatory: ObservatoryConfig
    source_detection: SourceDetectionConfig
    matching: MatchingConfig
    photometry: PhotometryConfig
    paths: PathsConfig
    plots: PlotsConfig
    catalog: str = "gaiadr3.gaia_source"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any], source_path: Path) -> "AppConfig":
        base_dir = source_path.parent.resolve()
        config = cls(
            source_path=source_path.resolve(),
            fits_keywords=FitsKeywordsConfig.from_mapping(_require_mapping("fits_keywords", mapping.get("fits_keywords"))),
            astrometry=AstrometryConfig.from_mapping(_require_mapping("astrometry", mapping.get("astrometry"))),
            instrument=InstrumentConfig.from_mapping(_require_mapping("instrument", mapping.get("instrument"))),
            observatory=ObservatoryConfig.from_mapping(_require_mapping("observatory", mapping.get("observatory"))),
            source_detection=SourceDetectionConfig.from_mapping(
                _require_mapping("source_detection", mapping.get("source_detection"))
            ),
            matching=MatchingConfig.from_mapping(_require_mapping("matching", mapping.get("matching"))),
            photometry=PhotometryConfig.from_mapping(_require_mapping("photometry", mapping.get("photometry"))),
            paths=PathsConfig.from_mapping(_require_mapping("paths", mapping.get("paths")), base_dir=base_dir),
            plots=PlotsConfig.from_mapping(_require_mapping("plots", mapping.get("plots"))),
            catalog=str(mapping.get("catalog", "gaiadr3.gaia_source")),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise ConfigurationError(f"Configuration file not found: {source_path}")
        with source_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
        if not isinstance(raw_config, dict):
            raise ConfigurationError("The configuration file must contain a top-level mapping.")
        return cls.from_mapping(raw_config, source_path=source_path)

    def validate(self) -> None:
        self.astrometry.validate()
        self.photometry.validate()
        self.plots.validate()
        if self.instrument.pixel_scale <= 0:
            raise ConfigurationError("'instrument.pixel_scale' must be positive.")
        if self.source_detection.threshold_sigma <= 0:
            raise ConfigurationError("'source_detection.threshold_sigma' must be positive.")
        if not self.paths.file_extension:
            raise ConfigurationError("'paths.file_extension' cannot be empty.")

    def ensure_runtime_dirs(self) -> None:
        self.paths.ensure_runtime_dirs()


def load_config(path: str | Path) -> AppConfig:
    """Convenience helper for callers."""

    return AppConfig.from_yaml(path)
