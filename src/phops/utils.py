"""Utility helpers shared across the package."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from .errors import ConfigurationError

if TYPE_CHECKING:
    from .config import FitsKeywordsConfig


FITS_FAMILY_PATTERNS = ("*.fits", "*.fit", "*.fts", "*.fits.gz", "*.fit.gz", "*.fts.gz")


def ra_to_deg(ra_value: str | float | int) -> float:
    """Convert RA in HH:MM:SS or decimal degrees into decimal degrees."""

    if isinstance(ra_value, (float, int)):
        return float(ra_value)
    parts = [float(item) for item in str(ra_value).strip().split(":")]
    if len(parts) != 3:
        raise ValueError(f"Unsupported RA value: {ra_value}")
    hours, minutes, seconds = parts
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def dec_to_deg(dec_value: str | float | int) -> float:
    """Convert Dec in DD:MM:SS or decimal degrees into decimal degrees."""

    if isinstance(dec_value, (float, int)):
        return float(dec_value)
    parts = str(dec_value).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported Dec value: {dec_value}")
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    sign = -1 if str(parts[0]).startswith("-") else 1
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def calculate_residuals(matched_stars) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate astrometric residuals in arcseconds."""

    c_img = SkyCoord(ra=matched_stars["img_ra"] * u.deg, dec=matched_stars["img_dec"] * u.deg)
    c_ref = SkyCoord(ra=matched_stars["gaia_ra"] * u.deg, dec=matched_stars["gaia_dec"] * u.deg)
    dra, ddec = c_ref.spherical_offsets_to(c_img)
    dra_arcsec = dra.to(u.arcsec).value
    ddec_arcsec = ddec.to(u.arcsec).value
    gmag = np.asarray(matched_stars["gaia_gmag"], dtype=float)
    mask = np.isfinite(dra_arcsec) & np.isfinite(ddec_arcsec) & np.isfinite(gmag)
    return dra_arcsec[mask], ddec_arcsec[mask], gmag[mask]


def _parse_time_value(raw_value: object, *, numeric_format: str = "jd") -> Time:
    if raw_value is None:
        raise ConfigurationError("Observation time could not be read from the FITS header.")
    if isinstance(raw_value, (int, float)):
        return Time(float(raw_value), format=numeric_format, scale="utc")
    text = str(raw_value).strip()
    try:
        numeric = float(text)
    except ValueError:
        pass
    else:
        return Time(numeric, format=numeric_format, scale="utc")

    for fmt in ("isot", "iso", None):
        try:
            if fmt is None:
                return Time(text, scale="utc")
            return Time(text, format=fmt, scale="utc")
        except ValueError:
            continue
    raise ConfigurationError(f"Unsupported observation time value: {raw_value}")


def observation_time_from_header(header, fits_keywords: FitsKeywordsConfig) -> Time:
    """Return the mid-exposure observation time from a FITS header."""

    jd_raw = header.get(fits_keywords.jd_key) if fits_keywords.jd_key else None
    if jd_raw is not None:
        time_value = _parse_time_value(jd_raw, numeric_format="jd")
    else:
        mjd_raw = header.get(fits_keywords.mjd_key) if fits_keywords.mjd_key else None
        if mjd_raw is not None:
            time_value = _parse_time_value(mjd_raw, numeric_format="mjd")
        else:
            raw_time = header.get(fits_keywords.date_key)
            if fits_keywords.time_key:
                raw_clock = header.get(fits_keywords.time_key)
                if raw_time is not None and raw_clock is not None:
                    date_text = str(raw_time).strip()
                    time_text = str(raw_clock).strip()
                    if "T" not in date_text and " " not in date_text:
                        raw_time = f"{date_text}T{time_text}"
                    elif time_text not in date_text:
                        raw_time = f"{date_text} {time_text}"
            time_value = _parse_time_value(raw_time)

    exposure_seconds = float(header.get(fits_keywords.exposure_key, 0.0) or 0.0)
    return time_value + TimeDelta(exposure_seconds / 2.0, format="sec")


def observation_jd_from_header(header, fits_keywords: FitsKeywordsConfig) -> float:
    """Return the mid-exposure Julian Date from a FITS header."""

    return float(observation_time_from_header(header, fits_keywords).jd)


def _merged_image_header(hdul: fits.HDUList, index: int) -> fits.Header:
    header = hdul[0].header.copy()
    if index == 0:
        return header
    header.extend(hdul[index].header, strip=True, update=True)
    return header


def load_fits_image(image_path: str | Path) -> tuple[np.ndarray, fits.Header, WCS]:
    """Load the best science image extension, merged header, and WCS."""

    image_path = Path(image_path)
    with fits.open(image_path) as hdul:
        candidates: list[tuple[bool, bool, int, np.ndarray, fits.Header, WCS]] = []
        for index, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None or getattr(data, "ndim", 0) != 2:
                continue
            header = _merged_image_header(hdul, index)
            try:
                wcs = WCS(header, fobj=hdul)
            except Exception:
                wcs = WCS(header)
            extname = str(hdu.header.get("EXTNAME", "")).upper()
            is_science_hdu = extname in {"", "PRIMARY", "SCI", "IMAGE"}
            candidates.append((bool(wcs.has_celestial), is_science_hdu, index, data.astype(float), header, wcs))

        if not candidates:
            raise ConfigurationError(f"No 2-D image extension could be read from FITS file: {image_path}")

        candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        _, _, _, data, header, wcs = candidates[0]
        return data, header, wcs


def append_rows_to_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Append structured rows to a CSV file."""

    row_list = list(rows)
    if not row_list:
        return
    frame = pd.DataFrame(row_list)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def input_file_patterns(extension: str) -> tuple[str, ...]:
    """Return glob-style filename patterns for the configured input selector."""

    normalized = extension.strip()
    if not normalized:
        return tuple()
    if any(char in normalized for char in "*?[]"):
        return (normalized,)
    normalized = normalized.lstrip(".").lower()
    if normalized in {"fits", "fit", "fts", "fits.gz", "fit.gz", "fts.gz"}:
        return FITS_FAMILY_PATTERNS
    return (f"*.{normalized}",)


def describe_input_selector(extension: str) -> str:
    """Return a compact human-readable description of the input selector."""

    patterns = input_file_patterns(extension)
    if not patterns:
        return "<none>"
    if len(patterns) == 1:
        return patterns[0]
    return ", ".join(patterns)


def iter_input_files(input_dir: Path, extension: str) -> list[Path]:
    """Return sorted input files for the configured extension."""

    patterns = tuple(pattern.lower() for pattern in input_file_patterns(extension))
    if not patterns or not input_dir.exists():
        return []
    matched_paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in patterns)
    ]
    return sorted(matched_paths)


def safe_stem(path: str | Path) -> str:
    """Produce a filesystem friendly stem."""

    stem = Path(path).stem
    return stem.replace(".", "_").replace(" ", "_")
