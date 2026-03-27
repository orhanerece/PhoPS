"""Utility helpers shared across the package."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

from .errors import ConfigurationError

if TYPE_CHECKING:
    from .config import FitsKeywordsConfig


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


def _parse_time_value(raw_value: object, *, prefer_jd: bool = False) -> Time:
    if raw_value is None:
        raise ConfigurationError("Observation time could not be read from the FITS header.")
    if isinstance(raw_value, (int, float)):
        return Time(float(raw_value), format="jd", scale="utc")
    text = str(raw_value).strip()
    try:
        numeric = float(text)
    except ValueError:
        pass
    else:
        return Time(numeric, format="jd" if prefer_jd else "jd", scale="utc")

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

    jd_raw = header.get(fits_keywords.jd_key)
    if jd_raw is not None:
        time_value = _parse_time_value(jd_raw, prefer_jd=True)
    else:
        raw_time = header.get(fits_keywords.date_key)
        time_value = _parse_time_value(raw_time)

    exposure_seconds = float(header.get(fits_keywords.exposure_key, 0.0) or 0.0)
    return time_value + TimeDelta(exposure_seconds / 2.0, format="sec")


def observation_jd_from_header(header, fits_keywords: FitsKeywordsConfig) -> float:
    """Return the mid-exposure Julian Date from a FITS header."""

    return float(observation_time_from_header(header, fits_keywords).jd)


def append_rows_to_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Append structured rows to a CSV file."""

    row_list = list(rows)
    if not row_list:
        return
    frame = pd.DataFrame(row_list)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def iter_input_files(input_dir: Path, extension: str) -> list[Path]:
    """Return sorted input files for the configured extension."""

    pattern = f"*.{extension.lstrip('.')}"
    return sorted(input_dir.glob(pattern))


def safe_stem(path: str | Path) -> str:
    """Produce a filesystem friendly stem."""

    stem = Path(path).stem
    return stem.replace(".", "_").replace(" ", "_")
