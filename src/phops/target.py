"""Target coordinate resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from astroquery.jplhorizons import Horizons

from .config import AppConfig
from .errors import TargetResolutionError
from .reporting import NullReporter, ProgressReporter, report
from .utils import dec_to_deg, observation_jd_from_header, ra_to_deg

MIRIADE_EPHEMCC_URL = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"
MIRIADE_TIMEOUT_SECONDS = 30


def _optional_float(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _ra_degrees(record: dict[str, Any]) -> float:
    value = _optional_float(record, ("RA", "RA (deg)", "_raj2000"))
    if value is not None:
        return value

    hour_value = record.get("RA") or record.get("RA (hour)")
    if hour_value in (None, ""):
        raise TargetResolutionError("Miriade response did not include a valid RA.")
    return ra_to_deg(":".join(str(hour_value).strip().split()))


def _dec_degrees(record: dict[str, Any]) -> float:
    value = _optional_float(record, ("DEC", "DEC (deg)", "_decj2000"))
    if value is not None:
        return value

    dec_value = record.get("DEC")
    if dec_value in (None, ""):
        raise TargetResolutionError("Miriade response did not include a valid Dec.")
    return dec_to_deg(":".join(str(dec_value).strip().split()))


def _miriade_data_record(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TargetResolutionError("Miriade ephemcc returned an unexpected response format.")

    flag = payload.get("flag")
    if flag in (0, -1, "0", "-1"):
        status = payload.get("status", "unknown error")
        raise TargetResolutionError(f"Miriade ephemcc returned an error: {status}")

    data = payload.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    raise TargetResolutionError("Miriade ephemcc did not return target ephemeris data.")


def _miriade_target_name(target_id: str) -> str:
    prefix, separator, _ = target_id.partition(":")
    if separator and prefix.isalpha():
        return target_id
    return f"a:{target_id}"


@dataclass
class TargetInfo:
    """Resolved target information."""

    ra: float
    dec: float
    jd: float
    r: float | None = None
    delta: float | None = None
    alpha: float | None = None


class TargetManager:
    """Resolve target coordinates for star and asteroid modes."""

    def __init__(self, config: AppConfig, reporter: ProgressReporter | None = None) -> None:
        self.config = config
        self.reporter = reporter or NullReporter()

    def get_jd_time(self, header) -> float:
        """Return the mid-exposure Julian Date."""

        return observation_jd_from_header(header, self.config.fits_keywords)

    def get_target_coordinates(self, header) -> tuple[float, float, dict[str, float] | None]:
        """Compatibility method returning coordinates and physical metadata."""

        info = self.resolve(header)
        physical = None
        if info.r is not None and info.delta is not None and info.alpha is not None:
            physical = {"r": info.r, "delta": info.delta, "alpha": info.alpha, "jd": info.jd}
        return info.ra, info.dec, physical

    def resolve(self, header) -> TargetInfo:
        """Resolve target coordinates using configuration and FITS header metadata."""

        mode = self.config.photometry.mode
        jd_value = self.get_jd_time(header)
        if mode == "star":
            if self.config.photometry.coords is None:
                raise TargetResolutionError("Star mode requires 'photometry.coords' in the configuration.")
            ra, dec = self.config.photometry.coords
            report(
                self.reporter,
                "info",
                f"Using configured star coordinates at RA={float(ra):.6f} deg, Dec={float(dec):.6f} deg",
                stage="target",
            )
            return TargetInfo(ra=float(ra), dec=float(dec), jd=jd_value)

        if not self.config.photometry.target_id:
            raise TargetResolutionError("Asteroid mode requires 'photometry.target_id' in the configuration.")

        if self.config.photometry.ephemeris_provider == "miriade":
            return self._resolve_asteroid_with_miriade(jd_value)
        return self._resolve_asteroid_with_jpl(jd_value)

    def _resolve_asteroid_with_jpl(self, jd_value: float) -> TargetInfo:
        target_id = str(self.config.photometry.target_id)
        report(
            self.reporter,
            "info",
            f"Querying JPL Horizons for target {target_id} at JD={jd_value:.6f}",
            stage="target",
        )
        try:
            query = Horizons(
                id=target_id,
                location=self.config.observatory.observatory_code,
                epochs=jd_value,
            )
            ephemerides = query.ephemerides()
            lighttime_days = float(ephemerides["lighttime"][0]) / 60.0 / 24.0
            return TargetInfo(
                ra=float(ephemerides["RA"][0]),
                dec=float(ephemerides["DEC"][0]),
                jd=jd_value - lighttime_days,
                r=float(ephemerides["r"][0]),
                delta=float(ephemerides["delta"][0]),
                alpha=float(ephemerides["alpha"][0]),
            )
        except Exception as exc:
            raise TargetResolutionError(f"JPL Horizons query failed: {exc}") from exc

    def _resolve_asteroid_with_miriade(self, jd_value: float) -> TargetInfo:
        target_id = str(self.config.photometry.target_id)
        report(
            self.reporter,
            "info",
            f"Querying Miriade ephemcc for target {target_id} at JD={jd_value:.6f}",
            stage="target",
        )
        params = {
            "-name": _miriade_target_name(target_id),
            "-ep": f"{jd_value:.8f}",
            "-nbd": "1",
            "-tscale": "UTC",
            "-observer": self.config.observatory.observatory_code,
            "-teph": "1",
            "-tcoor": "5",
            "-rplane": "1",
            "-oscelem": "astorb",
            "-mime": "json",
            "-output": "--jd,--lighttime",
            "-from": "PhoPS",
        }
        request = Request(
            f"{MIRIADE_EPHEMCC_URL}?{urlencode(params)}",
            headers={"User-Agent": "PhoPS/0.1"},
        )
        try:
            with urlopen(request, timeout=MIRIADE_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            record = _miriade_data_record(payload)
            lighttime_minutes = _optional_float(record, ("LightTime", "lighttime"))
            lighttime_days = lighttime_minutes / 60.0 / 24.0 if lighttime_minutes is not None else 0.0
            return TargetInfo(
                ra=_ra_degrees(record),
                dec=_dec_degrees(record),
                jd=jd_value - lighttime_days,
                r=_optional_float(record, ("Dhelio", "r")),
                delta=_optional_float(record, ("Dobs", "delta", "Delta")),
                alpha=_optional_float(record, ("Phase", "alpha")),
            )
        except TargetResolutionError:
            raise
        except Exception as exc:
            raise TargetResolutionError(f"Miriade ephemcc query failed: {exc}") from exc
