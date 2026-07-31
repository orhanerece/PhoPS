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

SKYBOT_RESOLVER_URL = "https://ssp.imcce.fr/webservices/skybot/api/resolver.php"
SKYBOT_TIMEOUT_SECONDS = 30
AU_LIGHT_TIME_DAYS = 499.004783836 / 86400.0


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


def _skybot_ra_degrees(record: dict[str, Any]) -> float:
    value = _optional_float(record, ("RA (deg)", "_raj2000", "RA"))
    if value is not None:
        return value

    hour_value = record.get("RA (hour)")
    if hour_value in (None, ""):
        raise TargetResolutionError("SkyBot response did not include a valid RA.")
    return ra_to_deg(":".join(str(hour_value).strip().split()))


def _skybot_dec_degrees(record: dict[str, Any]) -> float:
    value = _optional_float(record, ("DEC (deg)", "_decj2000", "DEC"))
    if value is not None:
        return value

    dec_value = record.get("DEC")
    if dec_value in (None, ""):
        raise TargetResolutionError("SkyBot response did not include a valid Dec.")
    return dec_to_deg(":".join(str(dec_value).strip().split()))


def _skybot_records_from_payload(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        flag = payload.get("flag")
        if flag in (0, -1, "0", "-1"):
            status = payload.get("status", "unknown error")
            raise TargetResolutionError(f"SkyBot resolver returned an error: {status}")
        result = payload.get("result")
        if isinstance(result, str):
            payload = json.loads(result)
        elif result is not None:
            payload = result

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise TargetResolutionError("SkyBot resolver returned an unexpected response format.")

    records = [item for item in payload if isinstance(item, dict)]
    if not records:
        raise TargetResolutionError("SkyBot resolver did not return a matching target.")
    return records


def _normalise_identifier(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").strip("()")


def _select_skybot_record(records: list[dict[str, Any]], target_id: str) -> dict[str, Any]:
    target_key = _normalise_identifier(target_id)
    for record in records:
        if _normalise_identifier(record.get("Num", "")) == target_key:
            return record
        if _normalise_identifier(record.get("Name", "")) == target_key:
            return record
    if len(records) == 1:
        return records[0]
    candidates = ", ".join(str(record.get("Name") or record.get("Num") or "<unnamed>") for record in records[:5])
    raise TargetResolutionError(f"SkyBot resolver returned multiple candidates for {target_id}: {candidates}")


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

        if self.config.photometry.ephemeris_provider == "skybot":
            return self._resolve_asteroid_with_skybot(jd_value)
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

    def _resolve_asteroid_with_skybot(self, jd_value: float) -> TargetInfo:
        target_id = str(self.config.photometry.target_id)
        report(
            self.reporter,
            "info",
            f"Querying SkyBot resolver for target {target_id} at JD={jd_value:.6f}",
            stage="target",
        )
        params = {
            "-name": target_id,
            "-ep": f"{jd_value:.8f}",
            "-mime": "json",
            "-output": "all",
            "-observer": self.config.observatory.observatory_code,
            "-refsys": "EQJ2000",
            "-from": "PhoPS",
        }
        request = Request(
            f"{SKYBOT_RESOLVER_URL}?{urlencode(params)}",
            headers={"User-Agent": "PhoPS/0.1"},
        )
        try:
            with urlopen(request, timeout=SKYBOT_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            record = _select_skybot_record(_skybot_records_from_payload(payload), target_id)
            delta = _optional_float(record, ("dg (ua)", "dg (au)", "delta", "Delta"))
            lighttime_days = delta * AU_LIGHT_TIME_DAYS if delta is not None else 0.0
            return TargetInfo(
                ra=_skybot_ra_degrees(record),
                dec=_skybot_dec_degrees(record),
                jd=jd_value - lighttime_days,
                r=_optional_float(record, ("dh (ua)", "dh (au)", "r")),
                delta=delta,
                alpha=_optional_float(record, ("Phase (deg)", "alpha")),
            )
        except TargetResolutionError:
            raise
        except Exception as exc:
            raise TargetResolutionError(f"SkyBot resolver query failed: {exc}") from exc
