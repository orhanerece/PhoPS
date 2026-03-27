"""Target coordinate resolution."""

from __future__ import annotations

from dataclasses import dataclass

from astroquery.jplhorizons import Horizons

from .config import AppConfig
from .errors import TargetResolutionError
from .reporting import NullReporter, ProgressReporter, report
from .utils import observation_jd_from_header


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

        report(
            self.reporter,
            "info",
            f"Querying JPL Horizons for target {self.config.photometry.target_id} at JD={jd_value:.6f}",
            stage="target",
        )
        try:
            query = Horizons(
                id=str(self.config.photometry.target_id),
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
