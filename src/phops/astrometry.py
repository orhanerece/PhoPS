"""Astrometry solver and Gaia index preparation."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.gaia import Gaia

from .config import AppConfig
from .errors import AstrometrySolveError, DependencyError
from .reporting import NullReporter, ProgressReporter, report
from .utils import dec_to_deg, load_fits_image, observation_time_from_header, ra_to_deg


@dataclass
class AstrometrySolver:
    """Create Gaia reference patches and run astrometry.net."""

    config: AppConfig
    reporter: ProgressReporter | None = None

    def __post_init__(self) -> None:
        self.reporter = self.reporter or NullReporter()
        self.config.ensure_runtime_dirs()
        self.key_file = self.config.paths.index_dir / "generated_index.csv"

    def _epoch_bucket(self, observation_time: Time) -> float:
        precision = self.config.astrometry.epoch_bucket_precision
        return round(float(observation_time.jyear), precision)

    def _area_suffix(self, ra: float, dec: float, observation_time: Time) -> str:
        epoch_tag = str(self._epoch_bucket(observation_time)).replace(".", "p")
        return f"{int(ra)}_{int(dec)}_{epoch_tag}"

    def catalog_patch_path(self, ra: float, dec: float, observation_time: Time) -> Path:
        suffix = self._area_suffix(ra, dec, observation_time)
        return self.config.paths.temp_dir / f"gaia_reference_{suffix}.fits"

    def extract_field_coordinates(self, header, image_name: str = "<header>", wcs: WCS | None = None) -> tuple[float, float, Time]:
        """Read field coordinates and observation time from a FITS header."""

        ra_raw = header.get(self.config.fits_keywords.ra_key)
        dec_raw = header.get(self.config.fits_keywords.dec_key)
        if ra_raw is not None and dec_raw is not None:
            ra_img = ra_to_deg(ra_raw)
            dec_img = dec_to_deg(dec_raw)
        else:
            if "CRVAL1" not in header or "CRVAL2" not in header:
                raise AstrometrySolveError(
                    f"Could not determine field coordinates for {image_name}. "
                    f"Expected FITS keywords '{self.config.fits_keywords.ra_key}'/'{self.config.fits_keywords.dec_key}' "
                    "or a celestial WCS with CRVAL1/CRVAL2."
                )
            resolved_wcs = wcs
            if resolved_wcs is None:
                try:
                    resolved_wcs = WCS(header)
                except Exception:
                    resolved_wcs = None
            if resolved_wcs is not None and not resolved_wcs.has_celestial:
                raise AstrometrySolveError(
                    f"Could not determine field coordinates for {image_name}. "
                    f"Expected FITS keywords '{self.config.fits_keywords.ra_key}'/'{self.config.fits_keywords.dec_key}' "
                    "or a celestial WCS with CRVAL1/CRVAL2."
                )
            ra_img = float(header["CRVAL1"])
            dec_img = float(header["CRVAL2"])

        observation_time = observation_time_from_header(header, self.config.fits_keywords)
        return ra_img, dec_img, observation_time

    def validate_existing_wcs(self, header, image_name: str, wcs: WCS | None = None) -> None:
        """Ensure the FITS header contains a usable celestial WCS."""

        wcs = wcs or WCS(header)
        if not wcs.has_celestial:
            raise AstrometrySolveError(
                f"'existing_wcs' mode requires a celestial WCS in {image_name}. "
                "Run astrometric solving first or switch back to 'solve' mode."
            )

    def _cache_entries(self) -> list[tuple[float, float, float]]:
        if not self.key_file.exists():
            return []
        entries: list[tuple[float, float, float]] = []
        with self.key_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if len(row) == 2:
                    entries.append((float(row[0]), float(row[1]), math.nan))
                elif len(row) >= 3:
                    entries.append((float(row[0]), float(row[1]), float(row[2])))
        return entries

    def _add_cache_entry(self, ra: float, dec: float, observation_time: Time) -> None:
        with self.key_file.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([ra, dec, self._epoch_bucket(observation_time)])

    def _is_cached_area(self, ra: float, dec: float, observation_time: Time) -> bool:
        tolerance = self.config.astrometry.cache_tolerance
        epoch_bucket = self._epoch_bucket(observation_time)
        for cached_ra, cached_dec, cached_epoch in self._cache_entries():
            distance = math.sqrt((ra - cached_ra) ** 2 + (dec - cached_dec) ** 2)
            same_epoch = math.isnan(cached_epoch) or cached_epoch == epoch_bucket
            if distance < tolerance and same_epoch:
                return True
        return False

    def _check_binary(self, binary_name: str) -> None:
        if shutil.which(binary_name) is None:
            raise DependencyError(
                f"Required external command '{binary_name}' was not found. Install astrometry.net and make sure the binary is on PATH."
            )

    def _run_command(self, args: list[str]) -> None:
        self._check_binary(args[0])
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "no stderr output"
            raise AstrometrySolveError(f"Command failed: {' '.join(args)} | {stderr}") from exc

    def ensure_reference_patch(self, ra: float, dec: float, observation_time: Time) -> Path:
        patch_path = self.catalog_patch_path(ra, dec, observation_time)
        if patch_path.exists() and self._is_cached_area(ra, dec, observation_time):
            return patch_path
        return self._prepare_gaia_index(ra, dec, observation_time, override_patch_path=patch_path)

    def _prepare_gaia_index(
        self,
        ra: float,
        dec: float,
        observation_time: Time,
        override_patch_path: Path | None = None,
    ) -> Path:
        suffix = self._area_suffix(ra, dec, observation_time)
        report(
            self.reporter,
            "info",
            f"Generating Gaia reference patch for RA={ra:.3f}, Dec={dec:.3f}, epoch={observation_time.jyear:.2f}",
            stage="astrometry",
        )

        query = f"""
        SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, bp_rp, parallax
        FROM {self.config.catalog}
        WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {self.config.astrometry.radius})) = 1
        ORDER BY phot_g_mean_mag
        """
        job = Gaia.launch_job_async(query)
        data = job.get_results()
        if len(data) == 0:
            raise AstrometrySolveError("Gaia query returned no stars for the requested field.")

        def _valid_column_mask(column) -> np.ndarray:
            raw_mask = getattr(column, "mask", None)
            if raw_mask is None:
                return np.ones(len(column), dtype=bool)
            return ~np.asarray(raw_mask, dtype=bool)

        mask = _valid_column_mask(data["pmra"]) & _valid_column_mask(data["pmdec"]) & _valid_column_mask(data["parallax"])
        data = data[mask]
        if len(data) == 0:
            raise AstrometrySolveError("Gaia stars do not contain enough proper-motion information.")

        parallax = np.asarray(data["parallax"], dtype=float)
        valid = (parallax > 0.1) & np.isfinite(parallax)
        data = data[valid]
        parallax = parallax[valid]
        if len(data) == 0:
            raise AstrometrySolveError("No Gaia stars remained after parallax filtering.")

        gaia_epoch = Time(2016.0, format="jyear", scale="tdb")
        target_time = observation_time.tdb
        distance = (parallax * u.mas).to(u.pc, equivalencies=u.parallax())
        coordinates = SkyCoord(
            ra=np.asarray(data["ra"], dtype=float) * u.deg,
            dec=np.asarray(data["dec"], dtype=float) * u.deg,
            pm_ra_cosdec=np.asarray(data["pmra"], dtype=float) * u.mas / u.yr,
            pm_dec=np.asarray(data["pmdec"], dtype=float) * u.mas / u.yr,
            distance=distance,
            obstime=gaia_epoch,
        )
        propagated = coordinates.apply_space_motion(new_obstime=target_time)
        data["ra"] = propagated.ra.deg
        data["dec"] = propagated.dec.deg
        data = data[np.isfinite(data["ra"]) & np.isfinite(data["dec"])]
        if len(data) == 0:
            raise AstrometrySolveError("No Gaia stars remained after epoch propagation.")

        patch_path = override_patch_path or self.catalog_patch_path(ra, dec, observation_time)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        data.write(patch_path, overwrite=True)

        if self.config.astrometry.solve_mode == "existing_wcs":
            self._add_cache_entry(ra, dec, observation_time)
            return patch_path

        hp_out = self.config.paths.temp_dir / f"gaia-hp%02i_{suffix}.fits"
        self._run_command(["hpsplit", "-o", str(hp_out), "-n", "2", str(patch_path)])
        tile_files = sorted(
            path
            for path in self.config.paths.temp_dir.iterdir()
            if path.name.startswith("gaia-hp")
            and path.name.endswith(f"_{suffix}.fits")
        )
        for tile_path in tile_files:
            tile_id = tile_path.name.split("gaia-hp", maxsplit=1)[1].split("_", maxsplit=1)[0]
            for scale in self.config.astrometry.quad_scales:
                index_path = self.config.paths.index_dir / f"index-550{scale:02d}-{tile_id}_{suffix}.fits"
                index_id = f"550{scale:02d}{tile_id}{abs(int(ra))}"
                self._run_command(
                    [
                        "build-astrometry-index",
                        "-i",
                        str(tile_path),
                        "-s",
                        "2",
                        "-P",
                        str(scale),
                        "-E",
                        "-S",
                        "phot_g_mean_mag",
                        "-o",
                        str(index_path),
                        "-I",
                        index_id,
                    ]
                )

        for tile_path in tile_files:
            tile_path.unlink(missing_ok=True)
        self._add_cache_entry(ra, dec, observation_time)
        return patch_path

    def solve(self, image_path: str | Path) -> Path:
        """Solve a FITS image and return the solved FITS path."""

        image_path = Path(image_path).resolve()
        _, header, wcs = load_fits_image(image_path)
        ra_img, dec_img, observation_time = self.extract_field_coordinates(header, image_path.name, wcs=wcs)

        if self.config.astrometry.solve_mode == "existing_wcs":
            self.validate_existing_wcs(header, image_path.name, wcs=wcs)
            report(self.reporter, "info", f"Using existing WCS from {image_path.name}", stage="astrometry")
            return image_path

        self.ensure_reference_patch(ra_img, dec_img, observation_time)
        output_fits = self.config.paths.solve_dir / f"{image_path.stem}_new.fits"
        pixel_scale = self.config.instrument.pixel_scale
        command = [
            "solve-field",
            str(image_path),
            "--dir",
            str(self.config.paths.solve_dir),
            "--new-fits",
            str(output_fits),
            "--scale-units",
            "arcsecperpix",
            "--scale-low",
            str(pixel_scale * 0.7),
            "--scale-high",
            str(pixel_scale * 1.4),
            "--index-dir",
            str(self.config.paths.index_dir),
            "--overwrite",
            "--no-plots",
            "--solved",
            "none",
            "--corr",
            "none",
            "--rdls",
            "none",
            "--match",
            "none",
            "--index-xyls",
            "none",
            "--axy",
            "none",
        ]
        report(self.reporter, "info", f"Solving astrometry for {image_path.name}", stage="astrometry")
        self._run_command(command)
        temp_wcs = self.config.paths.solve_dir / f"{image_path.stem}.wcs"
        temp_wcs.unlink(missing_ok=True)
        if not output_fits.exists():
            raise AstrometrySolveError(f"Astrometry command completed but {output_fits.name} was not produced.")
        report(self.reporter, "info", f"Astrometric solution saved: {output_fits.name}", stage="astrometry")
        return output_fits
