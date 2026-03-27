"""Photometry processing for reference stars and targets."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import warnings

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.centroids import centroid_quadratic
from photutils.detection import DAOStarFinder
import matplotlib

MPL_DIR = Path(tempfile.gettempdir()) / "phops-mpl"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor

from .config import AppConfig
from .plotting import plot_photometry_sources, save_target_cutout
from .reporting import NullReporter, ProgressReporter, report
from .target import TargetInfo
from .utils import safe_stem


class Photometry:
    """Photometry engine."""

    def __init__(self, config: AppConfig, reporter: ProgressReporter | None = None) -> None:
        self.config = config
        self.reporter = reporter or NullReporter()
        self.pixel_scale = self.config.instrument.pixel_scale
        self.saturation_limit = self.config.instrument.saturation_level
        self.fwhm_guess = self.config.source_detection.fwhm_guess
        self.threshold_sigma = self.config.source_detection.threshold_sigma
        self.edge_margin = self.config.source_detection.edge_margin
        self.min_area = self.config.source_detection.min_area
        self.isolation_radius_arcsec = self.config.matching.isolation_radius_arcsec
        self.match_distance_arcsec = self.config.matching.match_constraint_arcsec

    def get_clean_gaia_matches(self, image_path: str | Path, gaia_catalog_path: str | Path):
        """Detect image sources and match them to a Gaia reference patch."""

        image_path = Path(image_path)
        gaia_catalog_path = Path(gaia_catalog_path)
        with fits.open(image_path) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header
            wcs = WCS(header)

        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        finder = DAOStarFinder(
            fwhm=self.fwhm_guess,
            threshold=std * self.threshold_sigma,
            exclude_border=True,
        )
        image_sources = finder(data - median)
        if image_sources is None or len(image_sources) == 0:
            return None, None

        quality_mask = (
            (image_sources["peak"] < self.saturation_limit)
            & (image_sources["xcentroid"] > self.edge_margin)
            & (image_sources["xcentroid"] < (data.shape[1] - self.edge_margin))
            & (image_sources["ycentroid"] > self.edge_margin)
            & (image_sources["ycentroid"] < (data.shape[0] - self.edge_margin))
        )
        if "npix" in image_sources.colnames:
            quality_mask &= image_sources["npix"] >= self.min_area
        sources = image_sources[quality_mask]
        if len(sources) == 0:
            return None, image_sources

        isolation_radius_pixels = self.isolation_radius_arcsec / self.pixel_scale
        coordinates = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        final_indices: list[int] = []
        for index, point in enumerate(coordinates):
            distances = np.sqrt(np.sum((coordinates - point) ** 2, axis=1))
            if np.sum(distances < isolation_radius_pixels) == 1:
                final_indices.append(index)
        clean_sources = sources[final_indices]
        if len(clean_sources) == 0:
            return None, image_sources

        gaia_table = Table.read(gaia_catalog_path)
        gaia_coords = SkyCoord(ra=gaia_table["ra"], dec=gaia_table["dec"], unit="deg")
        ra_img, dec_img = wcs.all_pix2world(clean_sources["xcentroid"], clean_sources["ycentroid"], 0)
        image_coords = SkyCoord(ra=ra_img, dec=dec_img, unit="deg", frame="icrs")
        indices, distances, _ = image_coords.match_to_catalog_sky(gaia_coords)
        match_mask = distances < (self.match_distance_arcsec * u.arcsec)

        matched_table = clean_sources[match_mask]
        if len(matched_table) == 0:
            return None, image_sources
        matched_table["img_ra"] = ra_img[match_mask]
        matched_table["img_dec"] = dec_img[match_mask]
        matched_table["gaia_ra"] = gaia_table["ra"][indices[match_mask]]
        matched_table["gaia_dec"] = gaia_table["dec"][indices[match_mask]]
        matched_table["gaia_gmag"] = gaia_table["phot_g_mean_mag"][indices[match_mask]]
        matched_table["bp_rp"] = gaia_table["bp_rp"][indices[match_mask]]
        matched_table["pm_ra"] = gaia_table["pmra"][indices[match_mask]]
        matched_table["pm_dec"] = gaia_table["pmdec"][indices[match_mask]]
        matched_table["r_dist"] = np.sqrt(
            (matched_table["xcentroid"] - data.shape[1] / 2) ** 2
            + (matched_table["ycentroid"] - data.shape[0] / 2) ** 2
        )

        if self.config.plots.plot_image:
            output_path = self.config.paths.plot_dir / f"{image_path.stem}_sources_{self.config.plots.image_scale}.png"
            plot_photometry_sources(
                data,
                image_sources,
                matched_table,
                wcs,
                output_path=output_path,
                mode=self.config.plots.image_scale,
                reporter=self.reporter,
            )
        return matched_table, image_sources

    def transform_gaia_to_filter(self, matched_table: Table) -> Table:
        """Transform Gaia G magnitudes into the configured target filter."""

        target_filter = self.config.photometry.filter
        coefficients = {
            "B": [0.01448, -0.6874, -0.3604, 0.06718, -0.006061],
            "V": [-0.02704, 0.01424, -0.2156, 0.01426, 0.0],
            "R": [-0.02275, 0.3961, -0.1243, -0.01396, 0.003775],
            "I": [0.01753, 0.76, -0.0991, 0.0, 0.0],
            "g": [0.2199, -0.6365, -0.1548, 0.0064, 0.0],
            "r": [-0.09837, 0.08592, 0.1907, -0.1701, 0.02263],
            "i": [-0.293, 0.6404, -0.09609, -0.002104, 0.0],
            "z": [-0.4619, 0.8992, -0.08271, 0.005029, 0.0],
        }
        if target_filter not in coefficients:
            report(
                self.reporter,
                "warning",
                f"Unknown filter '{target_filter}'. Falling back to Gaia G magnitudes.",
                stage="photometry",
            )
            matched_table["standard_mag"] = np.asarray(matched_table["gaia_gmag"], dtype=float)
            return matched_table

        bprp = np.asarray(matched_table["bp_rp"], dtype=float)
        gaia_mag = np.asarray(matched_table["gaia_gmag"], dtype=float)
        c0, c1, c2, c3, c4 = coefficients[target_filter]
        delta_mag = c0 + c1 * bprp + c2 * bprp**2 + c3 * bprp**3 + c4 * bprp**4
        matched_table["standard_mag"] = gaia_mag - delta_mag
        valid_mask = ~np.isnan(matched_table["standard_mag"])
        transformed = matched_table[valid_mask]
        report(
            self.reporter,
            "info",
            f"Transformed {len(transformed)} Gaia sources into '{target_filter}' band.",
            stage="photometry",
        )
        return transformed

    def perform_aperture_photometry(self, data: np.ndarray, matched_table: Table, image_sources: Table):
        """Refine source centroids and measure aperture photometry."""

        if "fwhm" in image_sources.colnames:
            median_fwhm = float(np.median(image_sources["fwhm"]))
        else:
            estimated = 2.0 * np.sqrt(np.asarray(image_sources["npix"], dtype=float) / np.pi)
            median_fwhm = float(np.median(estimated))
        radii = self.calculate_radii(median_fwhm)
        radius, radius_in, radius_out = radii
        report(
            self.reporter,
            "info",
            f"Median FWHM={median_fwhm:.2f}px; aperture radii=({radius:.2f}, {radius_in:.2f}, {radius_out:.2f})",
            stage="photometry",
        )

        box_size = int(median_fwhm * 2)
        if box_size % 2 == 0:
            box_size += 1

        refined_positions: list[list[float]] = []
        for row in matched_table:
            x_init = float(row["xcentroid"])
            y_init = float(row["ycentroid"])
            y_min = int(y_init - box_size // 2)
            y_max = int(y_init + box_size // 2 + 1)
            x_min = int(x_init - box_size // 2)
            x_max = int(x_init + box_size // 2 + 1)
            if y_min < 0 or x_min < 0 or y_max > data.shape[0] or x_max > data.shape[1]:
                refined_positions.append([x_init, y_init])
                continue
            cutout = data[y_min:y_max, x_min:x_max]
            try:
                cutout_clean = cutout - np.median(cutout)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dx, dy = centroid_quadratic(cutout_clean)
                refined_positions.append([x_min + float(dx), y_min + float(dy)])
            except Exception:
                refined_positions.append([x_init, y_init])

        refined = np.asarray(refined_positions)
        valid_mask = ~np.isnan(refined).any(axis=1)
        refined = refined[valid_mask]
        measured_table = matched_table[valid_mask]
        measured_table["x_precise"] = refined[:, 0]
        measured_table["y_precise"] = refined[:, 1]

        apertures = CircularAperture(refined, r=radius)
        annuli = CircularAnnulus(refined, r_in=radius_in, r_out=radius_out)
        raw_photometry = aperture_photometry(data, [apertures, annuli])
        annulus_masks = annuli.to_mask(method="center")
        background_medians: list[float] = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(data)
            if annulus_data is None:
                background_medians.append(0.0)
                continue
            data_1d = annulus_data[mask.data > 0]
            background_medians.append(float(np.median(data_1d)) if data_1d.size else 0.0)

        background = np.asarray(background_medians)
        net_flux = np.asarray(raw_photometry["aperture_sum_0"], dtype=float) - (background * apertures.area)
        net_flux[net_flux <= 0] = np.nan
        gain = self.config.instrument.gain
        read_noise = self.config.instrument.read_noise
        noise = np.sqrt(net_flux / gain + (apertures.area * background) / gain + (apertures.area * read_noise**2))
        measured_table["inst_mag"] = 25.0 - 2.5 * np.log10(net_flux)
        measured_table["mag_err"] = 1.0857 * (noise / net_flux)
        measured_table["snr"] = net_flux / noise
        report(self.reporter, "info", f"Measured {len(measured_table)} reference stars.", stage="photometry")
        return measured_table, median_fwhm

    def calculate_zeropoint_model(
        self,
        matched_table: Table,
        plot: bool = True,
        save_plot: bool = True,
        output_path: str | Path | None = None,
    ) -> tuple[np.poly1d, float, float, float]:
        """Fit a first-order zeropoint model as a function of field radius."""

        mag_diff = np.asarray(matched_table["standard_mag"] - matched_table["inst_mag"], dtype=float)
        radius = np.asarray(matched_table["r_dist"], dtype=float)
        valid_mask = np.isfinite(mag_diff) & np.isfinite(radius)
        mag_diff = mag_diff[valid_mask]
        radius = radius[valid_mask]

        fallback_average = float(np.nanmean(mag_diff)) if len(mag_diff) else 25.0
        if len(mag_diff) < 4:
            report(self.reporter, "warning", "Not enough stars for zeropoint fit; using average fallback.", stage="photometry")
            return np.poly1d([0.0, fallback_average]), 0.0, 0.0, fallback_average

        x_data = radius.reshape(-1, 1)
        y_data = mag_diff.reshape(-1, 1)
        ransac = RANSACRegressor(residual_threshold=0.1, random_state=100)
        ransac.fit(x_data, y_data)
        inlier_mask = np.asarray(ransac.inlier_mask_, dtype=bool)
        outlier_mask = ~inlier_mask
        slope = float(ransac.estimator_.coef_[0][0])
        intercept = float(ransac.estimator_.intercept_[0])
        zeropoint_function = np.poly1d([slope, intercept])
        rms_error = float(np.std(y_data[inlier_mask] - zeropoint_function(x_data[inlier_mask])))
        average_zeropoint = float(np.average(mag_diff[inlier_mask]))

        report(
            self.reporter,
            "info",
            f"Zeropoint fit: slope={slope:.6f}, intercept={intercept:.4f}, rms={rms_error:.4f}",
            stage="photometry",
        )

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(radius[inlier_mask], mag_diff[inlier_mask], color="blue", alpha=0.6, label=f"Inliers ({np.sum(inlier_mask)})")
            ax.scatter(radius[outlier_mask], mag_diff[outlier_mask], color="red", alpha=0.3, label=f"Outliers ({np.sum(outlier_mask)})")
            sample = np.linspace(0, float(np.max(radius)), 100)
            ax.plot(sample, zeropoint_function(sample), color="black", linestyle="--", label=f"Model slope={slope:.6f}")
            ax.set_xlabel("Distance from image center (pixels)")
            ax.set_ylabel("Standard magnitude - instrumental magnitude")
            ax.legend()
            if save_plot and output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

        return zeropoint_function, rms_error, slope, average_zeropoint

    def calculate_radii(self, median_fwhm: float) -> tuple[float, float, float]:
        """Return aperture and annulus radii in pixels."""

        method = self.config.photometry.aperture_method
        aperture = self.config.photometry.aperture
        annulus_inner = self.config.photometry.annulus_inner
        annulus_outer = self.config.photometry.annulus_outer

        if method == "fwhm_factor":
            radius = median_fwhm * aperture
            radius_in = median_fwhm * annulus_inner
            radius_out = median_fwhm * annulus_outer
        elif method == "fixed_arcsec":
            radius = aperture / self.pixel_scale
            radius_in = radius + annulus_inner / self.pixel_scale
            radius_out = radius + annulus_outer / self.pixel_scale
        else:
            radius = aperture
            radius_in = annulus_inner
            radius_out = annulus_outer

        if radius_in <= radius:
            radius_in = radius + 2.0
        if radius_out <= radius_in:
            radius_out = radius_in + 5.0
        return float(radius), float(radius_in), float(radius_out)

    def measure_target(
        self,
        data: np.ndarray,
        wcs: WCS,
        target: TargetInfo,
        zp_function: np.poly1d,
        median_fwhm: float,
        all_detected: Table | None,
        filename: str,
        zp_average: float,
    ) -> dict[str, float] | None:
        """Measure the target source using the fitted zeropoint."""

        x_target, y_target = wcs.all_world2pix(target.ra, target.dec, 0)
        if all_detected is not None and len(all_detected) > 0:
            isolation_radius_pixels = self.isolation_radius_arcsec / self.pixel_scale
            all_coordinates = np.transpose((all_detected["xcentroid"], all_detected["ycentroid"]))
            distances = np.sqrt(np.sum((all_coordinates - [x_target, y_target]) ** 2, axis=1))
            nearby_mask = distances < isolation_radius_pixels
            nearby_count = int(np.sum(nearby_mask))
            if nearby_count > 1:
                report(
                    self.reporter,
                    "warning",
                    f"Target rejected by isolation criterion ({nearby_count} nearby detections).",
                    stage="photometry",
                )
                return None
            if nearby_count == 1:
                nearby_source = all_detected[nearby_mask][0]
                x_target = float(nearby_source["xcentroid"])
                y_target = float(nearby_source["ycentroid"])

        radius, radius_in, radius_out = self.calculate_radii(median_fwhm)
        box_size = int(median_fwhm * 2)
        if box_size % 2 == 0:
            box_size += 1
        y_min = int(y_target - box_size // 2)
        y_max = int(y_target + box_size // 2 + 1)
        x_min = int(x_target - box_size // 2)
        x_max = int(x_target + box_size // 2 + 1)
        if y_min < 0 or x_min < 0 or y_max > data.shape[0] or x_max > data.shape[1]:
            return None

        cutout = data[y_min:y_max, x_min:x_max]
        cutout_clean = cutout - np.median(cutout)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dx, dy = centroid_quadratic(cutout_clean)
            if not (np.isfinite(dx) and np.isfinite(dy)):
                raise ValueError("Centroiding returned non-finite values.")
            x_precise = x_min + float(dx)
            y_precise = y_min + float(dy)
        except Exception:
            x_precise = float(x_target)
            y_precise = float(y_target)

        aperture = CircularAperture([(x_precise, y_precise)], r=radius)
        annulus = CircularAnnulus([(x_precise, y_precise)], r_in=radius_in, r_out=radius_out)
        photometry = aperture_photometry(data, [aperture, annulus])
        annulus_mask = annulus.to_mask(method="center")[0]
        annulus_data = annulus_mask.multiply(data)
        background_values = annulus_data[annulus_mask.data > 0]
        background_median = float(np.median(background_values)) if background_values.size else 0.0
        net_flux = float(photometry["aperture_sum_0"][0]) - (background_median * aperture.area)
        if net_flux <= 0:
            return None

        radius_from_center = float(np.sqrt((x_precise - data.shape[1] / 2) ** 2 + (y_precise - data.shape[0] / 2) ** 2))
        if self.config.photometry.zeropoint == "average":
            target_zeropoint = float(zp_average)
        else:
            target_zeropoint = float(zp_function(radius_from_center))

        instrumental_magnitude = 25.0 - 2.5 * np.log10(net_flux)
        calibrated_magnitude = float(instrumental_magnitude + target_zeropoint)
        gain = self.config.instrument.gain
        read_noise = self.config.instrument.read_noise
        noise = float(
            np.sqrt(net_flux / gain + (aperture.area * background_median) / gain + (aperture.area * read_noise**2))
        )
        magnitude_error = float(1.0857 * (noise / net_flux))
        cutout_name = safe_stem(f"target_{filename}")
        save_target_cutout(
            data,
            x_precise,
            y_precise,
            cutout_name,
            output_dir=self.config.paths.cutout_dir,
            reporter=self.reporter,
        )
        return {
            "ra": float(target.ra),
            "dec": float(target.dec),
            "x": float(x_precise),
            "y": float(y_precise),
            "mag_inst": float(instrumental_magnitude),
            "zp": float(target_zeropoint),
            "mag_calib": calibrated_magnitude,
            "err": magnitude_error,
            "snr": float(net_flux / noise),
            "BG": background_median,
        }
