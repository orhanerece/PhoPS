"""Compatibility wrapper for the installable package."""

from phops.plotting import plot_astrometry_residuals, plot_photometry_sources, save_target_cutout
from phops.utils import (
    calculate_residuals,
    dec_to_deg,
    observation_jd_from_header,
    observation_time_from_header,
    ra_to_deg,
)

__all__ = [
    "calculate_residuals",
    "dec_to_deg",
    "observation_jd_from_header",
    "observation_time_from_header",
    "plot_astrometry_residuals",
    "plot_photometry_sources",
    "ra_to_deg",
    "save_target_cutout",
]
