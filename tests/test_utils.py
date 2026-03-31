from pathlib import Path

import numpy as np
from astropy.io import fits

from phops.config import FitsKeywordsConfig
from phops.utils import (
    dec_to_deg,
    describe_input_selector,
    iter_input_files,
    load_fits_image,
    observation_time_from_header,
    ra_to_deg,
)


def test_ra_to_deg_hms() -> None:
    assert round(ra_to_deg("12:00:00"), 6) == 180.0


def test_dec_to_deg_dms() -> None:
    assert round(dec_to_deg("-10:30:00"), 6) == -10.5


def test_observation_time_from_header_supports_mjd_keywords() -> None:
    header = fits.Header()
    header["EXPSTART"] = 61009.71623874
    header["EXPTIME"] = 97.0

    time = observation_time_from_header(
        header,
        FitsKeywordsConfig(
            date_key="DATE-OBS",
            time_key="TIME-OBS",
            exposure_key="EXPTIME",
            jd_key="JD",
            mjd_key="EXPSTART",
        ),
    )

    assert abs(time.mjd - (61009.71623874 + 97.0 / 2.0 / 86400.0)) < 1e-9


def test_load_fits_image_uses_science_extension_with_merged_primary_header(tmp_path: Path) -> None:
    image_path = tmp_path / "hst_like_flc.fits"
    primary = fits.PrimaryHDU()
    primary.header["DATE-OBS"] = "2025-11-30"
    primary.header["TIME-OBS"] = "17:11:23"
    primary.header["EXPTIME"] = 97.0
    primary.header["RA_TARG"] = 115.9569605945
    primary.header["DEC_TARG"] = 21.2563485379

    sci_header = fits.Header()
    sci_header["EXTNAME"] = "SCI"
    sci_header["CRPIX1"] = 1024.0
    sci_header["CRPIX2"] = 1025.0
    sci_header["CRVAL1"] = 115.9569605945
    sci_header["CRVAL2"] = 21.2563485379
    sci_header["CTYPE1"] = "RA---TAN"
    sci_header["CTYPE2"] = "DEC--TAN"
    sci_header["CD1_1"] = -1.1e-5
    sci_header["CD1_2"] = 0.0
    sci_header["CD2_1"] = 0.0
    sci_header["CD2_2"] = 1.1e-5
    sci = fits.ImageHDU(data=np.ones((20, 20)), header=sci_header)

    err_header = sci_header.copy()
    err_header["EXTNAME"] = "ERR"
    err = fits.ImageHDU(data=np.zeros((20, 20)), header=err_header)
    fits.HDUList([primary, sci, err]).writeto(image_path)

    data, header, wcs = load_fits_image(image_path)

    assert data.shape == (20, 20)
    assert header["DATE-OBS"] == "2025-11-30"
    assert header["CRVAL1"] == 115.9569605945
    assert wcs.has_celestial


def test_iter_input_files_matches_common_fits_family(tmp_path: Path) -> None:
    for filename in ("a.fits", "b.fit", "c.fts", "d.fits.gz", "e.fit.gz", "f.fts.gz", "j_flc.fits"):
        (tmp_path / filename).write_text("x", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    matched = [path.name for path in iter_input_files(tmp_path, "fits")]

    assert matched == ["a.fits", "b.fit", "c.fts", "d.fits.gz", "e.fit.gz", "f.fts.gz", "j_flc.fits"]


def test_iter_input_files_supports_glob_patterns(tmp_path: Path) -> None:
    for filename in ("a_flt.fits", "b_flc.fits", "c_flc.fits.gz", "d.fits"):
        (tmp_path / filename).write_text("x", encoding="utf-8")

    matched = [path.name for path in iter_input_files(tmp_path, "*_flc.fits")]

    assert matched == ["b_flc.fits"]
    assert describe_input_selector("*_flc.fits") == "*_flc.fits"
