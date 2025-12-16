#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ORHANERECE — Dynamic Astrometry Module

import os
import subprocess
from pathlib import Path
from astroquery.gaia import Gaia
from astropy.io import fits

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def ra_to_deg(ra_str):
    """Convert RA 'HH:MM:SS' to degrees."""
    h, m, s = [float(x) for x in ra_str.split(":")]
    return 15 * (h + m/60 + s/3600)


def dec_to_deg(dec_str):
    """Convert DEC '±DD:MM:SS' to degrees."""
    d, m, s = [float(x) for x in dec_str.split(":")]
    sign = -1 if str(d).startswith("-") else 1
    d = abs(float(d))
    return sign * (d + float(m)/60 + float(s)/3600)


# -------------------------------------------------
# Main Class
# -------------------------------------------------

class DynamicAstrometry:
    def __init__(
        self,
        fits_file,
        solve_dir=".",
        index_dir="output_index",
        hp_dir="gaia_hp",
        input_cat_dir="input_cat",
        radius_deg=0.5,
        pixel_scale=0.61
    ):
        self.fits_file = fits_file
        self.solve_dir = solve_dir
        self.index_dir = index_dir
        self.hp_dir = hp_dir
        self.input_cat_dir = input_cat_dir

        self.radius_deg = radius_deg
        self.pixel_scale = pixel_scale
        self.pixel_scale_low = pixel_scale * 0.7
        self.pixel_scale_high = pixel_scale * 1.4

        # Create directories
        os.makedirs(self.solve_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.hp_dir, exist_ok=True)
        os.makedirs(self.input_cat_dir, exist_ok=True)

        self.patch_file = os.path.join(self.input_cat_dir, "gaia_patch.fits")

    # -------------------------------------------------
    # Step 1: Read RA/DEC from FITS header
    # -------------------------------------------------
    def get_header_coords(self):
        with fits.open(self.fits_file) as hdul:
            hdr = hdul[0].header
            ra_str = hdr.get("OBJCTRA", hdr.get("RA"))
            dec_str = hdr.get("OBJCTDEC", hdr.get("DEC"))

        self.ra_deg = ra_to_deg(ra_str)
        self.dec_deg = dec_to_deg(dec_str)

        return self.ra_deg, self.dec_deg

    # -------------------------------------------------
    # Step 2: Download Gaia patch
    # -------------------------------------------------
    def download_gaia_patch(self):
        print("Gaia sorgusu gönderiliyor...")

        adql = f"""
        SELECT
            gaia_source.source_id,
            gaia_source.ra,
            gaia_source.dec,
            gaia_source.pmra,
            gaia_source.pmdec,
            gaia_source.phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', gaia_source.ra, gaia_source.dec),
            CIRCLE('ICRS', {self.ra_deg}, {self.dec_deg}, {self.radius_deg})
        ) = 1
        ORDER BY phot_g_mean_mag
        """

        job = Gaia.launch_job_async(adql)
        gaia_data = job.get_results()
        gaia_data.write(self.patch_file, overwrite=True)

        print(f"{len(gaia_data)} Gaia kaynağı indirildi → {self.patch_file}")

    # -------------------------------------------------
    # Step 3: hpsplit — HEALPix tiles
    # -------------------------------------------------
    def generate_healpix_tiles(self):
        print("HEALPix tile üretiliyor...")

        cmd = [
            "hpsplit",
            "-o", f"{self.hp_dir}/gaia-hp%02i.fits",
            "-n", "2",
            self.patch_file
        ]

        subprocess.run(cmd, check=True)
        print("hpsplit tamamlandı.")

        self.tile_files = sorted([
            f for f in os.listdir(self.hp_dir) if f.endswith(".fits")
        ])

        self.tile_ids = [int(f.split("gaia-hp")[1].split(".")[0]) for f in self.tile_files]

        print("Bulunan tile ID'leri:", self.tile_ids)

    # -------------------------------------------------
    # Step 4: build-astrometry-index
    # -------------------------------------------------
    def build_index_files(self):
        print("\nIndex üretimi başlıyor...")

        quad_scales = [0, 2, 4, 6]

        for tile_id in self.tile_ids:
            tile_path = f"{self.hp_dir}/gaia-hp{tile_id:02d}.fits"

            for P in quad_scales:
                ss = f"{P:02d}"
                output_index = f"{self.index_dir}/index-550{ss}-{tile_id:02d}.fits"
                index_id = f"550{ss}{tile_id:02d}"

                cmd = [
                    "build-astrometry-index",
                    "-i", tile_path,
                    "-s", "2",
                    "-P", str(P),
                    "-E",
                    "-S", "phot_g_mean_mag",
                    "-o", output_index,
                    "-I", index_id
                ]

                print(f"P={P}, Tile={tile_id}: {output_index}")
                subprocess.run(cmd, check=True)

        print("Tüm index dosyaları oluşturuldu.")

    # -------------------------------------------------
    # Step 5: solve-field
    # -------------------------------------------------
    def solve_with_astrometry(self):
        print("\nsolve-field çalıştırılıyor...")

        # indexleri ortam değişkenine ekle
        env = os.environ.copy()
        env["ASTROMETRY_INDEX_PATH"] = os.path.abspath(self.index_dir)

        cmd = [
            "solve-field",
            self.fits_file,
            "--dir", self.solve_dir,
            "--scale-units", "arcsecperpix",
            "--scale-low", str(self.pixel_scale_low),
            "--scale-high", str(self.pixel_scale_high),
            "--overwrite",
            "--no-plots",
            "--match", "none",
            "--rdls", "none",
            "--solved", "none",
            "--corr", "none",
            "--no-verify",
            "--index-xyls", "none",
            "--axy", "none",
            "--new-fits", os.path.splitext(self.fits_file)[0]+"_new.fits"
        ]
        print("Komut:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        print("solve-field tamamlandı!")

    # -------------------------------------------------
    # RUN ALL PIPELINE
    # -------------------------------------------------
    def run(self):
        self.get_header_coords()
        self.download_gaia_patch()
        self.generate_healpix_tiles()
        self.build_index_files()
        self.solve_with_astrometry()

# from dynamic_astrometry import DynamicAstrometry

# ast = DynamicAstrometry("bf_2059_0016_V.fits")
# ast.solve_with_astrometry()
