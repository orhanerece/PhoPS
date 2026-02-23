import os
import subprocess
import math
from astropy.io import fits
from astroquery.gaia import Gaia
from core.utils import ra_to_deg, dec_to_deg


class AstrometrySolver:
    """
    Handles the astrometric plate solving process by creating local Gaia indexes
    and using the astrometry.net solve-field engine.
    """

    def __init__(self, config):
        self.cfg = config

        # Ensure required directories exist
        for path in self.cfg['paths'].values():
            os.makedirs(path, exist_ok=True)

        # Path for the index verification key (Using .txt for cross-platform compatibility)
        self.key_file = os.path.join(self.cfg['paths']['index_dir'], "generated_index.txt")

    def _get_all_key_coords(self):
        """Read all coordinate pairs from the .txt key file."""
        coords_list = []
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    for line in f:
                        if line.strip():
                            ra, dec = line.strip().split(",")
                            coords_list.append((float(ra), float(dec)))
            except Exception as e:
                print(f"⚠️ Error reading key file: {e}")
        return coords_list

    def _add_key(self, ra, dec):
        """Append new coordinates to the .txt key file."""
        with open(self.key_file, "a") as f:  # Open in append mode
            f.write(f"{ra},{dec}\n")

    def _is_same_area(self, ra, dec):
        """Check if current image matches any area stored in the key file."""
        all_keys = self._get_all_key_coords()
        if not all_keys:
            return False

        tolerance = self.cfg['astrometry'].get('cache_tolerance', 0.1)

        for k_ra, k_dec in all_keys:
            dist = math.sqrt((ra - k_ra) ** 2 + (dec - k_dec) ** 2)
            if dist < tolerance:
                return True
        return False

    def _prepare_gaia_index(self, ra, dec, override_patch_path=None):
        """Query Gaia catalog, split into HEALPix tiles, and build index files."""

        # Area suffix'i belirle (Hala index dosyaları için lazım)
        area_suffix = f"{int(ra)}_{int(dec)}"
        print(f"🔭 New area detected. Generating indexes: RA={ra:.3f}, Dec={dec:.3f}")

        # 1. Gaia Query
        radius = self.cfg['astrometry']['radius']
        query = f"""
        SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, bp_rp
        FROM {self.cfg['catalog']}
        WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
        ORDER BY phot_g_mean_mag
        """
        job = Gaia.launch_job_async(query)
        data = job.get_results()

        # --- KRİTİK DÜZELTME BURASI ---
        # Eğer dışarıdan bir yol (main.py'den) gelmişse onu kullan
        if override_patch_path:
            patch_path = override_patch_path
        else:
            patch_path = os.path.join(self.cfg['paths']['temp_dir'], f"gaia_patch_{area_suffix}.fits")

        data.write(patch_path, overwrite=True)

        # 2. HPSplit (HEALPix bölme işlemi)
        # hpsplit geçici dosyaları için de unique bir isim şablonu kullanalım
        hp_out = os.path.join(self.cfg['paths']['temp_dir'], f"gaia-hp%02i_{area_suffix}.fits")
        subprocess.run(["hpsplit", "-o", hp_out, "-n", "2", patch_path], check=True)

        # 3. Build-Index Loop
        tile_files = sorted([f for f in os.listdir(self.cfg['paths']['temp_dir'])
                             if f.startswith("gaia-hp") and f.endswith(f"_{area_suffix}.fits")])

        for tile_f in tile_files:
            # Örn: gaia-hp01_316_8.fits -> tile_id_str = "01"
            tile_id_str = tile_f.split("gaia-hp")[1].split("_")[0]
            tile_path = os.path.join(self.cfg['paths']['temp_dir'], tile_f)

            for p in self.cfg['astrometry']['quad_scales']:
                output_index = os.path.join(self.cfg['paths']['index_dir'],
                                            f"index-550{p:02d}-{tile_id_str}_{area_suffix}.fits")

                # Index ID'nin benzersiz olması için sonuna koordinat bilgisinden bir parça ekliyoruz
                index_id = f"550{p:02d}{tile_id_str}{abs(int(ra))}"

                subprocess.run([
                    "build-astrometry-index", "-i", tile_path, "-s", "2",
                    "-P", str(p), "-E", "-S", "phot_g_mean_mag",
                    "-o", output_index, "-I", index_id
                ], check=True, stdout=subprocess.DEVNULL)

        # 4. Geçici HEALPix tile dosyalarını temizle (Ana patch dosyasını match işlemi için tutuyoruz)
        for tile_f in tile_files:
            os.remove(os.path.join(self.cfg['paths']['temp_dir'], tile_f))

        self._add_key(ra, dec)
        return patch_path  # Yeni oluşturulan patch yolunu dönebilirsin

    def solve(self, image_path):
        """Main method to perform plate solving and clean up auxiliary files."""
        with fits.open(image_path) as hdul:
            hdr = hdul[0].header
            ra_k = self.cfg['fits_keywords'].get('ra_key', 'RA')
            dec_k = self.cfg['fits_keywords'].get('dec_key', 'DEC')

            ra_val = hdr.get(ra_k)
            dec_val = hdr.get(dec_k)

            if ra_val is None or dec_val is None:
                raise KeyError(f"Keywords '{ra_k}'/'{dec_k}' not found in FITS header.")

            ra_img = ra_to_deg(ra_val)
            dec_img = dec_to_deg(dec_val)

        # Cache check against all known areas
        if not self._is_same_area(ra_img, dec_img):
            self._prepare_gaia_index(ra_img, dec_img)
        else:
            print(f"♻️ Area already indexed. Using stored indexes from generated_index.txt")

        # Define output paths
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_fits = os.path.join(self.cfg['paths']['solve_dir'], f"{base_name}_new.fits")
        ps = self.cfg['instrument']['pixel_scale']

        cmd = [
            "solve-field", image_path,
            "--dir", self.cfg['paths']['solve_dir'],
            "--new-fits", output_fits,
            "--scale-units", "arcsecperpix",
            "--scale-low", str(ps * 0.7),
            "--scale-high", str(ps * 1.4),
            "--index-dir", self.cfg['paths']['index_dir'],
            "--overwrite",
            "--no-plots",
            "--solved", "none",
            "--corr", "none",
            "--rdls", "none",
            "--match", "none",
            "--index-xyls", "none",
            "--axy", "none"
        ]

        print(f"🔎 Solving: {image_path}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        # Final Cleanup of unwanted .wcs files
        wcs_temp = os.path.join(self.cfg['paths']['solve_dir'], f"{base_name}.wcs")
        if os.path.exists(wcs_temp):
            os.remove(wcs_temp)

        print(f"✅ Success: {output_fits} created.")