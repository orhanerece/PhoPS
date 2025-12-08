#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.io import fits

fits_file = "atlas.fit"
#
# # FITS dosyasını aç, değişiklik yapmak için 'update' modunda
# with fits.open(fits_file, mode='update') as hdul:
#     hdr = hdul[0].header  # primary header
#
#     # TEMPERAT keyword'ünü sil
#     if "NOTES" in hdr:
#         del hdr["NOTES"]
#         print("✅ TEMPERAT silindi.")
#
#
# exit()
import os
import subprocess
from astroquery.gaia import Gaia
from astropy.table import Table
from astropy.io import fits
def ra_to_deg(ra_str):
    """RA 'HH:MM:SS' → derece"""
    h, m, s = [float(x) for x in ra_str.split(":")]
    return 15 * (h + m/60 + s/3600)


def dec_to_deg(dec_str):
    """DEC '±DD:MM:SS' → derece"""
    parts = dec_str.split(":")
    d = float(parts[0])
    m = float(parts[1])
    s = float(parts[2])

    sign = 1
    if d < 0:
        sign = -1
        d = abs(d)

    return sign * (d + m/60 + s/3600)

# ---------------------------------------------------
# Girdi parametreleri (Senin değerlerin)
# ---------------------------------------------------
solve_image = "atlas.fit"   # Çözülecek görüntü
solve_dir = "solve_output"
index_dir = "output_index"

with fits.open(solve_image) as hdul:
    hdr = hdul[0].header
    print(hdr)
    RA_CENTER = ra_to_deg(hdr.get("RA", hdr.get("RA")))
    DEC_CENTER = dec_to_deg(hdr.get("DEC", hdr.get("DEC")))

RADIUS_DEG = 0.5   # Gaia patch yarıçapı

# Solve-field parametreleri
pixel_scale = 0.4      # arcsec/pixel
pixel_scale_low  = pixel_scale * 0.7
pixel_scale_high = pixel_scale * 1.4


# ---------------------------------------------------
# 1) GAIA ADQL QUERY
# ---------------------------------------------------

ADQL_QUERY = f"""
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
    CIRCLE('ICRS', {RA_CENTER}, {DEC_CENTER}, {RADIUS_DEG})
) = 1
ORDER BY phot_g_mean_mag
"""

# ---------------------------------------------------
# 2) Klasörler
# ---------------------------------------------------

os.makedirs("input_cat", exist_ok=True)
os.makedirs("gaia_hp", exist_ok=True)
os.makedirs("output_index", exist_ok=True)
os.makedirs(solve_dir, exist_ok=True)

PATCH_FILE = "input_cat/gaia_patch.fits"

# ---------------------------------------------------
# 3) GAIA PATCH İNDİR
# ---------------------------------------------------

print("🔭 Gaia sorgusu gönderiliyor...")
job = Gaia.launch_job_async(ADQL_QUERY)
gaia_data = job.get_results()

print(f"💾 {len(gaia_data)} Gaia kaynağı indirildi.")
gaia_data.write(PATCH_FILE, overwrite=True)
print(f"💾 Gaia patch kaydedildi: {PATCH_FILE}")

# ---------------------------------------------------
# 4) HPSPLIT – HEALPIX TILES
# ---------------------------------------------------

print("\n🧩 HEALPix tile'lar oluşturuluyor...")

cmd_hpsplit = [
    "hpsplit",
    "-o", "gaia_hp/gaia-hp%02i.fits",
    "-n", "2",              # NSIDE = 2
    PATCH_FILE
]

subprocess.run(cmd_hpsplit, check=True)
print("✅ hpsplit tamamlandı.")

# ---------------------------------------------------
# 5) TILE listesini al
# ---------------------------------------------------

tile_files = sorted([f for f in os.listdir("gaia_hp") if f.endswith(".fits")])
tile_ids = [int(f.split("gaia-hp")[1].split(".")[0]) for f in tile_files]

print(f"📦 Bulunan HEALPix tile ID'leri: {tile_ids}")

# ---------------------------------------------------
# 6) BUILD ASTROMETRY INDEX FILES
# ---------------------------------------------------

quad_scales = [0, 2, 4, 6]

print("\n⚙ Index üretimi başlıyor...\n")

for tile_id in tile_ids:
    tile_path = f"gaia_hp/gaia-hp{tile_id:02d}.fits"

    for P in quad_scales:
        SS = f"{P:02d}"
        output_index = f"{index_dir}/index-550{SS}-{tile_id:02d}.fits"
        index_id = f"550{SS}{tile_id:02d}"

        cmd_index = [
            "build-astrometry-index",
            "-i", tile_path,
            "-s", "2",                   # NSIDE=2
            "-P", str(P),
            "-E",
            "-S", "phot_g_mean_mag",
            "-o", output_index,
            "-I", index_id
        ]

        print(f"➡️ Tile {tile_id:02d}, P={P}: {output_index}")
        subprocess.run(cmd_index, check=True)

print("\n🎉 TÜM index dosyaları oluşturuldu!")

#---------------------------------------------------
# 7) SOLVE-FIELD ÇALIŞTIR
# ---------------------------------------------------

print("\n🔎 solve-field çalıştırılıyor...")

cmd_solve = [
    "solve-field",
    solve_image,
    "--dir", solve_dir,
    "--scale-units", "arcsecperpix",
    "--scale-low", str(pixel_scale_low),
    "--scale-high", str(pixel_scale_high),
    "--overwrite",
    "--no-plots",
    "--index-dir", index_dir
]

print("➡️ Komut:", " ".join(cmd_solve))
subprocess.run(cmd_solve, check=True)

print("\n✅ solve-field tamamlandı!")
print(f"📁 Çıkış dizini: {solve_dir}")
