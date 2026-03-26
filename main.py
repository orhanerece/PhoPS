import os
import yaml
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from core.utils import ra_to_deg, dec_to_deg, plot_astrometry_residuals, calculate_residuals
from core.astrometry import AstrometrySolver
from core.photometry import Photometry
from core.target import TargetManager


def run_pipeline():
    # --- 1. CONFIG VE YOLLAR ---
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    path_cfg = cfg.get('paths', {})
    input_dir = path_cfg.get('input_dir', 'files')
    solve_dir = path_cfg.get('solve_dir', 'output')
    temp_dir = path_cfg.get('temp_dir', 'data/temp')
    file_name = path_cfg.get('file_name', 'test.csv')
    file_ext = path_cfg.get('file_extension', 'fits')

    # --- 2. ÇIKTI DOSYALARI (Checkpoint Hazırlığı) ---
    # Fotometri sonuçları için dosya
    final_photometry_csv = os.path.join(solve_dir, file_name)
    # Astrometri (tüm yıldızlar) için dosya
    all_stars_astrometry_csv = os.path.join(solve_dir, "astrometry_all_stars_yayin_dosyasi_star_testx.csv")

    # Temiz bir başlangıç için eski dosyaları silebilirsin (opsiyonel)
    for f in [final_photometry_csv, all_stars_astrometry_csv]:
        if os.path.exists(f): os.remove(f)

    # --- 3. MOTORLARI BAŞLAT ---
    solver = AstrometrySolver(cfg)
    photometry = Photometry(cfg)
    target_mgr = TargetManager(cfg)

    raw_files = sorted(glob.glob(os.path.join(input_dir, f"*.{file_ext}")))
    print(f"🚀 Pipeline started. Processing {len(raw_files)} files.")

    # --- 4. ANA DÖNGÜ ---
    for img_path in raw_files:
        fname = os.path.basename(img_path)
        print(f"\n{'-' * 40}\n📂 File: {fname}")

        # ADIM A: Astrometri Çözümü
        solver.solve(img_path)
        base_name = os.path.splitext(fname)[0]
        solved_fits = os.path.join(solve_dir, f"{base_name}_new.fits")

        if not os.path.exists(solved_fits):
            print("⏩ Astrometric solution failed, skipping.")
            continue

        # ADIM B: Veri ve WCS Yükleme
        with fits.open(solved_fits) as hdul:
            header = hdul[0].header
            data = hdul[0].data.astype(float)
            wcs = WCS(header)

        # ADIM C: Gaia Patch Kontrolü
        ra_img = ra_to_deg(header[cfg['fits_keywords'].get('ra_key', 'RA')])
        dec_img = dec_to_deg(header[cfg['fits_keywords'].get('dec_key', 'DEC')])
        area_suffix = f"{int(ra_img)}_{int(dec_img)}"
        gaia_patch = os.path.join(temp_dir, f"gaia_patch_{area_suffix}.fits")

        if not os.path.exists(gaia_patch):
            print("🔍 Generating Gaia patch file...")
            solver._prepare_gaia_index(ra_img, dec_img, override_patch_path=gaia_patch)

        # ADIM D: Yıldız Eşleştirme ve Astrometri Kaydı
        matched_stars, _, all_detected = photometry.get_clean_gaia_matches(solved_fits, gaia_patch)

        if matched_stars is not None and len(matched_stars) >= 5:
            # Astrometrik farkları hesapla
            dra_corr, ddec, g_magnitudes = calculate_residuals(matched_stars)  # Yardımcı fonksiyon aşağıda

            # Anlık Astrometri Kaydı (Append)
            astrom_df = pd.DataFrame(
                {'filename': [fname] * len(dra_corr), 'dra_corr': dra_corr, 'ddec': ddec, 'gmag': g_magnitudes})
            astrom_df.to_csv(all_stars_astrometry_csv, mode='a', header=not os.path.exists(all_stars_astrometry_csv),
                             index=False)
            print(f"✅ Saved {len(matched_stars)} matched stars.")
        else:
            print("⚠️ Not enough stars detected; skipping frame.")
            continue

        # ADIM E: Zeropoint ve Hedef Ölçümü
        matched_stars = photometry.transform_gaia_to_filter(matched_stars)
        matched_stars, median_fwhm = photometry.perform_aperture_photometry(data, matched_stars, all_detected)
        zp_func, zp_scatter, zp_slope, zp_average = photometry.calculate_zeropoint_model(matched_stars, save_plot=True,
                                                                                     output_path=fname)

        ra_t, dec_t, phys = target_mgr.get_target_coordinates(header)
        target_res = photometry.measure_target(data, wcs, ra_t, dec_t, zp_func, median_fwhm, all_detected, base_name,
                                               zp_average)

        if target_res:
            # Sonuç Satırı Oluştur
            res = {
                'filename': fname,
                'jd': phys['jd'] if phys else target_mgr.get_jd_time(header),
                'mag_inst': target_res['mag_inst'],
                'mag_calib': target_res['mag_calib'],
                'snr': target_res['snr'],
                'mag_err': target_res['err'],
                'x_target': target_res['x'],
                'y_target': target_res['y'],
                'bg': target_res['BG'],
                'zp': target_res['zp'],
                'zp_scatter': zp_scatter,
                'fwhm': median_fwhm
            }
            if phys:
                res.update(
                    {'reduced_mag': target_res['mag_calib'] - 5 * np.log10(phys['r'] * phys['delta']), 'r_au': phys['r'], 'delta_au': phys['delta']})

            # Anlık Fotometri Kaydı (Append)
            pd.DataFrame([res]).to_csv(final_photometry_csv, mode='a', header=not os.path.exists(final_photometry_csv),
                                       index=False)
            print(f"🎯 Target magnitude measured: {target_res['mag_calib']:.3f} mag")
            break
    # --- 5. FİNAL GRAFİK ---
    print(f"\n✨ Processing complete. Reports saved in '{solve_dir}'.")
    if cfg['plots'].get('plot_astrometry', True):
        plot_astrometry_residuals(all_stars_astrometry_csv, solve_dir)


if __name__ == "__main__":
    run_pipeline()
