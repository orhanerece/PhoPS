import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_com, centroid_quadratic
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt


class Photometry:
    def __init__(self, config):
        self.cfg = config
        self.pxscale = self.cfg['instrument']['pixel_scale']

        # Detection & Filtering Parameters
        self.sat_limit = self.cfg['instrument'].get('saturation_level', 60000)
        self.fwhm_guess = self.cfg['source_detection'].get('fwhm_guess', 3.0)
        self.thresh_sigma = self.cfg['source_detection'].get('threshold_sigma', 5.0)
        self.edge_margin = self.cfg['source_detection'].get('edge_margin', 30)

        # Matching Parameters
        self.iso_radius = self.cfg['matching'].get('isolation_radius_arcsec', 2.0)
        self.match_dist = self.cfg['matching'].get('match_constraint_arcsec', 1.0)

    def get_clean_gaia_matches(self, image_path, gaia_catalog_path):
        """
        Main workflow: Detects, filters for quality, and matches with Gaia.
        """
        with fits.open(image_path) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header
            wcs = WCS(header)
        threshold = self.cfg['source_detection'].get('threshold_sigma', 1.0)

        # 1. Detection
        std = np.median(data)
        median = np.median(data)
        finder = DAOStarFinder(fwhm=self.fwhm_guess, threshold=std*threshold, exclude_border=True,)
        image_sources = finder(data - median)

        if image_sources is None:
            return None

        # 2. Quality Filtering (Saturation + Edges)
        ny, nx = data.shape
        quality_mask = (image_sources['peak'] < self.sat_limit) & \
                       (image_sources['xcentroid'] > self.edge_margin) & \
                       (image_sources['xcentroid'] < (nx - self.edge_margin)) & \
                       (image_sources['ycentroid'] > self.edge_margin) & \
                       (image_sources['ycentroid'] < (ny - self.edge_margin))

        sources = image_sources[quality_mask]

        # 3. Isolation Filtering
        iso_px = self.iso_radius / self.pxscale
        coords = np.transpose((sources['xcentroid'], sources['ycentroid']))

        final_idx = []
        for i, pt in enumerate(coords):
            dists = np.sqrt(np.sum((coords - pt) ** 2, axis=1))
            if np.sum(dists < iso_px) == 1:
                final_idx.append(i)
        clean_sources = sources[final_idx]

        # 4. Gaia Matching
        gaia_table = Table.read(gaia_catalog_path)
        gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit='deg')

        ra_img, dec_img = wcs.all_pix2world(clean_sources['xcentroid'], clean_sources['ycentroid'], 0)
        img_coords = SkyCoord(ra=ra_img, dec=dec_img, unit='deg', frame='icrs')

        idx, d2d, _ = img_coords.match_to_catalog_sky(gaia_coords)
        match_mask = d2d < (self.match_dist * u.arcsec)

        # Create final matched table
        matched_table = clean_sources[match_mask]
        matched_table['img_ra'] = ra_img[match_mask]
        matched_table['img_dec'] = dec_img[match_mask]
        matched_table['gaia_ra'] = gaia_table['ra'][idx[match_mask]]
        matched_table['gaia_dec'] = gaia_table['dec'][idx[match_mask]]
        matched_table['gaia_gmag'] = gaia_table['phot_g_mean_mag'][idx[match_mask]]
        matched_table['bp_rp'] = gaia_table['bp_rp'][idx[match_mask]]

        # Calculate r_dist for Zeropoint(r)
        matched_table['r_dist'] = np.sqrt((matched_table['xcentroid'] - nx / 2) ** 2 +
                                          (matched_table['ycentroid'] - ny / 2) ** 2)

        import matplotlib.pyplot as plt

        # ... fonksiyonun sonuna, return'den hemen önce ekle ...

        plt.figure(figsize=(12, 12))

        # Görüntüyü arka plan olarak çiz
        # vmin ve vmax değerlerini (median-std) ve (median+5*std) yaparak yıldızları daha belirgin kılabilirsin
        vmin, vmax = np.percentile(data, [50, 99])  # Daha stabil bir kontrast için percentile kullandık

        plt.imshow(data, cmap='gray', origin='lower',
                   vmin=vmin,
                   vmax=vmax)

        # 1. Görüntüde bulunan TÜM kaynakları çiz (Mavi küçük noktalar)
        plt.scatter(image_sources['xcentroid'], image_sources['ycentroid'],
                    edgecolor='blue', facecolor='none', s=20, alpha=0.5, label='Tespit Edilen Tüm Kaynaklar')

        # 2. Gaia ile EŞLEŞEN temiz kaynakları çiz (Kırmızı büyük halkalar)
        plt.scatter(matched_table['xcentroid'], matched_table['ycentroid'],
                    edgecolor='red', facecolor='none', s=100, lw=1.5, label='Gaia ile Eşleşen Temiz Kaynaklar')

        plt.title(f"Kaynak Analizi: {len(image_sources)} Tespit / {len(matched_table)} Gaia Eşleşmesi")
        plt.xlabel("X (pixel)")
        plt.ylabel("Y (pixel)")
        plt.legend(loc='upper right')

        # İstersen bu grafiği kaydetmesi için:
        plt.savefig("matching_test_result.png", dpi=300)
        plt.close()

        # plt.show()
        return matched_table, data, image_sources  # image_sources is for plotting only

    def transform_gaia_to_filter(self, matched_table):
        """
        Step 3: Transforms Gaia G-mag and BP-RP color to a standard photometric band.
        """
        target_filter = self.cfg['photometry'].get('filter', 'V')

        coeffs = {
            'B': [0.01448, -0.6874, -0.3604, 0.06718, -0.006061],
            'V': [-0.02704, 0.01424, -0.2156, 0.01426, 0.0],
            'R': [-0.02275, 0.3961, -0.1243, -0.01396, 0.003775],
            'I': [0.01753, 0.76, -0.0991, 0.0, 0.0],
            'g': [0.2199, -0.6365, -0.1548, 0.0064, 0.0],
            'r': [-0.09837, 0.08592, 0.1907, -0.1701, 0.02263],
            'i': [-0.293, 0.6404, -0.09609, -0.002104, 0.0],
            'z': [-0.4619, 0.8992, -0.08271, 0.005029, 0.0]
        }

        if target_filter not in coeffs:
            print(f"⚠️ Warning: Filter '{target_filter}' not recognized. Using Gaia G-mag.")
            matched_table['standard_mag'] = np.array(matched_table['gaia_gmag'], dtype=float)
            return matched_table

        c = coeffs[target_filter]

        # --- KRİTİK DÜZELTME: Birimlerden kurtulma ---
        # Verileri açıkça float numpy array'ine çeviriyoruz (.value veya np.array kullanarak)
        bprp = np.array(matched_table['bp_rp'], dtype=float)
        g_mag = np.array(matched_table['gaia_gmag'], dtype=float)

        # Polinom hesabı artık tamamen birimsiz sayılarla yapılacak
        delta_mag = (c[0] +
                     c[1] * bprp +
                     c[2] * (bprp ** 2) +
                     c[3] * (bprp ** 3) +
                     c[4] * (bprp ** 4))

        matched_table['standard_mag'] = g_mag - delta_mag

        # NaN kontrolü (BP-RP olmayan yıldızlar için)
        nan_mask = ~np.isnan(matched_table['standard_mag'])
        final_table = matched_table[nan_mask]

        print(f"✨ Successfully transformed {len(final_table)} stars to '{target_filter}' band.")
        return final_table

    def perform_aperture_photometry(self, data, matched_table, image_sources):
        """
        Step 4: Görüntüdeki referans yıldızlar için hassas merkezleme ve
        aperture fotometrisi. calculate_radii fonksiyonunu kullanır.
        """
        # 1. Görüntünün genel medyan FWHM değerini hesapla
        if 'fwhm' in image_sources.colnames:
            median_fwhm = np.median(image_sources['fwhm'])
        else:
            # FWHM yoksa npix üzerinden kaba bir tahmin yap (Area -> FWHM dönüşümü)
            # 2 * sqrt(npix / pi)
            estimated_fwhms = 2.0 * np.sqrt(image_sources['npix'] / np.pi)
            median_fwhm = np.median(estimated_fwhms)

        print(f"ℹ️ Median FWHM estimated as: {median_fwhm:.2f} px")

        # 2. Yarıçapları merkezi fonksiyondan al (Birim dönüşümleri burada yapılır)
        r, r_in, r_out = self.calculate_radii(median_fwhm)

        print(f"✨ Photometry Radii: r={r:.2f}, r_in={r_in:.2f}, r_out={r_out:.2f} (pixels)")

        # 3. Hassas Centroiding (Sub-pixel refinement)
        refined_positions = []
        box_size = int(median_fwhm * 2)
        if box_size % 2 == 0: box_size += 1

        for row in matched_table:
            x_init, y_init = row['xcentroid'], row['ycentroid']

            y_min, y_max = int(y_init - box_size // 2), int(y_init + box_size // 2 + 1)
            x_min, x_max = int(x_init - box_size // 2), int(x_init + box_size // 2 + 1)

            if y_min < 0 or x_min < 0 or y_max > data.shape[0] or x_max > data.shape[1]:
                refined_positions.append([x_init, y_init])
                continue

            cutout = data[y_min:y_max, x_min:x_max]
            try:
                cutout_clean = cutout - np.median(cutout)
                dx, dy = centroid_quadratic(cutout_clean)
                refined_positions.append([x_min + dx, y_min + dy])
            except:
                refined_positions.append([x_init, y_init])

        refined_positions = np.array(refined_positions)
        nan_mask = np.isnan(refined_positions).any(axis=1)
        if np.any(nan_mask):
            print(f"⚠️ {np.sum(nan_mask)} yıldızda merkezleme hatası (NaN) saptandı, eleniyor.")
            # Hem pozisyonları hem de tablodaki karşılıklarını temizle
            refined_positions = refined_positions[~nan_mask]
            matched_table = matched_table[~nan_mask]

        matched_table['x_precise'] = refined_positions[:, 0]
        matched_table['y_precise'] = refined_positions[:, 1]

        # 4. Aperture ve Annulus Nesnelerini Oluştur
        apertures = CircularAperture(refined_positions, r=r)
        annulus_apertures = CircularAnnulus(refined_positions, r_in=r_in, r_out=r_out)

        # 5. Fotometri ve Arka Plan Çıkarma

        raw_phot = aperture_photometry(data, [apertures, annulus_apertures])

        annulus_masks = annulus_apertures.to_mask(method='center')
        bkg_medians = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(data)
            if annulus_data is not None:
                annulus_data_1d = annulus_data[mask.data > 0]
                bkg_medians.append(np.median(annulus_data_1d))
            else:
                bkg_medians.append(0.0)

        bkg_medians = np.array(bkg_medians)
        net_flux = raw_phot['aperture_sum_0'] - (bkg_medians * apertures.area)
        net_flux[net_flux <= 0] = np.nan

        # 6. Magnitüd ve Hata Hesapları
        # gain ve read_noise init kısmında tanımlanmış olmalı (self.gain, self.read_noise)
        matched_table['inst_mag'] = 25.0 - 2.5 * np.log10(net_flux)

        gain = self.cfg['instrument'].get('gain', 1.0)
        rn = self.cfg['instrument'].get('read_noise', 5.0)

        noise = np.sqrt(net_flux / gain +
                        (apertures.area * bkg_medians) / gain +
                        (apertures.area * rn ** 2))

        matched_table['mag_err'] = 1.0857 * (noise / net_flux)
        matched_table['snr'] = net_flux / noise

        print(f"📸 Photometry complete. Sources: {len(matched_table)}")

        return matched_table, median_fwhm

    def calculate_zeropoint_model(self, matched_table, plot=True, save_plot=True, output_path=None):
        """
        Step 5 & 6: Calculates field-dependent zeropoint using RANSAC.
        """
        # --- KRİTİK DÜZELTME: .values yerine doğrudan sütun adlarını kullanıyoruz ---
        # Astropy Column nesnesini Numpy dizisine zorlamak için np.array() kullanmak en güvenlisidir
        mag_diff = np.array(matched_table['standard_mag'] - matched_table['inst_mag'])
        r_dist = np.array(matched_table['r_dist'])

        # NaN değerleri temizleyelim (Her ihtimale karşı)
        valid_mask = ~np.isnan(mag_diff)
        mag_diff = mag_diff[valid_mask]
        r_dist = r_dist[valid_mask]

        if len(mag_diff) < 4:
            print("⚠️ Zeropoint için yeterli geçerli yıldız kalmadı.")
            # Güvenli bir fallback (boş model dönebiliriz)
            return np.poly1d([0, 25.0]), 0, 0

        # Scikit-learn 2D array bekler
        X = r_dist.reshape(-1, 1)
        y = mag_diff.reshape(-1, 1)

        # ... (RANSAC kısımları aynı kalıyor) ...

        # 2. RANSAC Regression
        # residual_threshold: 0.15 mag limit for being an inlier
        ransac = RANSACRegressor(residual_threshold=0.15, random_state=42)
        ransac.fit(X, y)

        inlier_mask = ransac.inlier_mask_
        outlier_mask = ~inlier_mask

        # 3. Final Linear Model on Inliers
        # ZP(r) = slope * r + intercept
        slope = ransac.estimator_.coef_[0][0]
        intercept = ransac.estimator_.intercept_[0]
        zp_function = np.poly1d([slope, intercept])

        # Calculate RMS error for inliers
        rms_error = np.std(y[inlier_mask] - zp_function(X[inlier_mask]))

        print(f"✅ Zeropoint Calibration: ZP(r) = {slope:.6f}*r + {intercept:.4f}")
        print(f"📊 Statistics: {np.sum(inlier_mask)} inliers, RMS: {rms_error:.4f} mag")

        # 4. Plotting
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(r_dist[inlier_mask], mag_diff[inlier_mask], color='blue', alpha=0.6, label='Inliers')
            plt.scatter(r_dist[outlier_mask], mag_diff[outlier_mask], color='red', alpha=0.3, label='Outliers')

            # Plot the model line
            r_range = np.linspace(0, np.max(r_dist), 100)
            plt.plot(r_range, zp_function(r_range), color='black', linestyle='--', label=f'Model (Slope: {slope:.6f})')

            plt.xlabel("Distance from Image Center (pixels)")
            plt.ylabel("Standard - Instrumental Mag (ZP)")
            plt.title("RANSAC Field-Dependent Zeropoint")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_plot and output_path:
                plt.savefig(output_path.replace('.fits', '_zp.png'))
            plt.close()
            plt.show()

        return zp_function, rms_error, slope, np.average(mag_diff[inlier_mask])

    # core/photometry.py içine eklenecek/güncellenecek kısımlar

    def calculate_radii(self, median_fwhm):
        """
        Config dosyasındaki birime göre r, r_in ve r_out değerlerini
        piksel cinsinden hesaplar.
        """
        p_cfg = self.cfg['photometry']
        method = p_cfg.get('aperture_method', 'fwhm_factor')

        # Değerleri config'den çek
        val = float(p_cfg.get('aperture', 2.5))
        ann_in_val = float(p_cfg.get('annulus_inner', 4.0))
        ann_out_val = float(p_cfg.get('annulus_outer', 6.0))

        if method == "fwhm_factor":
            r = median_fwhm * val
            r_in = median_fwhm * ann_in_val
            r_out = median_fwhm * ann_out_val
        elif method == "fixed_arcsec":
            r = val / self.pxscale
            r_in = r + ann_in_val / self.pxscale
            r_out = r + ann_out_val / self.pxscale
        elif method == "fixed_pixel":
            r = val
            r_in = ann_in_val
            r_out = ann_out_val
        else:
            # Varsayılan (fallback)
            r = median_fwhm * 2.5
            r_in = median_fwhm * 4.0
            r_out = median_fwhm * 6.0

        # Mantıksal güvenlik: r_in her zaman r'den büyük olmalı
        if r_in <= r:
            r_in = r + 2.0
            r_out = r_in + 5.0
        return r, r_in, r_out

    def save_target_cutout(self, data, x, y, filename, output_dir="test/"):
        """
        Hedef kaynağın etrafında 50x50 piksellik bir kesit alır ve
        üzerine hedefi işaretleyerek PNG olarak kaydeder.
        """
        import matplotlib.pyplot as plt
        from astropy.nddata import Cutout2D
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        size = (200, 200)
        try:
            cutout = Cutout2D(data, (x, y), size, mode='partial', fill_value=0)

            fig, ax = plt.subplots(figsize=(4, 4))
            im_data = cutout.data

            # Kontrast ayarı
            vmin, vmax = np.percentile(im_data, [50, 99])  # Daha stabil bir kontrast için percentile kullandık
            ax.imshow(im_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

            cx, cy = cutout.input_position_cutout
            circle = plt.Circle((cx, cy), radius=5, color='cyan', fill=False, lw=1.5)
            ax.add_patch(circle)

            ax.set_title(f"{filename}", fontsize=8)
            ax.axis('off')

            # --- KRİTİK DÜZELTME: Dosya ismindeki noktaları güvenli hale getir ---
            # Dosya isminden .fits'i atıp, koordinatlardaki noktaları alt tire yapıyoruz
            safe_fname = str(filename).replace('.fits', '').replace('.', '_')
            save_path = os.path.join(output_dir, f"cutout_{safe_fname}.png")  # Uzantıyı sona ekle

            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Cutout kaydedilemedi ({filename}): {e}")
    def measure_target(self, data, wcs, ra, dec, zp_function, median_fwhm, all_detected, filename, zp_average):
        """
        Hedef cismin (Asteroid/Yıldız) koordinatlarını bulur ve
        ZP modelini uygulayarak fotometri yapar.
        """
        # 1. Koordinat Dönüşümü (RA/Dec -> Pixel)
        x_target, y_target = wcs.all_world2pix(ra, dec, 0)

        if all_detected is not None:
            iso_px = self.iso_radius / self.pxscale
            all_coords = np.transpose((all_detected['xcentroid'], all_detected['ycentroid']))
            dists = np.sqrt(np.sum((all_coords - [x_target, y_target]) ** 2, axis=1))
            nearby_mask = (dists < iso_px+50)
            nearby_count = np.sum(nearby_mask)
            nearby_sources = all_detected[nearby_mask]

            if nearby_count > 1:
                neighbor_ids = nearby_sources['id'] if 'id' in nearby_sources.colnames else "N/A"

                print(f"⚠️ Hedef izole değil! {iso_px:.1f} px içinde {nearby_count} komşu var.")
                print(f"🚫 Yakın kaynak ID(ler)i: {neighbor_ids}")

                return None
            elif nearby_count == 1:
                x_target, y_target = nearby_sources["xcentroid"][0], nearby_sources["ycentroid"][0]

        # 2. Yarıçapları Hesapla (Birim tutarlılığı burada sağlanıyor)
        r, r_in, r_out = self.calculate_radii(median_fwhm)
        # 3. Hassas Centroiding (Merkezleme)
        box_size = int(median_fwhm * 2)
        if box_size % 2 == 0: box_size += 1

        y_min, y_max = int(y_target - box_size // 2), int(y_target + box_size // 2 + 1)
        x_min, x_max = int(x_target - box_size // 2), int(x_target + box_size // 2 + 1)
        # Görüntü sınırı kontrolü
        if y_min < 0 or x_min < 0 or y_max > data.shape[0] or x_max > data.shape[1]:
            return None

        cutout = data[y_min:y_max, x_min:x_max]
        cutout_clean = cutout - np.median(cutout)

        try:
            dx, dy = centroid_quadratic(cutout_clean)

            if not (np.isfinite(dx) and np.isfinite(dy)):
                raise ValueError("Centroid returned non-finite values")

            x_precise, y_precise = x_min + dx, y_min + dy

        except Exception:
            x_precise, y_precise = x_target, y_target

        # 4. Fotometri Uygula
        pos = [(x_precise, y_precise)]
        aperture = CircularAperture(pos, r=r)
        annulus = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
        phot_table = aperture_photometry(data, [aperture, annulus])

        # Arka plan hesabı
        annulus_mask = annulus.to_mask(method='center')[0]
        annulus_data = annulus_mask.multiply(data)
        bkg_median = np.median(annulus_data[annulus_mask.data > 0])

        net_flux = phot_table['aperture_sum_0'][0] - (bkg_median * aperture.area)
        if net_flux <= 0: return None

        # 5. Zeropoint Uygulama
        # Cismin merkezden uzaklığını bul (ZP modelimiz r_dist'e bağlıydı)
        ny, nx = data.shape
        r_dist_target = np.sqrt((x_precise - nx / 2) ** 2 + (y_precise - ny / 2) ** 2)
        zp = self.cfg['photometry'].get('zeropoint', "fit")

        if zp == "average":
            target_zp = zp_average
        else:
            target_zp = zp_function(r_dist_target)

        inst_mag = 25.0 - 2.5 * np.log10(net_flux)
        final_mag = inst_mag + target_zp

        # Hata Hesabı
        gain = self.cfg['instrument'].get('gain', 1.0)
        rn = self.cfg['instrument'].get('read_noise', 5.0)
        noise = np.sqrt(net_flux / gain + (aperture.area * bkg_median) / gain + (aperture.area * rn ** 2))
        mag_err = 1.0857 * (noise / net_flux)
        self.save_target_cutout(data, x_precise, y_precise, f"target_{filename}")

        return {
            'ra': ra, 'dec': dec,
            'x': x_precise, 'y': y_precise,
            'inst_mag': inst_mag, 'zp': target_zp,
            'mag': final_mag, 'err': mag_err,
            'snr': net_flux / noise
        }