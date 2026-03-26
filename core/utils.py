from astropy.coordinates import SkyCoord
import astropy.units as u
import seaborn as sns
import matplotlib.lines as mlines
from astropy.stats import sigma_clipped_stats
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from astropy.visualization import simple_norm


def ra_to_deg(ra_str):
    """RA 'HH:MM:SS' → degree"""
    if isinstance(ra_str, (float, int)): return ra_str
    h, m, s = [float(x) for x in ra_str.split(":")]
    return 15 * (h + m / 60 + s / 3600)


def dec_to_deg(dec_str):
    """DEC '±DD:MM:SS' → degree"""
    if isinstance(dec_str, (float, int)): return dec_str
    parts = dec_str.split(":")
    d = float(parts[0])
    m = float(parts[1])
    s = float(parts[2])
    sign = -1 if "-" in parts[0] else 1
    return sign * (abs(d) + m / 60 + s / 3600)


def calculate_residuals(matched_stars):
    c_img = SkyCoord(
        ra=matched_stars['img_ra'] * u.deg,
        dec=matched_stars['img_dec'] * u.deg
    )

    c_ref = SkyCoord(
        ra=matched_stars['gaia_ra'] * u.deg,
        dec=matched_stars['gaia_dec'] * u.deg
    )

    dra, ddec = c_ref.spherical_offsets_to(c_img)

    dra = dra.to(u.arcsec).value
    ddec = ddec.to(u.arcsec).value
    gmag = matched_stars["gaia_gmag"]
    mask = np.isfinite(dra) & np.isfinite(ddec) & np.isfinite(gmag)

    return dra[mask], ddec[mask], gmag[mask]


def plot_astrometry_residuals(csv_path, solve_dir, telescope_name="TUG100"):
    """CSV'den verileri okur ve yayın kalitesinde astrometri grafiği çizer."""
    if not os.path.exists(csv_path):
        print(f"⚠️ Data file not found: {csv_path}")
        return

    # 1. Veri Okuma ve Hazırlık
    df = pd.read_csv(csv_path)

    # Medyanı çıkararak tam merkeze oturtma (Residual centering)
    dra = df['dra_corr'].values - np.median(df['dra_corr'])
    ddec = df['ddec'].values - np.median(df['ddec'])

    # 2. İstatistiksel Hesaplamalar (3-sigma clipped veriden RMS)
    _, _, std_ra = sigma_clipped_stats(dra, sigma=3)
    _, _, std_dec = sigma_clipped_stats(ddec, sigma=3)
    total_rms = np.sqrt(std_ra ** 2 + std_dec ** 2)

    sns.set_theme(style="ticks")

    # 4. JointGrid Yapısını Oluşturma (Histogramlar + Ana Plot)
    g = sns.JointGrid(x=dra, y=ddec, space=0)

    # rasterized=True: Binlerce nokta varken PDF/PNG dosya boyutunu optimize eder
    g.plot_joint(sns.scatterplot, s=3, alpha=0.2, color='darkblue', edgecolor=None, rasterized=True)

    # 7. Histogramlar (Üst ve Sağ Dağılım Eğrileri)
    g.plot_marginals(sns.histplot, kde=True, color='darkblue', alpha=0.3, bins=80)

    # 8. Eksen Çizgileri (Crosshair)
    g.ax_joint.axhline(0, color='black', lw=0.8, ls='--')
    g.ax_joint.axvline(0, color='black', lw=0.8, ls='--')

    # 9. Limitler ve Etiketler
    limit = 0.9  # 1.0 arcsec sınırını rahat görmek için biraz pay bıraktık
    g.ax_joint.set_xlim(-limit, limit)
    g.ax_joint.set_ylim(-limit, limit)
    g.set_axis_labels(r'$\Delta \alpha \cos \delta$ (arcsec)', r'$\Delta \delta$ (arcsec)', fontsize=11)

    # 10. Profesyonel Legend (Proxy Artist ile Büyük Mavi Nokta)
    # Title yerine tüm bilgileri buraya topluyoruz
    info_text = (f"{telescope_name} Residuals\n"
                 f"RMS: {total_rms:.3f}'' (3$\sigma$-clipped)\n"
                 f"Stars: {len(df)}")

    # markersize=10 yaparak legend'daki noktayı görünür kılıyoruz
    blue_proxy = mlines.Line2D([], [], color='darkblue', marker='o', linestyle='None',
                               markersize=5, label=info_text)

    g.ax_joint.legend(handles=[blue_proxy], loc='upper right', frameon=True,
                      fontsize=9, facecolor='white', framealpha=0.9, edgecolor='lightgray')

    # 11. Yerleşim Ayarları (Histogramların kesilmemesi için)
    g.fig.subplots_adjust(top=0.92, right=0.92)

    # 12. Kayıt ve Temizlik
    save_name = os.path.join(solve_dir, f"MASTER_astrometry_final_son_centroidquadraticx.png")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ Astrometric residuals plot saved: {save_name}")


def plot_astrometry_magnitude_density(csv_path, solve_dir, telescope_name="TUG100"):
    df = pd.read_csv(csv_path)

    dra_all = df['dra_corr'].values
    ddec_all = df['ddec'].values
    gmag = df['gmag'].values

    print(f"📊 Processing {len(df):,} stars in total...")

    mag_bins = [5, 14, 16, 20]
    mag_labels = [
        r'9$^m$ < G$_{\rm mag}$ < 14$^m$',
        r'14$^m$ < G$_{\rm mag}$ < 16$^m$',
        r'16$^m$ < G$_{\rm mag}$ < 18$^m$'
    ]

    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.05, left=0.15, right=0.85, bottom=0.08, top=0.95)

    last_h = None
    number = [18138, 63524, 41650]
    for idx, (mag_min, mag_max) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
        ax = axes[idx]

        mask = (gmag >= mag_min) & (gmag < mag_max)
        dra_sel = dra_all[mask]
        ddec_sel = ddec_all[mask]
        n_stars = mask.sum()

        if n_stars > 0:
            # Görselleştirme için medyan merkezleme
            dra_c = dra_sel - np.median(dra_sel)
            ddec_c = ddec_sel - np.median(ddec_sel)

            # 3-sigma clipping maskesi (merkezlenmiş residual'lar üzerinde)
            dra_std = np.std(dra_c)
            ddec_std = np.std(ddec_c)

            keep = (
                    (np.abs(dra_c) <= 3.0 * dra_std) &
                    (np.abs(ddec_c) <= 3.0 * ddec_std)
            )

            dra_clip = dra_c[keep]
            ddec_clip = ddec_c[keep]


            # RMS hesapları
            rms_ra = np.sqrt(np.mean(dra_clip ** 2))
            rms_dec = np.sqrt(np.mean(ddec_clip ** 2))
            rms_tot = np.sqrt(np.mean(dra_clip ** 2 + ddec_clip ** 2))

            # Yoğunluk haritası
            h = ax.hist2d(
                dra_c, ddec_c,
                bins=50,
                cmap='viridis',
                norm=LogNorm(),
                alpha=0.9,
                range=[[-1, 1], [-1, 1]]
            )
            last_h = h

            info_text = (
                f"N = {number[idx]:,}\n"
                f"RMS$_{{RA}}$ = {rms_ra:.3f}''\n"
                f"RMS$_{{Dec}}$ = {rms_dec:.3f}''\n"
                f"RMS$_{{tot}}$ = {rms_tot:.3f}''"
            )
            print(info_text)
            ax.text(
                0.97, 0.97, info_text,
                transform=ax.transAxes,
                fontsize=7,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.8, edgecolor='none')
            )

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axhline(0, color='red', lw=0.6, ls='--', alpha=0.5)
        ax.axvline(0, color='red', lw=0.6, ls='--', alpha=0.5)

        ax.text(
            0.03, 0.97, mag_labels[idx],
            transform=ax.transAxes,
            fontsize=7,
            va='top',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                      alpha=0.8, edgecolor='none')
        )

        ax.grid(True, alpha=0.15, ls=':', lw=0.4)
        ax.tick_params(labelsize=7)
        ax.set_ylabel(r'$\Delta \delta$ (")', fontsize=8)

        if idx == 2:
            ax.set_xlabel(r'$\Delta \alpha \cos \delta$ (")', fontsize=8)

    if last_h is not None:
        cbar = fig.colorbar(last_h[3], ax=axes, fraction=0.03, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=6)

    save_name = os.path.join(solve_dir, "MASTER_astrometry_by_mag_3panel.png")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# plot_astrometry_magnitude_density(csv_path, solve_dir="../output/")

def plot_photometry_sources(data, image_sources, matched_sources, wcs, path, type, target_coord=None):
    if type == "wcs":
        # WCS porjected figure
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(111, projection=wcs)

        # ---- simple_norm ile görüntü ölçekleme (ÇOK ÖNEMLİ!) ----
        norm = simple_norm(data,
                           stretch='sinh',  # 'linear', 'sqrt', 'log', 'asinh'
                           percent=97.2,  # Piksellerin %99'unu kapsa
                           min_percent=0.1)  # En düşük %0.5'i kes

        # Görüntüyü norm ile çiz
        im = ax.imshow(data, cmap='viridis', origin='lower', norm=norm)

        # 1. TÜM kaynakları çiz (Mavi küçük noktalar)
        ax.scatter(image_sources['xcentroid'], image_sources['ycentroid'],
                   transform=ax.get_transform('pixel'),
                   edgecolor='cyan', facecolor='none', s=100, alpha=1)

        # 2. Gaia ile EŞLEŞEN kaynakları çiz (Kırmızı büyük halkalar)
        ax.scatter(matched_sources['xcentroid'], matched_sources['ycentroid'],
                   transform=ax.get_transform('pixel'),
                   edgecolor='red', facecolor='none', s=200, lw=1.5)

        # # 3. Target noktasını çiz (Yeşil büyük halka)
        # ax.scatter([717], [807],
        #            transform=ax.get_transform('pixel'),
        #            edgecolor='white', facecolor='none', s=350, lw=3)

        # # 3. Target noktasını çiz (Yeşil büyük halka)
        # ax.scatter([1209], [624],
        #            transform=ax.get_transform('pixel'),
        #            edgecolor='white', facecolor='none', s=350, lw=3)

        ax.scatter([-10], [-10],
                   transform=ax.get_transform('pixel'),
                   edgecolor='cyan', facecolor='none', s=100, lw=1.5,
                   label=f'All detected sources ({len(image_sources)})')

        ax.scatter([-10], [-10],
                   transform=ax.get_transform('pixel'),
                   edgecolor='red', facecolor='none', s=100, lw=1.5,
                   label=f'Matched Gaia sources ({len(matched_sources)})')

        ax.scatter([-10], [-10],
                   transform=ax.get_transform('pixel'),
                   edgecolor='white', facecolor='none', s=100, lw=1.5, label='Targets')

        # RA ve Dec eksenlerini ayarla
        x_min, x_max = 0, 2048
        y_min, y_max = 0, 2048
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ra = ax.coords[0]
        dec = ax.coords[1]

        # RA ayarları (sağ açıklık)
        ra.set_axislabel('Right Ascension', fontsize=12)
        ra.set_major_formatter('hh:mm:ss')  # Saat:dakika:saniye formatı
        ra.set_ticks(number=8)  # Yaklaşık 8 tane etiket göster
        ra.set_ticks_position('bt')  # Alt ve üstte göster
        ra.set_ticklabel_position('bt')

        # Dec ayarları (dik açıklık)
        dec.set_axislabel('Declination', fontsize=12)
        dec.set_major_formatter('dd:mm:ss')  # Derece:dakika:saniye
        dec.set_ticks(number=8)
        dec.set_ticks_position('lr')  # Sol ve sağda göster
        dec.set_ticklabel_position('lr')

        # Izgara ekle
        ax.grid(color='white', alpha=0.5, linewidth=0.5, linestyle='--')

        # Lejant
        ax.legend(loc='upper left', framealpha=0.5)

        # Başlık ve kaydet
        # plt.title(f"Source Analysis: {len(image_sources)} Detected / {len(matched_table)} Gaia Matches")
        plt.savefig(f"{path}_wcs.png", format='png',
                    bbox_inches='tight', dpi=150)
        plt.close()

    elif type == "pixel":
        plt.figure(figsize=(14, 14))  # Görüntüyü arka plan olarak çiz #
        # vmin ve vmax değerlerini (median-std) ve (median+5*std) yaparak yıldızları daha belirgin kılabilirsin
        vmin, vmax = np.percentile(data, [30, 99])
        # Daha stabil bir kontrast için percentile kullandık
        plt.imshow(data, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        # X şartı: 2040 ile 2055 arasındakileri dışla
        mask_x = ~((image_sources['xcentroid'] >= 1010) & (image_sources['xcentroid'] <= 1035))
        mask_y = (image_sources['ycentroid'] >= 15) & (image_sources['ycentroid'] <= 2030)
        # İki maskeyi birleştir (Hem X hem Y şartını sağlayanlar)
        final_mask = mask_x & mask_y
        # Filtrelenmiş veriyi oluştur
        image_sources = image_sources[final_mask]
        # 1. Görüntüde bulunan TÜM kaynakları çiz (Mavi küçük noktalar)
        plt.scatter(image_sources['xcentroid'], image_sources['ycentroid'],
                    edgecolor='blue', facecolor='none', s=20, alpha=1)
        # 2. Gaia ile EŞLEŞEN temiz kaynakları çiz (Kırmızı büyük halkalar)
        plt.scatter(matched_sources['xcentroid'], matched_sources['ycentroid'],
                    edgecolor='red', facecolor='none', s=100, lw=1.5)
        if target_coord:
            plt.scatter(716.03672835367195, 807.52522584019504, edgecolor='green', facecolor='none', s=200, lw=2)
            plt.scatter([-10], [-10],
                        edgecolor='green', facecolor='none', s=100, lw=1.5, label='Targets')
        # 3. Target noktasını çiz (Yeşil büyük halka)

        plt.scatter([-10], [-10],
                    edgecolor='blue', facecolor='none', s=100, lw=1.5,
                    label=f'All detected sources ({len(image_sources)})')

        plt.scatter([-10], [-10],
                    edgecolor='red', facecolor='none', s=100, lw=1.5,
                    label=f'Matched Gaia sources ({len(matched_sources)})')

        plt.legend(loc='upper left', framealpha=0.5)

        x_min, x_max = 0, 2048
        y_min, y_max = 0, 2048
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xlabel("X (pixel)")
        plt.ylabel("Y (pixel)")
        plt.legend(loc='upper right')
        # İstersen bu grafiği kaydetmesi için:
        plt.savefig(f"{path}_pixel.png", format='png', bbox_inches='tight', dpi=300)
        plt.close()
