import glob

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.stats import sigma_clipped_stats
import os


# Dosyayı aç (update modunda)

# from astropy.io import fits
#
# files = glob.glob("../rename/test/*fits")
# for file in files:
#
#     with fits.open(file, mode='update') as hdul:
#         header = hdul[0].header
#
#         # OBJCTRA -> RA değişikliği
#         if 'RA' in header:
#             header.rename_keyword('RA', 'OBJCTRA')
#             print("OBJCTRA anahtarı RA olarak değiştirildi.")
#
#         # OBJCTDEC -> DEC değişikliği
#         if 'DEC' in header:
#             header.rename_keyword('DEC', 'OBJCTDEC')
#             print("OBJCTDEC anahtarı DEC olarak değiştirildi.")
#
#         # Değişiklikleri diske yaz
#         hdul.flush()
# exit()
def ra_to_deg(ra_str):
    """RA 'HH:MM:SS' → degree"""
    if isinstance(ra_str, (float, int)): return ra_str*15
    h, m, s = [float(x) for x in ra_str.split(":")]
    return 15 * (h + m/60 + s/3600)

def dec_to_deg(dec_str):
    """DEC '±DD:MM:SS' → degree"""
    if isinstance(dec_str, (float, int)): return dec_str
    parts = dec_str.split(":")
    d = float(parts[0])
    m = float(parts[1])
    s = float(parts[2])
    sign = -1 if "-" in parts[0] else 1
    return sign * (abs(d) + m/60 + s/3600)


def analyze_astrometry(matched_table, telescope_name="T100", image_name="frame"):
    """
    Her kare için ayrı bir residual grafiği oluşturur ve kaydeder.
    """
    img_ra = np.asarray(matched_table['img_ra'])
    gaia_ra = np.asarray(matched_table['gaia_ra'])
    img_dec = np.asarray(matched_table['img_dec'])
    gaia_dec = np.asarray(matched_table['gaia_dec'])

    # Farklar (arcsec)
    dra = (img_ra - gaia_ra) * 3600.0
    ddec = (img_dec - gaia_dec) * 3600.0

    # Cos(delta) düzeltmesi
    avg_dec = np.deg2rad(np.mean(gaia_dec))
    dra_corr = dra * np.cos(avg_dec)

    # Sigma-Clipping
    _, mean_ra, std_ra = sigma_clipped_stats(dra_corr, sigma=3)
    _, mean_dec, std_dec = sigma_clipped_stats(ddec, sigma=3)
    total_rms = np.sqrt(std_ra ** 2 + std_dec ** 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dra_corr, ddec, alpha=0.6, s=40, color='darkblue', edgecolors='white')

    limit = max(np.abs(dra_corr).max() if len(dra_corr) > 0 else 0.5,
                np.abs(ddec).max() if len(ddec) > 0 else 0.5, 0.5)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.axvline(0, color='black', lw=1, ls='--')

    ax.set_xlabel(r'$\Delta \alpha \cos \delta$ (arcsec)')
    ax.set_ylabel(r'$\Delta \delta$ (arcsec)')

    ax.set_title(f"Astrometric Residuals\nFile: {image_name}\n"
                 f"RMS: {total_rms:.3f}'' | Stars: {len(matched_table)}")

    plt.grid(alpha=0.3)

    save_path = f"residuals/residuals_{telescope_name}_{image_name}.png"
    plt.savefig(save_path, dpi=300)

    plt.close(fig)

    return total_rms, std_ra, std_dec




def plot_publication_astrometry(csv_path, solve_dir, telescope_name="T100"):
    """CSV'den verileri okur ve histogramlı/konturlu yayın grafiği çizer."""
    if not os.path.exists(csv_path):
        print(f"⚠️ Veri dosyası bulunamadı: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    dra = df['dra_corr'].values
    ddec = df['ddec'].values

    # İstatistikler
    _, _, std_ra = sigma_clipped_stats(dra, sigma=3)
    _, _, std_dec = sigma_clipped_stats(ddec, sigma=3)
    total_rms = np.sqrt(std_ra ** 2 + std_dec ** 2)

    # Grafik Ayarları
    sns.set_theme(style="whitegrid")
    g = sns.JointGrid(x=dra, y=ddec, space=0)

    # Ana Dağılım (Noktalar + Yoğunluk Konturları)
    g.plot_joint(sns.scatterplot, s=1, alpha=0.1, color='darkblue', edgecolor=None)
    g.plot_joint(sns.kdeplot, color='red', alpha=0.6, levels=5, linewidths=1.5)

    # Histogramlar (Üst ve Sağ)
    g.plot_marginals(sns.histplot, kde=True, color='darkblue', alpha=0.3, bins=80)

    # Eksenler ve Çizgiler
    g.ax_joint.axhline(0, color='black', lw=1, ls='--')
    g.ax_joint.axvline(0, color='black', lw=1, ls='--')

    limit = 1.0  # 1 arcsec sınır (genelde yeterlidir)
    g.ax_joint.set_xlim(-limit, limit)
    g.ax_joint.set_ylim(-limit, limit)

    g.set_axis_labels(r'$\Delta \alpha \cos \delta$ (arcsec)', r'$\Delta \delta$ (arcsec)')

    title = f"{telescope_name} Master Residuals\nCombined RMS: {total_rms:.3f}'' | Stars: {len(df)}"
    g.fig.suptitle(title, y=1.02, fontsize=14)

    save_name = os.path.join(solve_dir, f"MASTER_astrometry_publication.png")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafik kaydedildi: {save_name}")

# print(ra_to_deg("06:08:4.34"))
# print(ra_to_deg("14:15:46.84"))
# print(ra_to_deg("14:15:37.77"))
#
# print(dec_to_deg("08:21:38.48"))
# print(dec_to_deg("-10:48:09.0"))
# print(dec_to_deg("-10:47:02.6"))
