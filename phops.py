import datetime
import glob
import math
import time
import warnings

import matplotlib.pyplot
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)
import astropy.io.fits as fits
from astropy.table import Table, SerializedColumn
from astropy.wcs import WCS
import astropy.coordinates as coord
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.stats import signal_to_noise_oir_ccd
from astroquery.vizier import Vizier
from photutils.centroids import centroid_quadratic
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.aperture import aperture_photometry
from photutils.background import MedianBackground
from photutils.background import SExtractorBackground
import sewpy
import numpy as np
import os
from astroquery.jplhorizons import Horizons
from astroquery.imcce import Skybot
from astropy.coordinates import SkyCoord
from astropy.io import ascii
import matplotlib.pyplot as plt
from f2n import f2n

# gain = 0.57
# rdnoise = 4.11
# pxscale = 0.61


gain = 0.9
rdnoise = 5.50
pxscale = 0.32641

def object_plot(path, x, y, id, t=None, z=None):
    data = fits.open(path)[0].data.astype(float)
    pix_width = 110
    cropped_data = data[(y - pix_width):(y + pix_width), (x - pix_width):(x + pix_width)]
    fits.writeto("out.fits", cropped_data, overwrite=True)

    fitsfile = "out.fits"
    image = f2n.fromfits(fitsfile, verbose=False)
    image.setzscale('auto', 'auto')
    image.makepilimage('log', negative=True)

    print('\033[1;34mPlotting sources on {0}...\033[0m'.format("catalog"))
    # x = x.astype(np.float)
    # y = y.astype(np.float)
    # for i in range(len(x)):
    #     image.drawcircle(int(x[i]),
    #                      int(y[i]),
    #                      r=15 / pxscale,
    #                      colour=(255, 0, 0),
    #                      fontsize=10
    #                      )

    # image.drawcircle(t,
    #                  z,
    #                  r=15 / pxscale,
    #                  label=str(id),
    #                  colour=(0, 0, 255))
    # #
    image.drawcircle(pix_width,
                     pix_width,
                     label=str(id),
                     r=15 / pxscale,
                     fontsize=15,
                     colour=(0, 255, 0))

    # image.writetitle(os.path.basename("out.fits"))
    fitshead, fitsextension = os.path.splitext(path)
    image.tonet('{0}_asteroid.png'.format(fitshead))

    print('\033[1;34mAll sources plotted on: {0}.png\033[0m'.format(fitshead))

    return True


# paths = glob.glob("/home/orhan/Desktop/kolimasyon_TZ/yldz3_bati/*.fits", recursive=True)
# for i in paths:
#     object_plot(i, 1647, 1284)
# exit()
def solve_field(path,
                tweak_order=2,
                downsample=4,
                radius=0.2,
                ra=None,
                dec=None,
                # ra_keyword="OBJCTRA",
                ra_keyword="ra",
                # dec_keyword="OBJCTDEC"
                dec_keyword="dec"
                ):
    """
    The astrometry engine will take any image and return
    the astrometry world coordinate system (WCS).

    @param image_path: FITS image file name with path
    @type image_path: str
    @param tweak_order: Polynomial order of SIP WCS corrections
    @type tweak_order: integer
    @param downsample: Downsample the image by factor int before
    running source extraction
    @type downsample: integer
    @param radius: Only search in indexes within 'radius' of the
    field center given by --ra and --dec
    @type radius: str
    @param ra: RA of field center for search, format: degrees or hh:mm:ss
    @type ra: str
    @param dec: DEC of field center for search, format: degrees or hh:mm:ss
    @type dec: str
    @param ra_keyword: RA keyword in the FITS image header
    @type ra_keyword: str
    @param dec_keyword: DEC keyword in the FITS image header
    @type dec_keyword: str
    @return: boolean
    """
    try:
        if ra is None and dec is None:
            fo = fits.open(path)
            ra = fo[0].header[ra_keyword]
            dec = fo[0].header[dec_keyword]
            ra = ra.strip()
            dec = dec.strip()
            ra = ra.replace(" ", ":")
            dec = dec.replace(" ", ":")
        else:
            ra = ra.strip()
            dec = dec.strip()
            ra = ra.replace(" ", ":")
            dec = dec.replace(" ", ":")

        # Cleaning
        if ".gz" in path:
            root = '.'.join(path.split('.')[:-2])
        else:
            root, extension = os.path.splitext(path)

        os.system(("solve-field --axy none --index-xyls none "
                   "--solved none --corr none --no-plots "
                   "--match none --rdls none --solved none "
                   "--no-verify --tweak-order {0} --downsample {1} "
                   "--overwrite --radius {2} --no-tweak "
                   "--ra {3} --dec {4} --new-fits {5}_new.fits {6}").format(tweak_order,
                                                                            downsample,
                                                                            radius,
                                                                            ra,
                                                                            dec,
                                                                            root,
                                                                            path))
        os.system("rm -rf none {0}.wcs".format(root))

        return True

    except Exception as e:
        print(e, "1")


def source_extract(path):
    """
    It detects sources on FITS image with sep module.
    @return: astropy.table
    """
    sew = sewpy.SEW(
        params=['FLAGS', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', "FWHM_IMAGE", "MAG_APER", "FLUX_RADIUS"],
        config={'DETECT_TYPE': 'CCD',
                'DETECT_MINAREA': 5,
                'DETECT_THRESH': 1.5,
                'ANALYSIS_THRESH': 1.5,
                'PHOT_APERTURES': 8,
                'PHOT_PETROPARAMS': '"5, 5"',
                'SATUR_LEVEL': 65535.0,
                'DEBLEND_NTHRESH': 32,
                'DEBLEND_MINCONT': 0.005,
                'PHOT_AUTOPARAMS': '"2.5, 3.5"',
                'BACK_SIZE': 64,
                'BACK_FILTERSIZE': 3,
                'FILTER': 'Y',
                'MAG_ZEROPOINT': 25.0,
                'GAIN': 0.57,
                'PIXEL_SCALE': 0.61,
                'SEEING_FWHM': 0,
                'CLEAN': "Y",
                'VERBOSE_TYPE': 'QUIET'})
    out = sew(path)
    return out["table"]


def get_header(key, path):
    """
    Extracts requested keyword from FITS header.
    @param key: Requested keyword.
    @type key: str
    @return: str
    """

    try:
        file = fits.open(path)
        header_key = file[0].header[key]
        ret = header_key
    except Exception as e:
        print(e, "2")
        ret = False

    return ret


def xy2wcs_coord(x=None, y=None, path=None, center=False):
    file = fits.open(path)
    if center:
        x, y = get_header("NAXIS1", path) / 2, get_header("NAXIS2", path) / 2
    w = WCS(file[0].header)
    ra, dec = w.wcs_pix2world(x, y, 0)
    return ra, dec


def radec2xy_coord(ra, dec, path=None):
    file = fits.open(path)
    w = WCS(file[0].header)
    x, y = w.wcs_world2pix(ra, dec, 0)
    return x, y


def findcenter(sources, ra, dec, path):
    separation = (coord.SkyCoord(ra=sources['ALPHA_J2000'], dec=sources['DELTA_J2000'], unit=(u.deg, u.deg),
                                 frame='icrs').separation(
        coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))).arcsecond)
    x = sources[separation.argmin()]["X_IMAGE"]
    y = sources[separation.argmin()]["Y_IMAGE"]
    rax = sources[separation.argmin()]["ALPHA_J2000"]
    dex = sources[separation.argmin()]["DELTA_J2000"]
    fwhm = sources[separation.argmin()]["FWHM_IMAGE"]
    data = fits.open(path)[0].data.astype(float)
    pix_x = int(float(x))
    pix_y = int(float(y))
    pix_width = int(float(fwhm))
    cropped_data = data[(pix_y - pix_width):(pix_y + pix_width), (pix_x - pix_width):(pix_x + pix_width)]
    x_cropped, y_cropped = centroid_quadratic(cropped_data)
    x_cent = x_cropped + pix_x - pix_width + 1
    y_cent = y_cropped + pix_y - pix_width + 1
    if np.isnan(x_cent):
        return x, y, fwhm
    elif abs(x_cent - x) > 3 or abs(y_cent - y) > 3:
        return x, y, fwhm
    else:
        return x_cent, y_cent, fwhm


def findobjects(sources, path):
    imagex = sources['X_IMAGE']
    imagey = sources['Y_IMAGE']
    fwhm = sources['FWHM_IMAGE']
    ra = sources['ALPHA_J2000']
    dec = sources['DELTA_J2000']
    sex_mag = sources['MAG_APER']
    # asteroid_ra, asteroid_dec, _, _, _, _, _, _, _ = find_asteroid(name=2059, path=path)
    x_image_center, y_image_center = xy2wcs_coord(path=path, center=True)
    gaia_match = gaia_query(x_image_center, y_image_center, width=18, max_sources=1000, max_mag=18)
    # gaia_match = gaia_query(asteroid_ra, asteroid_dec, width=int(400*pxscale/60), max_sources=1000, max_mag=18)
    gaia_ra = gaia_match["RA_ICRS"]
    gaia_dec = gaia_match["DE_ICRS"]
    gaia_mag = gaia_match["Gmag"]
    gaia_bprp = gaia_match["BP-RP"]

    data = fits.open(path)[0].data.astype(float)
    source = []
    counter = len(sources)
    for i in range(len(sources)):
        pix_y = int(float(imagey[i]))
        pix_x = int(float(imagex[i]))
        pix_width = int(float(fwhm[i]) * 2.5)
        cropped_data = data[(pix_y - pix_width):(pix_y + pix_width), (pix_x - pix_width):(pix_x + pix_width)]

        if len(cropped_data) == 0 or np.any((cropped_data < 0)) or fwhm[i] < 1 or fwhm[
            i] > 20 or pix_y < 30 or pix_y > 2018 or pix_x < 40 or pix_x > 2000 or pix_x == 1024 or sex_mag[i] > 17.5:
            print(counter)
            counter -= 1
            continue
        x_centx, y_centy = centroid_quadratic(cropped_data)
        x_cent = (x_centx + pix_x - pix_width + 1)
        y_cent = (y_centy + pix_y - pix_width + 1)
        ra_cent, dec_cent = xy2wcs_coord(x_cent, y_cent, path)
        if np.isnan(ra_cent):
            print(counter)
            counter -= 1
            continue
        separation = (coord.SkyCoord(ra=gaia_ra, dec=gaia_dec, unit=(u.deg, u.deg), frame='icrs').separation(
            coord.SkyCoord(ra=ra_cent, dec=dec_cent,
                           unit=(u.deg, u.deg))).arcsecond)  # yalnızca gaia yıldızlarının separationı

        # Gaia ve Landolt yıldızlarını matchlerken 2 yay saniye üzerinde fark varsa yoksay.
        if separation.min() > 2:
            print(counter)
            counter -= 1
            continue

        # Fotometrisi yapılacak Gaia yıldızlarının etrafında ~25 px civarında yıldız varsa yoksay.
        separation2 = (coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs').separation(
            coord.SkyCoord(ra=ra_cent, dec=dec_cent, unit=(u.deg, u.deg))).arcsecond)
        if sorted(separation2)[1] < 13:
            print(counter)
            counter -= 1
            continue

        else:
            g0 = sex_mag[i]
            g1 = gaia_ra[separation.argmin()]
            g2 = gaia_dec[separation.argmin()]
            g3 = gaia_mag[separation.argmin()]
            g4 = gaia_bprp[separation.argmin()]
            g5, g6 = radec2xy_coord(g1, g2, path)
            g7 = x_cent - g5
            g8 = y_cent - g6
            if abs(g7) > 3 or abs(g8) > 3:
                print(counter)
                counter -= 1
                continue
        source.append([path,
                       x_cent,
                       y_cent,
                       ra[i],
                       dec[i],
                       ra_cent,
                       dec_cent,
                       g0,
                       g1,
                       g2,
                       g3,
                       g4,
                       fwhm[i]
                       ])

        sourcetable = np.asarray(source)
        centers = Table(sourcetable, names=('FileName',
                                            'X_center',
                                            'Y_center',
                                            'RA_sex',
                                            'Dec_sex',
                                            'RA_center',
                                            'Dec_center',
                                            'Sex_Mag',
                                            'RA_Gaia',
                                            'Dec_Gaia',
                                            'G_Mag',
                                            'BP-RP',
                                            'FWHM_IMAGE'
                                            ))
        print(counter)
        counter -= 1
    ra_center_calc = centers["RA_center"]
    ra_gaia = centers["RA_Gaia"]
    dec_center_calc = centers["Dec_center"]
    dec_gaia = centers["Dec_Gaia"]
    diff_coord_ra = np.std(abs(np.asarray(ra_center_calc, dtype=float) - np.asarray(ra_gaia, dtype=float)))
    diff_coord_dec = np.std(abs(np.asarray(dec_center_calc, dtype=float) - np.asarray(dec_gaia, dtype=float)))
    ascii.write(centers, "phot_{}_centers.dat".format(str(id)), overwrite=True)
    return centers


def gaia_query(ra_deg=None, dec_deg=None, width=18, max_mag=18,
               max_coo_err=0.05,
               max_sources=1000):
    """
    Query Gaia DR1 @ VizieR using astroquery.vizier
    parameters: ra_deg, dec_deg, width: RA, Dec, field
    @param ra_deg: RA in degrees
    @type ra_dec: float
    @param dec_deg: DEC in degrees
    @type dec_deg: float
    @param max_mag: Limit G magnitude to be queried object(s)
    @type max_mag: float
    @max_coo_err: Max error of position
    @type max_coo_err: float
    @max_sources: Maximum number of sources
    @type max_sources: int
    @returns: astropy.table object
    :param width:
    :param width:
    """

    vquery = Vizier(columns=['Source', 'RA_ICRS',
                             'DE_ICRS', 'e_RA_ICRS',
                             'e_DE_ICRS', 'Gmag', 'bp_rp',
                             'pmRA', 'pmDE',
                             'e_pmRA', 'e_pmDE',
                             'Epoch', 'Plx'],
                    column_filters={"phot_g_mean_mag":
                                        ("<{:f}".format(max_mag)),
                                    "e_RA_ICRS":
                                        ("<{:f}".format(max_coo_err)),
                                    "e_DE_ICRS":
                                        ("<{:f}".format(max_coo_err))},
                    row_limit=max_sources)

    field = coord.SkyCoord(ra="%.6f" % ra_deg, dec="%.6f" % dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')

    return (vquery.query_region(field,
                                width="{:f}d".format(width / 60),
                                catalog="I/355/gaiadr3")[0])


def Gmag2Vmag(gmag, bprp, filter):
    #### GaiaDR3 - JohnsonCousin B relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T9
    if filter == "B":
        a0 = 0.01448
        a1 = -0.6874
        a2 = -0.3604
        a3 = 0.06718
        a4 = -0.006061

    #### GaiaDR3 - JohnsonCousin V relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T9
    if filter == "V":
        a0 = -0.02704
        a1 = 0.01424
        a2 = -0.2156
        a3 = 0.01426
        a4 = 0

    #### GaiaDR3 - JohnsonCousin R relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T9
    if filter == "R":
        a0 = -0.02275
        a1 = 0.3961
        a2 = -0.1243
        a3 = -0.01396
        a4 = 0.003775

    #### GaiaDR3 - JohnsonCousin I relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T9
    if filter == "I":
        a0 = 0.01753
        a1 = 0.76
        a2 = -0.0991
        a3 = 0
        a4 = 0

    #### GaiaDR3 - SDSS12 r relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T8
    if filter == "r":
        a0 = -0.09837
        a1 = 0.08592
        a2 = 0.1907
        a3 = -0.1701
        a4 = 0.02263

    #### GaiaDR3 - SDSS12 i relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T8
    if filter == "i":
        a0 = -0.293
        a1 = 0.6404
        a2 = -0.09609
        a3 = -0.002104
        a4 = 0

    #### GaiaDR3 - SDSS12 g relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T8
    if filter == "g":
        a0 = 0.2199
        a1 = -0.6365
        a2 = -0.1548
        a3 = 0.0064
        a4 = 0

    #### GaiaDR3 - SDSS12 z relation
    # https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T8
    if filter == "z":
        a0 = -0.4619
        a1 = 0.8992
        a2 = -0.08271
        a3 = 0.005029
        a4 = 0

    if filter == "E":
        # EMPTY
        a0 = 0
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0
    # else:
    #     print("Filter must be V, R, I")
    #     exit()
    px = a0 + (a1 * bprp) + (a2 * bprp ** 2) + (a3 * bprp ** 3) + (a4 * bprp ** 4)
    gmag2vmag = gmag - px
    return gmag2vmag


def phot(sources, path, length=True):
    if length:
        forrange = len(sources)
        fwhm = np.median(sources["FWHM_IMAGE"].astype(float))

    else:
        forrange = 1
        fwhm = sources["FWHM_IMAGE"][0]

    imagex = sources["X_center"]
    imagey = sources["Y_center"]
    data = fits.open(path)[0].data.astype(float)
    fulldata_median_bkg = MedianBackground().calc_background(data)
    aper_flux_sum = []
    Inst_Mag = []
    bkg_mean = []
    mag_err = []

    for i in range(forrange):
        r = fwhm * 2.5
        r = 4 / pxscale
        aperture = CircularAperture([imagex[i], imagey[i]], r)
        annulus_aperture = CircularAnnulus([imagex[i], imagey[i]], r_in=r + (7 / pxscale), r_out=r + (12 / pxscale))
        apers = [aperture, annulus_aperture]
        phot_table = aperture_photometry(data, apers)
        mean_bkg = phot_table['aperture_sum_1'][0] / annulus_aperture.area
        median_bkg = ApertureStats(data, annulus_aperture).median
        if abs(mean_bkg - fulldata_median_bkg) / fulldata_median_bkg * 100 > 400:
            bkg_sum = fulldata_median_bkg * aperture.area
            mean_bkg = fulldata_median_bkg
        else:
            bkg_sum = mean_bkg * aperture.area
        bkg_sum = median_bkg * aperture.area
        final_sum = phot_table['aperture_sum_0'][0] - bkg_sum
        if final_sum < 0:
            final_sum = 1
        aper_flux_sum.append(final_sum)
        Inst_Mag.append(25 - 2.5 * math.log(final_sum, 10))
        bkg_mean.append(mean_bkg)
        tot_noise = math.sqrt(final_sum * gain + bkg_sum * gain + rdnoise * rdnoise * aperture.area)
        snr = final_sum * gain / tot_noise
        error = np.sqrt(final_sum / gain + bkg_sum / gain)
        mag_err.append(1.08573620476 * (error / final_sum))
        if not length:
            tot_noise = math.sqrt(final_sum * gain + bkg_sum * gain + rdnoise * rdnoise * aperture.area)
            snr = final_sum * gain / tot_noise
            error = np.sqrt(final_sum / gain + bkg_sum / gain)
            magerr = 1.08573620476 * (error / final_sum)

    if len(aper_flux_sum) == 1:
        return aper_flux_sum[0], bkg_mean[0], Inst_Mag[0], snr, magerr
    else:
        return aper_flux_sum, bkg_mean, Inst_Mag, 0, mag_err


from sklearn.linear_model import RANSACRegressor


def zeropoint(sources, x, y, magerr, plot=False, show=False, path=None):
    r_distance = math.sqrt(math.pow(x - 1024, 2) + math.pow(y - 1024, 2))
    calcV = sources["Calculated_Vmag"]
    instmag = sources["Inst_Mag"]
    x_coord = sources["X_center"]
    y_coord = sources["Y_center"]
    print("lenler", len(calcV), len(instmag), len(x_coord), len(y_coord), "lenler")
    tt = []
    # calcV, instmag = zip(*((x, y) for x, y in zip(calcV, instmag) if x <= 17 and x >= 13.5 and y < 16.5))
    calcV, instmag, x_coord, y_coord, magerr = zip(
        *((x, y, z, q, m) for x, y, z, q, m in zip(calcV, instmag, x_coord, y_coord, magerr) if
          17 >= x >= 9.5 and y < 25))
    # mag_diffs = np.asarray(sources["Calculated_Vmag"], dtype=float) - np.asarray(sources["Inst_Mag"], dtype=float)
    mag_diffs = np.asarray(calcV, dtype=float) - np.asarray(instmag, dtype=float)
    zero_point = np.mean(sigma_clip(mag_diffs, sigma=3, cenfunc="mean"))
    stddev = np.std(sigma_clip(mag_diffs, sigma=3, cenfunc="mean"))
    r_resultant = (
            ((np.asarray(x_coord, dtype=float) - 1024) ** 2 + (np.asarray(y_coord, dtype=float) - 1024) ** 2) ** (
            1 / 2))
    x_resultant = abs(np.asarray(x_coord, dtype=float))
    y_resultant = abs(np.asarray(y_coord, dtype=float))
    # print(list(zip(x_coord, y_coord)))
    # for i in (sigma_clip(mag_diffs, sigma=3, cenfunc="mean")):
    #     print(i)
    # print(list(zip(r_resultant, sigma_clip(mag_diffs, sigma=3, cenfunc="mean"))))
    # xaxis, yaxis, zaxis, raxis = zip(
    #     *((x, y, z, q) for x, y, z, q in
    #       zip(x_resultant, y_resultant, sigma_clip(mag_diffs, sigma=2, cenfunc="mean"), r_resultant) if z))
    #
    xaxis, yaxis = zip(
        *((x, y) for x, y in
          zip(r_resultant, sigma_clip(mag_diffs, sigma=3, cenfunc="mean")) if y))
    # math.sqrt(sources["X_centziper"]**2+sources["Y_center"]**2)
    # print(math.sqrt(math.pow(x_coord,2) + math.pow(y_coord,2)))
    # for i in range(len(xaxis)):
    #     tt.append([xaxis[i], yaxis[i], zaxis[i], raxis[i]])
    # zz = np.asarray(tt)
    # table = Table(zz, names=("x", "y", "r", "f"))
    # ascii.write(table, "test2.dat")
    # magerr_list = [[i] for i in magerr]
    ######RANSAC############
    magerr = np.asarray(magerr, dtype=float)
    r_resultant = r_resultant.reshape(-1, 1)
    mag_diffs = mag_diffs.reshape(-1, 1)
    magerr = magerr.reshape(-1, 1)
    ransac = RANSACRegressor()
    ransac.fit(r_resultant, mag_diffs)
    inlier_mask = ransac.inlier_mask_
    zero_point = np.mean(mag_diffs[inlier_mask])
    stddev = np.std(mag_diffs[inlier_mask])
    xaxis, yaxis, error = r_resultant[inlier_mask].flatten(), mag_diffs[inlier_mask].flatten(), magerr[
        inlier_mask].flatten()
    #############################
    zmag_fit = np.polyfit(xaxis, yaxis, 1)
    slope = math.atan(zmag_fit[0]) * 180 / math.pi
    zmag = np.poly1d(zmag_fit)
    r = np.linspace(30, 1160, len(xaxis))
    zero_point = (zmag(r_distance))  # use this if polynomial fit for center distance r
    # zero_point = np.median(mag_diffs)
    print("SLOPE", slope)
    print("SLOPE", zmag)
    if plot:
        plt.plot(xaxis, yaxis, "o", markersize=2, color="#1f77b4",
                 label="Gaia DR3 kaynakları: {}".format(len(xaxis)))
        plt.plot(r, zmag(r), markersize=2, label="doğrusal fit-slope {}".format(str("%.4f" % slope)), linestyle='dashed', color="#ff7f0e")
        plt.xlabel("r - Görüntü merkezine olan uzaklık (piksel)")
        plt.ylabel("Gaia DR3 kaynaklarının parlaklık farkı (Hesaplanan - Aletsel)")
        plt.legend(loc="lower left")
        plt.ylim(matplotlib.pyplot.ylim()[0] - 0.1, matplotlib.pyplot.ylim()[1] + 0.08)
        plt.xlim(0, 1200)
        plt.tight_layout()
        save = os.path.splitext(path)[0] + ".pdf"
        plt.savefig(save)

        # fig, axs = plt.subplots(1,2)
        # fig.suptitle('Vertically stacked subplots')
        # axs[0].plot(r_resultant, mag_diffs, "o", markersize=2, color="#ff7f0e")
        # axs[0].plot(xaxis, yaxis, "o", markersize=2, color="#1f77b4")
        #
        # axs[0].set_ylim(axs[0].get_ylim()[0]-0.1, axs[0].get_ylim()[1])
        #
        # save = os.path.splitext(path)[0] + "eski.pdf"
        # axs[1].plot(xaxis, yaxis, "o", markersize=2, label="{} Gaia sources in the image".format(len(yaxis)))
        # axs[1].plot(r, zmag(r), "o", markersize=2)
        #
        # # plt.plot(mag_diffs, "o", markersize=2,)
        # # plt.plot(sigma_clip(mag_diffs, sigma=3, cenfunc="mean"), "o", markersize=2,)
        # plt.xlabel("r - Distance to the center of the image (px)")
        # plt.ylabel("Magnitude Difference (Instrumental - Calculated)")
        # plt.legend(loc="upper left")
        # # plt.title("zeromag={:.2f} and stddev={:.2f}".format(zero_point, stddev))
        # # plt.title("stddev={:.2f}".format(stddev))
        # plt.text(210, 3.08, "zero magnitude = {:.7f} r + {:.3f}".format(zmag_fit[0], zmag_fit[1], stddev), fontsize=12, style='italic')
        # plt.ylim(matplotlib.pyplot.ylim()[0]-0.1, matplotlib.pyplot.ylim()[1])
        # plt.xlim(0, 1200)
        # plt.show()
        # plt.savefig(save)
        # #
        #
        # plt.plot(mag_diffs, "o", markersize=2, color="#2c506a",
        #          label="Görüntü içerisinde yer alan Gaia DR3 kaynakları: {}".format(len(mag_diffs)))
        # plt.xlabel("Sayı")
        # plt.ylabel("Gaia DR3 kaynaklarının parlaklık farkı (Hesaplanan - Aletsel)")
        # plt.legend(loc="lower left")
        # plt.ylim(matplotlib.pyplot.ylim()[0] + 0.1, matplotlib.pyplot.ylim()[1] + 0.1)
        # # plt.xlim(0, 1200)
        # plt.tight_layout()
        # plt.savefig("/home/orhan/Desktop/baboquivari/tes/zeropoint1.pdf")
        # plt.show()
        # plt.clf()
        # #####################################################
        # plt.plot(r_resultant, mag_diffs, "o", markersize=2, color="#2c506a",
        #          label="Görüntü içerisinde yer alan Gaia DR3 kaynakları: {}".format(len(mag_diffs)))
        # plt.xlabel("r - Görüntü merkezine olan uzaklık (piksel)")
        # plt.ylabel("Gaia DR3 kaynaklarının parlaklık farkı (Hesaplanan - Aletsel)")
        # plt.legend(loc="lower left")
        # plt.ylim(matplotlib.pyplot.ylim()[0] + 0.1, matplotlib.pyplot.ylim()[1] + 0.1)
        # plt.xlim(0, 1200)
        # plt.tight_layout()
        # plt.savefig("/home/orhan/Desktop/baboquivari/tes/zeropoint2.pdf")
        # plt.show()
        # plt.clf()
        # #####################################################
        # plt.plot(r_resultant, mag_diffs, "o", markersize=2, color="#ff7f0e", label="RANSAC sonrası kullanılacak Gaia DR3 kaynakları: {}".format(len(xaxis)))
        # plt.plot(xaxis, yaxis, "o", markersize=2, color="#2c506a", label="Artık Gaia DR3 kaynakları: {}".format(len(mag_diffs)-len(xaxis)))
        # plt.xlabel("r - Görüntü merkezine olan uzaklık (piksel)")
        # plt.ylabel("Gaia DR3 kaynaklarının parlaklık farkı (Hesaplanan - Aletsel)")
        # plt.legend(loc="lower left")
        # plt.ylim(matplotlib.pyplot.ylim()[0] + 0.1, matplotlib.pyplot.ylim()[1] + 0.1)
        # plt.xlim(0, 1200)
        # plt.tight_layout()
        # plt.savefig("/home/orhan/Desktop/baboquivari/tes/zeropint3.pdf")
        # plt.show()
        # plt.clf()
        # ######################################
        # plt.errorbar(xaxis, yaxis, error, fmt="o", capsize=3, markersize=3, color="#2c506a",
        #              label="Görüntü içerisinde yer alan Gaia DR3 kaynakları: {}".format(len(yaxis)))
        # plt.plot(r, zmag(r), markersize=2, label="doğrusal fit", linestyle='dashed', color="#ff7f0e")
        # plt.xlabel("r - Görüntü merkezine olan uzaklık (piksel)")
        # plt.ylabel("Gaia DR3 kaynaklarının parlaklık farkı (Hesaplanan - Aletsel)")
        # plt.legend(loc="upper left")
        # plt.text(170, 3.11, "sıfır-nokta parlaklığı = {:.7f} r + {:.3f}".format(zmag_fit[0], zmag_fit[1], stddev),
        #          fontsize=12, style='italic')
        # plt.ylim(matplotlib.pyplot.ylim()[0] - 0.03, matplotlib.pyplot.ylim()[1] + 0.03)
        # plt.xlim(0, 1200)
        # plt.tight_layout()
        # plt.savefig("/home/orhan/Desktop/baboquivari/tes/zeropoint4.pdf")
        # plt.show()
        # plt.clf()
        ############################

        # plt.plot(mag_diffs, "o", markersize=2,)
        # plt.plot(sigma_clip(mag_diffs, sigma=3, cenfunc="mean"), "o", markersize=2,)
        # plt.title("zeromag={:.2f} and stddev={:.2f}".format(zero_point, stddev))
        # plt.title("stddev={:.2f}".format(stddev))

        # plt.show()

        if show:
            plt.show()
        plt.clf()
    return zero_point, stddev, slope


def find_asteroid(name, path, location="A84"):
    file_head = os.path.splitext(path)[0]
    solve_field(path)
    new_file = file_head + "_new.fits"
    data = fits.open(path)
    ep = data[0].header["JD"] + data[0].header['EXPTIME'] * 0.00000578703
    obj = Horizons(id="{}".format(name), location="{}".format(location), epochs=ep)
    ra, dec, phase_angle, r, delta, lighttime = obj.ephemerides()["RA"][0], obj.ephemerides()["DEC"][0], \
                                                obj.ephemerides()["alpha"][0], obj.ephemerides()["r"][0], \
                                                obj.ephemerides()["delta"][0], obj.ephemerides()["lighttime"][0]
    earth_coord = -obj.vectors()["x"][0], -obj.vectors()["y"][0], -obj.vectors()["z"][0]
    obj = Horizons(id="{}".format(name), location="@sun", epochs=ep)
    sun_coord = float(-1 * obj.vectors()["x"][0]), float(-1 * obj.vectors()["y"][0]), float(-1 * obj.vectors()["z"][0])
    jd = ep - lighttime / 60 / 24

    return ra, dec, phase_angle, new_file, r, delta, earth_coord, sun_coord, jd


def cone_search(files):
    check_file = files[round(len(files) / 2)]
    solve_field(check_file)
    data = fits.open(check_file)
    file_head = os.path.splitext(check_file)[0]
    new_file = file_head + "_new.fits"
    ra, dec = xy2wcs_coord(path=new_file, center=True)
    field = SkyCoord(ra * u.deg, dec * u.deg)
    ep = data[0].header["JD"] + data[0].header['EXPTIME'] * 0.00000578703
    obj = Skybot.cone_search(field, 20 / 60 * u.deg, ep)
    os.remove(new_file)
    return SerializedColumn(obj)["Number"].data.astype(str), SerializedColumn(obj)["Name"].data.astype(str)


def hg_func(x, v_alpha, G):
    x = x * math.pi / 180
    phi1 = np.exp(-3.33 * np.tan(x / 2) ** 0.63)
    phi2 = np.exp(-1.87 * np.tan(x / 2) ** 1.22)
    a = (1 - G) * phi1 + G * phi2
    H = v_alpha + 2.5 * np.log10(a)
    return H


files = sorted(glob.glob("/home/orhan/Desktop/baboquivari/rtt150/**/mp*.fit", recursive=True))
files = sorted(glob.glob("/home/orhan/Desktop/zeropoint_test/**/*15*.fits", recursive=True))
files = sorted(glob.glob("/home/orhan/t100_article/test/test/bf_*.fits", recursive=True))
print(len(files))
id = 528673
filter = "V"

county = len(files)
asteroid = []
slopes = []
asteroid_number, asteroid_name = cone_search(files)

while True:
    if str(id) in asteroid_number or str(id) in asteroid_name:
        break
    else:
        print("{} asteroid not found in image".format(id), "\n"
                                                           "Possible asteroids: \n{}".format(
            tuple(zip(asteroid_number, asteroid_name))))
        id = input()

for k in range(10):
    for path in files:
        if "new" in path:
            continue
        try:
            print("REMAINING DATA: ", county, path)

            data = fits.open(path)
            ra, dec, pa, new_path, r, delta, earth_coord, sun_coord, jd = find_asteroid(name=id,
                                                                                        path=path)  # asteroid x, y from wcs conversion in image and phase angle
            ra = 96.59175
            dec = 36.8525555556

            if os.path.exists(new_path):
                sources_in_image = source_extract(new_path)
            else:
                continue
            sources = findobjects(sources_in_image, new_path)
            x_center, y_center, fwhm = findcenter(sources_in_image, ra, dec, new_path)  # asteroid info
            median_fwhm = np.median(sources["FWHM_IMAGE"].astype(float))  # median FWHM from sextractor
            asteroid_info = {"X_center": [x_center], "Y_center": [y_center], "FWHM_IMAGE": [median_fwhm]}
            aper_flux_sum, bkg_mean, inst_mag, snr, magerr = phot(asteroid_info, new_path, length=False)
            ra_asteroid, dec_asteroid = xy2wcs_coord(x_center, y_center, new_path)
            sources["FLux_Total"], sources["Bkg_Mean"], sources["Inst_Mag"], snr_sources, sources["Mag_Err"] = phot(sources,
                                                                                                                    new_path)

            sources["Calculated_Vmag"] = Gmag2Vmag(sources["G_Mag"].astype(float), sources["BP-RP"].astype(float), filter)
            zero_point_mag, stddev, slope = zeropoint(sources, x_center, y_center, sources["Mag_Err"], plot=True, path=path,
                                                      show=False)
            # object_plot(path, sources["X_center"], sources["Y_center"], id, int(x_center), int(y_center)) # all field
            object_plot(path, int(x_center), int(y_center), id)  # crop image

            slopes.append(slope)
            calc_mag = inst_mag + zero_point_mag
            reduced_mag = calc_mag - 5 * math.log10(delta * r)
            abs_mag = hg_func(pa, reduced_mag, 0.15)

            separation_closest = sorted(
                coord.SkyCoord(ra=sources_in_image['ALPHA_J2000'], dec=sources_in_image['DELTA_J2000'], unit=(u.deg, u.deg),
                               frame='icrs').separation(
                    coord.SkyCoord(ra=ra_asteroid, dec=dec_asteroid, unit=(u.deg, u.deg))).arcsecond)[1]
            asteroid.append([new_path,
                             jd,
                             "%.4f" % inst_mag,
                             "%.4f" % calc_mag,
                             "%.4f" % reduced_mag,
                             "%.4f" % abs_mag,
                             "%.4f" % snr,
                             "%.4f" % magerr,
                             "%.4f" % stddev,
                             pa,
                             "%.4f" % slope,
                             "%.4f" % bkg_mean,
                             "%.4f" % x_center,
                             "%.4f" % y_center,
                             r,
                             delta,
                             tuple(earth_coord),
                             tuple(sun_coord),
                             "%.4f" % median_fwhm,
                             "%.4f" % (separation_closest / pxscale)
                             ])

            allinfo = np.asarray(asteroid)
            centers = Table(allinfo, names=('FileName',
                                            'JD',
                                            'Inst_Mag',
                                            'Calc_Mag',
                                            'Reduced_Mag',
                                            'Abs_MagG015',
                                            'SNR',
                                            'Magerr',
                                            'StdDev',
                                            'PhaseAngle',
                                            'Slope',
                                            'Bkg_Mean',
                                            'X_Center',
                                            'Y_Center',
                                            'r',
                                            'delta',
                                            'Earth_XYZ',
                                            'Sun_XYZ',
                                            'somefwhm_median',
                                            'closest_source(px)'
                                            ))
            ascii.write(centers, "/home/orhan/t100_article/test/test/test2/t100_test_{}.dat".format(str(k)),
            # ascii.write(centers, "test/phot_{}photutil_2059_fullt150.dat".format(str(id).replace(" ", "")),

                        overwrite=True)
            county -= 1
        except:
            continue
    asteroid = []

print(datetime.datetime.now())
