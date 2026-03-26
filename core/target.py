from astropy.time import Time
from astroquery.jplhorizons import Horizons


class TargetManager:
    def __init__(self, config):
        self.cfg = config

    def get_jd_time(self, header):
        """
        Header'dan zaman bilgisini alır ve JD'ye (float) dönüştürür.
        ISO string (2025-10-25T02:01:00) veya sayısal JD kabul eder.
        """
        date_key = self.cfg.get('fits_keywords', {}).get('date_key', 'DATE-OBS')
        exposure_key = self.cfg.get('fits_keywords', {}).get('exposure_key', 'EXPTIME')
        raw_time = header.get(date_key)
        exp_time = header.get(exposure_key)
        jd_time = header.get("JD")

        if raw_time is None:
            raise ValueError(f"❌ Keyword '{date_key}' not found in the FITS header.")
        elif jd_time is None:
            raise ValueError("❌ JD keyword not found in the FITS header.")
        try:
            # Eğer zaten sayısal (int/float) bir JD ise
            if isinstance(raw_time, (int, float)):
                return float(raw_time)
            elif isinstance(jd_time, (int, float)):
                return float(jd_time)
            # Eğer string ise (ISO formatı vb.)
            # Önce sayıya dönmeyi dener (bazı headerlarda JD string olarak saklanabilir '2460264.5')
            try:
                return float(raw_time)
            except ValueError:
                # Sayı değilse astropy ile ISO/ISOT formatından JD'ye çevir
                t = Time(raw_time, format='isot', scale='utc')
                return t.jd + exp_time * 0.00000578703

        except Exception as e:
            raise ValueError(f"❌ Time conversion error ({raw_time}): {str(e)}")

    def get_target_coordinates(self, header):
        """
        Akıllı zaman dönüşümü yaparak JPL Horizons sorgusu gerçekleştirir.
        """
        photo_cfg = self.cfg.get('photometry', {})
        mode = photo_cfg.get('mode', 'asteroid')

        if mode == 'star':
            coords = photo_cfg.get('coords')
            return float(coords[0]), float(coords[1]), None

        elif mode == 'asteroid':
            target_id = photo_cfg.get('target_id')

            # AKILLI ZAMAN DÖNÜŞÜMÜ BURADA ÇALIŞIYOR
            jd_value = self.get_jd_time(header)

            location = self.cfg.get('observatory', {}).get('observatory_code', '500')
            print(f"☄️ Querying JPL Horizons | JD: {jd_value:.6f}")

            try:
                obj = Horizons(id=str(target_id), location=location, epochs=jd_value)
                eph = obj.ephemerides()

                ra = float(eph['RA'][0])
                dec = float(eph['DEC'][0])
                lighttime =  obj.ephemerides()["lighttime"][0] / 60 / 24
                for i in eph:
                    print(i)
                return ra, dec, {
                    'r': float(eph['r'][0]),
                    'delta': float(eph['delta'][0]),
                    'alpha': float(eph['alpha'][0]),
                    'jd': jd_value - lighttime
                }
            except Exception as e:
                print(f"❌ JPL query error: {e}")
                return None, None, None