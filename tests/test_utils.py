from phops.utils import dec_to_deg, ra_to_deg


def test_ra_to_deg_hms() -> None:
    assert round(ra_to_deg("12:00:00"), 6) == 180.0


def test_dec_to_deg_dms() -> None:
    assert round(dec_to_deg("-10:30:00"), 6) == -10.5
