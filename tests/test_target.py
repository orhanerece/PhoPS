from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from astropy.io import fits

from phops.config import load_config
from phops.errors import TargetResolutionError
from phops.target import AU_LIGHT_TIME_DAYS, TargetManager


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def read(self) -> bytes:
        return self.body


def _write_asteroid_config(tmp_path: Path, *, provider: str = "skybot") -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
fits_keywords:
  ra_key: "RA"
  dec_key: "DEC"
  date_key: "DATE-OBS"
  exposure_key: "EXPTIME"
  jd_key: "JD"
instrument:
  pixel_scale: 0.62
observatory:
  observatory_code: "A84"
photometry:
  mode: "asteroid"
  target_id: "Penelope"
  ephemeris_provider: "{provider}"
paths:
  input_dir: "input"
  temp_dir: "temp"
  index_dir: "indexes"
  solve_dir: "output"
        """.strip(),
        encoding="utf-8",
    )
    return config_path


def _header() -> fits.Header:
    header = fits.Header()
    header["JD"] = 2461125.5
    header["EXPTIME"] = 30.0
    return header


def test_skybot_resolver_returns_target_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse(
            [
                {
                    "Num": 201,
                    "Name": "Penelope",
                    "RA (deg)": 123.456,
                    "DEC (deg)": -12.345,
                    "dg (ua)": 1.5,
                    "dh (ua)": 2.5,
                    "Phase (deg)": 10.0,
                }
            ]
        )

    monkeypatch.setattr("phops.target.urlopen", fake_urlopen)
    config = load_config(_write_asteroid_config(tmp_path))

    info = TargetManager(config).resolve(_header())

    assert info.ra == pytest.approx(123.456)
    assert info.dec == pytest.approx(-12.345)
    assert info.r == pytest.approx(2.5)
    assert info.delta == pytest.approx(1.5)
    assert info.alpha == pytest.approx(10.0)
    expected_mid_exposure_jd = 2461125.5 + 15.0 / 86400.0
    assert info.jd == pytest.approx(expected_mid_exposure_jd - 1.5 * AU_LIGHT_TIME_DAYS)

    query = parse_qs(urlparse(str(captured["url"])).query)
    assert query["-name"] == ["Penelope"]
    assert query["-observer"] == ["A84"]
    assert query["-mime"] == ["json"]
    assert query["-output"] == ["all"]
    assert captured["timeout"] == 30


def test_skybot_resolver_raises_for_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return FakeResponse([])

    monkeypatch.setattr("phops.target.urlopen", fake_urlopen)
    config = load_config(_write_asteroid_config(tmp_path))

    with pytest.raises(TargetResolutionError, match="did not return"):
        TargetManager(config).resolve(_header())
