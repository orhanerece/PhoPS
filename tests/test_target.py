from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from astropy.io import fits

from phops.config import load_config
from phops.errors import TargetResolutionError
from phops.target import TargetManager


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


def _write_asteroid_config(tmp_path: Path, *, provider: str = "miriade") -> Path:
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


def test_miriade_ephemcc_returns_target_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "sso": {"num": 201, "name": "Penelope", "type": "asteroid"},
                "data": [
                    {
                        "Date": 2461125.50017361,
                        "RA": 123.456,
                        "DEC": -12.345,
                        "Dobs": 1.5,
                        "LightTime": 21.6,
                        "Dhelio": 2.5,
                        "Phase": 10.0,
                    }
                ],
            }
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
    assert info.jd == pytest.approx(expected_mid_exposure_jd - 21.6 / 60.0 / 24.0)

    query = parse_qs(urlparse(str(captured["url"])).query)
    assert query["-name"] == ["a:Penelope"]
    assert query["-observer"] == ["A84"]
    assert query["-nbd"] == ["1"]
    assert query["-teph"] == ["1"]
    assert query["-tcoor"] == ["5"]
    assert query["-mime"] == ["json"]
    assert query["-output"] == ["--jd,--lighttime"]
    assert captured["timeout"] == 30


def test_miriade_ephemcc_raises_for_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return FakeResponse({"data": []})

    monkeypatch.setattr("phops.target.urlopen", fake_urlopen)
    config = load_config(_write_asteroid_config(tmp_path))

    with pytest.raises(TargetResolutionError, match="did not return"):
        TargetManager(config).resolve(_header())
