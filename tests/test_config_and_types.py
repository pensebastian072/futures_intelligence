from pathlib import Path

import pytest

from futures_intelligence.config import load_settings, secret_status
from futures_intelligence.types import DataRequest


def test_settings_are_shadow_and_use_d_drive():
    settings = load_settings()
    assert settings.mode == "SHADOW"
    assert str(settings.data_root).lower().startswith("d:")
    assert settings.target_dtes == (30, 60, 90, 180, 270, 365)


def test_request_hash_is_stable_and_secret_free():
    a = DataRequest("databento", "GLBX.MDP3", ("ES.FUT",), "statistics",
                    "2018-06-01", "2018-06-02", "parent")
    b = DataRequest("databento", "GLBX.MDP3", ("ES.FUT",), "statistics",
                    "2018-06-01", "2018-06-02", "parent")
    assert a.request_time != b.request_time or a.request_hash == b.request_hash
    assert a.request_hash == b.request_hash
    assert "key" not in str(a.safe_dict()).lower()
    with pytest.raises(ValueError):
        DataRequest("x", "y", tuple(), "z", "", "",
                    parameters={"api_key": "never"}).request_hash


def test_secret_status_does_not_reveal_value():
    assert secret_status("very-secret") == "configured"
    assert "very-secret" not in secret_status("very-secret")

