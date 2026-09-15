import os

import numpy as np
import pandas as pd
import pytest

from futures_intelligence.backtest import research_backtest
from futures_intelligence.config import load_settings
from futures_intelligence.providers.base import PaidDownloadBlocked
from futures_intelligence.providers.databento import DatabentoProvider
from futures_intelligence.types import DataRequest
from futures_intelligence.validation import canonical_gate


def test_databento_download_needs_two_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("FI_ALLOW_PAID_DOWNLOAD", raising=False)
    request = DataRequest("databento", "GLBX.MDP3", ("ES.FUT",), "statistics",
                          "2018-06-01", "2018-06-02", "parent")
    provider = DatabentoProvider(tmp_path)
    with pytest.raises(PaidDownloadBlocked):
        provider.fetch_raw(request, allow_paid=True)
    monkeypatch.setenv("FI_ALLOW_PAID_DOWNLOAD", "YES")
    with pytest.raises(PaidDownloadBlocked):
        provider.fetch_raw(request, allow_paid=False)


def test_canonical_gate_is_shadow_even_when_available():
    settings = load_settings()
    result = canonical_gate(
        np.tile([0.01, -0.005, 0.008, -0.002, 0.004], 20),
        n_trials=5, validation_root=settings.validation_root,
    )
    assert result["available"]
    assert result["promoted"] is False
    assert result["mode"] == "SHADOW"


def test_research_backtest_delays_signal_and_stays_shadow():
    returns = pd.Series([0.1, 0.1, -0.1], index=pd.date_range("2020-01-01", periods=3))
    signal = pd.Series([1.0, -1.0, 1.0], index=returns.index)
    detail, metrics = research_backtest(returns, signal, cost_fraction=0)
    assert detail["position"].tolist() == [0.0, 1.0, -1.0]
    assert metrics["promoted"] is False

