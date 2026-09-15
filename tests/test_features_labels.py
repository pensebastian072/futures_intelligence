import numpy as np
import pandas as pd

from futures_intelligence.features import build_curve_features, continuous_rate_curve
from futures_intelligence.labels import build_forward_targets


def test_rate_conversion_and_curve_features():
    rate = continuous_rate_curve({90: 5.0, 180: 5.5, 365: 6.0})
    assert np.isclose(rate(90), np.log1p(0.05))
    dates = pd.date_range("2020-01-01", periods=25, tz="UTC")
    rows = []
    for i, stamp in enumerate(dates):
        for dte in (30, 60, 90, 180, 365):
            rows.append({"asof_timestamp": stamp, "target_dte": dte,
                         "basis": dte / 10000 + i / 100000,
                         "annualized_basis": dte / 1000 + i / 10000,
                         "carry_residual": dte / 2000 + i / 10000})
    features = build_curve_features(pd.DataFrame(rows))
    assert "basis_slope_90d_30d" in features
    assert "basis_curvature_30_90_180" in features
    assert features["basis_30d_change_1d"].iloc[1] > 0


def test_forward_targets_start_after_t():
    prices = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10),
        "close": np.exp(np.arange(10) * 0.01),
    })
    result = build_forward_targets(prices, return_horizons=[1, 3], vol_horizons=[3])
    assert np.isclose(result.loc[0, "return_1d"], 0.01)
    assert np.isclose(result.loc[0, "return_3d"], 0.03)
    assert np.isclose(result.loc[0, "rv_3d"], np.sqrt(252 * 0.01**2))
    assert pd.isna(result.loc[9, "return_1d"])

