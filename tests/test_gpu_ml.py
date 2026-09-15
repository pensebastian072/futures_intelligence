import numpy as np
import pandas as pd
import pytest

from futures_intelligence.backtest import non_overlapping_forecast_backtest
from futures_intelligence.features import build_forward_open_returns, build_spot_context_features
from futures_intelligence.models import XgbResearchConfig, walk_forward_xgb
from futures_intelligence.rl import check_rl_eligibility


def test_spot_features_do_not_look_forward():
    dates = pd.bdate_range("2020-01-01", periods=300)
    spx = pd.DataFrame({"date": dates, "close": np.arange(300) + 3000.0})
    vix = pd.DataFrame({"date": dates, "close": np.arange(300) / 10 + 15.0})
    original = build_spot_context_features(spx, vix)
    changed = spx.copy()
    changed.loc[changed.index[-1], "close"] *= 2
    perturbed = build_spot_context_features(changed, vix)
    pd.testing.assert_series_equal(
        original.iloc[-2].drop(labels="spx_close"),
        perturbed.iloc[-2].drop(labels="spx_close"),
    )


def test_forward_open_returns_begin_next_session():
    bars = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=5),
        "open": [100.0, 110.0, 121.0, 133.1, 146.41],
    })
    result = build_forward_open_returns(bars, horizons=(2,))
    assert np.isclose(result.loc[0, "spy_open_return_2d"], np.log(133.1 / 110.0))


def test_walk_forward_xgb_outputs_outer_predictions_on_cpu():
    dates = pd.bdate_range("2010-06-01", "2017-12-31")
    rng = np.random.default_rng(7)
    x = rng.normal(size=len(dates))
    frame = pd.DataFrame({
        "date": dates,
        "spot": x,
        "curve": rng.normal(size=len(dates)),
        "target": np.roll(x, -5),
    })
    predictions, metrics = walk_forward_xgb(
        frame,
        date_col="date",
        target_col="target",
        feature_sets={"D": ["spot"], "E": ["spot", "curve"]},
        horizon=5,
        device="cpu",
        config=XgbResearchConfig(n_estimators=5, max_depth=2),
        first_test_year=2016,
        last_test_year=2017,
    )
    assert set(predictions["model"]) == {"D", "E"}
    assert predictions.groupby(["test_year", "model"]).size().groupby(level=0).nunique().eq(1).all()
    assert predictions["date"].dt.year.min() == 2016
    assert metrics["device"].eq("cpu").all()
    assert predictions["promoted"].eq(False).all()


def test_non_overlapping_backtest_and_rl_fail_closed():
    frame = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=12),
        "actual": np.repeat(0.01, 12),
        "prediction": np.repeat(0.02, 12),
    })
    detail, metrics = non_overlapping_forecast_backtest(
        frame, horizon=5, cost_fraction=0.001
    )
    assert len(detail) == 3
    assert metrics["promoted"] is False
    eligibility = check_rl_eligibility(
        has_dated_contract_curve=False,
        supervised_all_horizons_pass=True,
        oos_transitions=2_000,
        transaction_costs_enabled=True,
        holdout_touched=False,
    )
    assert eligibility.eligible is False
    assert "dated-contract" in eligibility.reasons[0]


def test_non_overlapping_backtest_rejects_interleaved_models():
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")],
        "actual": [0.01, 0.01],
        "prediction": [0.02, -0.02],
        "model": ["D", "E"],
    })
    with pytest.raises(ValueError, match="one model at a time"):
        non_overlapping_forecast_backtest(frame, horizon=5)
