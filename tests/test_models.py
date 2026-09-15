import numpy as np
import pandas as pd

from futures_intelligence.models import compare_models


def test_model_ladder_uses_same_outer_test_rows():
    dates = pd.bdate_range("2010-06-01", "2018-12-31")
    x = np.sin(np.arange(len(dates)) / 30)
    frame = pd.DataFrame({
        "date": dates,
        "target": np.roll(x, -5),
        "momentum": x,
        "volatility": np.abs(x),
        "spot_context": x * 0.5,
        "curve": x * 0.25,
    })
    result = compare_models(
        frame, date_col="date", target_col="target",
        momentum_columns=["momentum"], volatility_columns=["volatility"],
        spot_columns=["momentum", "volatility", "spot_context"],
        curve_columns=["curve"], horizon=5, first_test_year=2016, last_test_year=2018,
    )
    assert set(result["model"]) == {
        "A_mean", "B_momentum", "C_momentum_volatility", "D_spot", "E_spot_curve"
    }
    assert result.groupby("test_year")["n_test"].nunique().eq(1).all()
