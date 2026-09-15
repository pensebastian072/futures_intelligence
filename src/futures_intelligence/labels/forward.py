from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


RETURN_HORIZONS = (1, 3, 5, 10, 21, 42, 63, 126, 252)
VOL_HORIZONS = (5, 21, 63)


def _forward_realized_vol(log_returns: pd.Series, horizon: int) -> pd.Series:
    values = log_returns.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[i + 1:i + horizon + 1]
        if len(window) == horizon and np.all(np.isfinite(window)):
            result[i] = np.sqrt((252.0 / horizon) * np.square(window).sum())
    return pd.Series(result, index=log_returns.index)


def build_forward_targets(prices: pd.DataFrame, *, date_col: str = "date",
                          close_col: str = "close",
                          return_horizons: Iterable[int] = RETURN_HORIZONS,
                          vol_horizons: Iterable[int] = VOL_HORIZONS) -> pd.DataFrame:
    frame = prices[[date_col, close_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna().sort_values(date_col).drop_duplicates(date_col).reset_index(drop=True)
    if (frame[close_col] <= 0).any():
        raise ValueError("target prices must be positive")
    log_price = np.log(frame[close_col])
    daily = log_price.diff()
    out = pd.DataFrame({"date": frame[date_col]})
    for horizon in return_horizons:
        h = int(horizon)
        out[f"return_{h}d"] = log_price.shift(-h) - log_price
        out[f"up_{h}d"] = (out[f"return_{h}d"] > 0).astype("boolean")
        out.loc[out[f"return_{h}d"].isna(), f"up_{h}d"] = pd.NA
    for horizon in vol_horizons:
        h = int(horizon)
        out[f"rv_{h}d"] = _forward_realized_vol(daily, h)
    return out

