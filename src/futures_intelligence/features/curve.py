from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


def continuous_rate_curve(quoted_percent_by_days: Mapping[int, float]) -> Callable[[float], float]:
    """Piecewise-linear quoted yields, converted to continuous annual rates."""
    if not quoted_percent_by_days:
        raise ValueError("at least one rate point is required")
    days = np.array(sorted(quoted_percent_by_days), dtype=float)
    rates = np.array([quoted_percent_by_days[int(d)] for d in days], dtype=float)
    if np.any(~np.isfinite(rates)) or np.any(rates <= -100):
        raise ValueError("rates must be finite percentages above -100")
    continuous = np.log1p(rates / 100.0)

    def interpolate(dte: float) -> float:
        return float(np.interp(float(dte), days, continuous))

    return interpolate


def _pivot(curve: pd.DataFrame, value: str) -> pd.DataFrame:
    wide = curve.pivot(index="asof_timestamp", columns="target_dte", values=value)
    return wide.sort_index().rename(columns=lambda d: f"{value}_{int(d)}d")


def build_curve_features(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    levels = pd.concat(
        [_pivot(curve, name) for name in ("basis", "annualized_basis", "carry_residual")],
        axis=1,
    )
    out = levels.copy()
    for prefix in ("basis", "annualized_basis", "carry_residual"):
        for far in (60, 90, 180, 365):
            near_col, far_col = f"{prefix}_30d", f"{prefix}_{far}d"
            if near_col in out and far_col in out:
                out[f"{prefix}_slope_{far}d_30d"] = out[far_col] - out[near_col]
        triples = ((30, 90, 180), (90, 180, 365))
        for near, middle, far in triples:
            cols = [f"{prefix}_{v}d" for v in (near, middle, far)]
            if all(col in out for col in cols):
                weight = (middle - near) / (far - near)
                line = (1.0 - weight) * out[cols[0]] + weight * out[cols[2]]
                out[f"{prefix}_curvature_{near}_{middle}_{far}"] = out[cols[1]] - line
    derived = {}
    base_columns = list(out.columns)
    for column in base_columns:
        for lag in (1, 5, 10, 21):
            derived[f"{column}_change_{lag}d"] = out[column] - out[column].shift(lag)
    for column in [c for c in out if "slope" in c]:
        derived[f"{column}_acceleration_5d"] = out[column].diff().diff(5)
    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
    out.index.name = "asof_timestamp"
    return out.reset_index()


def trailing_dividend_yield(dividends: pd.DataFrame, prices: pd.Series,
                            *, window_days: int = 365) -> pd.Series:
    """Point-in-time trailing cash dividend proxy using effective dates only."""
    if dividends.empty:
        return pd.Series(np.nan, index=prices.index)
    div = dividends.copy()
    div["effective_date"] = pd.to_datetime(div["effective_date"])
    div = div.set_index("effective_date")["cash_amount"].astype(float).sort_index()
    daily = div.groupby(level=0).sum().reindex(prices.index, fill_value=0.0)
    trailing = daily.rolling(f"{window_days}D", closed="both").sum()
    return trailing / prices
