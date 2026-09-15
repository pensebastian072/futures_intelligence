from __future__ import annotations

import numpy as np
import pandas as pd


def research_backtest(returns: pd.Series, signal: pd.Series, *,
                      cost_fraction: float = 0.0005) -> tuple[pd.DataFrame, dict]:
    """Research-only close-to-next-period simulation; never routes orders."""
    if cost_fraction < 0:
        raise ValueError("cost_fraction must be non-negative")
    aligned = pd.concat(
        [returns.rename("asset_return"), signal.rename("signal")], axis=1
    ).dropna().copy()
    aligned["position"] = aligned["signal"].clip(-1, 1).shift(1).fillna(0.0)
    aligned["turnover"] = aligned["position"].diff().abs().fillna(aligned["position"].abs())
    aligned["cost"] = aligned["turnover"] * float(cost_fraction)
    aligned["strategy_return"] = aligned["position"] * aligned["asset_return"] - aligned["cost"]
    aligned["equity"] = (1.0 + aligned["strategy_return"]).cumprod()
    peak = aligned["equity"].cummax()
    aligned["drawdown"] = aligned["equity"] / peak - 1.0
    std = aligned["strategy_return"].std(ddof=1)
    metrics = {
        "n": len(aligned),
        "total_return": float(aligned["equity"].iloc[-1] - 1.0) if len(aligned) else None,
        "sharpe": float(np.sqrt(252) * aligned["strategy_return"].mean() / std)
        if len(aligned) > 1 and std > 0 else None,
        "max_drawdown": float(aligned["drawdown"].min()) if len(aligned) else None,
        "turnover": float(aligned["turnover"].sum()),
        "promoted": False,
        "mode": "SHADOW",
    }
    return aligned, metrics


def non_overlapping_forecast_backtest(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    return_col: str = "actual",
    prediction_col: str = "prediction",
    cost_fraction: float = 0.0005,
) -> tuple[pd.DataFrame, dict]:
    """Score OOS forecasts in non-overlapping, future-return episodes.

    The caller must supply a return beginning after the feature cutoff. Returns are
    treated as log returns and costs are charged on both entry and exit.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if cost_fraction < 0:
        raise ValueError("cost_fraction must be non-negative")
    required = {"date", return_col, prediction_col}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    ordered = predictions.sort_values("date").dropna(subset=[return_col, prediction_col]).copy()
    if ordered["date"].duplicated().any():
        raise ValueError("prediction dates must be unique; backtest one model at a time")
    selected = ordered.iloc[::int(horizon)].copy()
    selected["position"] = np.sign(selected[prediction_col]).astype(float)
    selected["cost"] = selected["position"].abs() * 2.0 * float(cost_fraction)
    selected["strategy_log_return"] = (
        selected["position"] * selected[return_col] - selected["cost"]
    )
    selected["equity"] = np.exp(selected["strategy_log_return"].cumsum())
    peak = selected["equity"].cummax()
    selected["drawdown"] = selected["equity"] / peak - 1.0
    std = selected["strategy_log_return"].std(ddof=1)
    periods_per_year = 252.0 / float(horizon)
    metrics = {
        "n": len(selected),
        "horizon": int(horizon),
        "cost_fraction_per_side": float(cost_fraction),
        "total_return": float(selected["equity"].iloc[-1] - 1.0) if len(selected) else None,
        "sharpe": float(
            np.sqrt(periods_per_year) * selected["strategy_log_return"].mean() / std
        ) if len(selected) > 1 and std > 0 else None,
        "max_drawdown": float(selected["drawdown"].min()) if len(selected) else None,
        "promoted": False,
        "mode": "SHADOW",
    }
    return selected, metrics
