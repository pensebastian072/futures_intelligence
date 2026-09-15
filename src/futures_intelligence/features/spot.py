from __future__ import annotations

import numpy as np
import pandas as pd


def _daily(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    required = {"date", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing {prefix} columns: {sorted(missing)}")
    out = frame[["date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna().sort_values("date").drop_duplicates("date")
    return out.rename(columns={"close": f"{prefix}_close"})


def build_spot_context_features(spx: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Build close-known context features using trailing observations only."""
    out = _daily(spx, prefix="spx").merge(
        _daily(vix, prefix="vix"), on="date", how="left", validate="one_to_one"
    )
    if (out["spx_close"] <= 0).any():
        raise ValueError("SPX closes must be positive")
    log_spx = np.log(out["spx_close"])
    daily = log_spx.diff()
    for window in (5, 21, 63, 126, 252):
        out[f"spx_momentum_{window}d"] = log_spx.diff(window)
    for window in (5, 21, 63):
        out[f"spx_realized_vol_{window}d"] = daily.rolling(window).std(ddof=1) * np.sqrt(252)
    out["vix_level"] = out["vix_close"] / 100.0
    out["vix_change_5d"] = out["vix_close"].diff(5) / 100.0
    out["vix_change_21d"] = out["vix_close"].diff(21) / 100.0
    return out


SPOT_CONTEXT_COLUMNS = [
    "spx_momentum_5d", "spx_momentum_21d", "spx_momentum_63d",
    "spx_momentum_126d", "spx_momentum_252d",
    "spx_realized_vol_5d", "spx_realized_vol_21d", "spx_realized_vol_63d",
    "vix_level", "vix_change_5d", "vix_change_21d",
]


def build_forward_open_returns(
    bars: pd.DataFrame, *, horizons: tuple[int, ...] = (5, 21, 63)
) -> pd.DataFrame:
    """Return next-session-open to future-open log returns for delayed backtests."""
    required = {"date", "open"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"missing execution columns: {sorted(missing)}")
    out = bars[["date", "open"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out = out.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if (out["open"] <= 0).any():
        raise ValueError("execution opens must be positive")
    log_open = np.log(out["open"])
    result = pd.DataFrame({"date": out["date"]})
    for horizon in horizons:
        h = int(horizon)
        result[f"spy_open_return_{h}d"] = log_open.shift(-(h + 1)) - log_open.shift(-1)
    return result
