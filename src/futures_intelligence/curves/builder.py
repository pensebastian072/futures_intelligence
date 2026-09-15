from __future__ import annotations

import math
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


SETTLEMENT = 3
CLEARED_VOLUME = 6
OPEN_INTEREST = 9


def _as_utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _latest(rows: pd.DataFrame, stat_type: int) -> pd.DataFrame:
    selected = rows[rows["stat_type"].eq(stat_type)].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(["available_at", "sequence"])
        .drop_duplicates("instrument_id", keep="last")
    )


def build_contract_curve(definitions: pd.DataFrame, statistics: pd.DataFrame, *,
                         asof_timestamp, session_date: date | str, spot: float,
                         rate_by_dte=None, dividend_yield: float = 0.0,
                         day_count: float = 365.25) -> pd.DataFrame:
    """Reconstruct one point-in-time curve using only rows available by the cutoff."""
    asof = _as_utc(asof_timestamp)
    session = pd.Timestamp(session_date).date()
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be finite and positive")
    defs = definitions.copy()
    defs["available_at"] = pd.to_datetime(defs["available_at"], utc=True)
    defs["expiration_timestamp"] = pd.to_datetime(defs["expiration_timestamp"], utc=True)
    defs = defs[
        defs["available_at"].le(asof)
        & defs["expiration_timestamp"].gt(asof)
    ].sort_values("available_at").drop_duplicates("instrument_id", keep="last")

    stats = statistics.copy()
    stats["available_at"] = pd.to_datetime(stats["available_at"], utc=True)
    stats = stats[stats["available_at"].le(asof)]
    stats = stats[pd.to_datetime(stats["trading_reference_date"]).dt.date <= session]

    settles = stats[
        stats["stat_type"].eq(SETTLEMENT)
        & (pd.to_datetime(stats["trading_reference_date"]).dt.date == session)
    ]
    settles = _latest(settles, SETTLEMENT)
    volume = _latest(stats, CLEARED_VOLUME)
    oi = _latest(stats, OPEN_INTEREST)
    if settles.empty or defs.empty:
        return pd.DataFrame()

    curve = defs.merge(
        settles[["instrument_id", "price", "is_final", "is_actual", "available_at"]]
        .rename(columns={"price": "settlement", "available_at": "settlement_available_at"}),
        on="instrument_id", how="inner",
    )
    curve = curve.merge(
        volume[["instrument_id", "quantity", "trading_reference_date", "available_at"]]
        .rename(columns={"quantity": "volume", "trading_reference_date": "volume_reference_date",
                         "available_at": "volume_available_at"}),
        on="instrument_id", how="left",
    )
    curve = curve.merge(
        oi[["instrument_id", "quantity", "trading_reference_date", "available_at"]]
        .rename(columns={"quantity": "open_interest", "trading_reference_date": "oi_reference_date",
                         "available_at": "oi_available_at"}),
        on="instrument_id", how="left",
    )
    curve["asof_timestamp"] = asof
    curve["session_date"] = session
    curve["dte"] = (
        (curve["expiration_timestamp"] - asof).dt.total_seconds() / 86400.0
    )
    curve = curve[
        curve["settlement"].gt(0) & curve["dte"].gt(0)
    ].copy()
    curve["spot"] = float(spot)
    curve["tau"] = curve["dte"] / float(day_count)
    curve["basis"] = curve["settlement"] / curve["spot"] - 1.0
    curve["annualized_basis"] = np.log(curve["settlement"] / curve["spot"]) / curve["tau"]
    if callable(rate_by_dte):
        curve["rate_proxy"] = curve["dte"].map(rate_by_dte).astype(float)
    elif rate_by_dte is None:
        curve["rate_proxy"] = np.nan
    else:
        curve["rate_proxy"] = float(rate_by_dte)
    curve["dividend_yield_proxy"] = float(dividend_yield)
    curve["carry_residual"] = (
        curve["annualized_basis"] - (curve["rate_proxy"] - curve["dividend_yield_proxy"])
    )
    curve["settlement_status"] = np.select(
        [curve["is_final"] & curve["is_actual"], curve["is_final"], curve["is_actual"]],
        ["final_actual", "final_theoretical", "preliminary_actual"],
        default="preliminary_theoretical",
    )
    curve["quality_flags"] = curve.apply(
        lambda row: ";".join(filter(None, [
            "settlement_preliminary" if not row["is_final"] else "",
            "settlement_theoretical" if not row["is_actual"] else "",
            "missing_volume" if pd.isna(row.get("volume")) else "",
            "missing_open_interest" if pd.isna(row.get("open_interest")) else "",
        ])), axis=1,
    )
    return curve.sort_values(["dte", "raw_symbol"]).reset_index(drop=True)


def standardize_curve(contract_curve: pd.DataFrame, target_dtes: Iterable[int]) -> pd.DataFrame:
    """Interpolate log price only between listed contracts; never extrapolate."""
    if contract_curve.empty:
        return pd.DataFrame()
    curve = contract_curve.sort_values("dte").drop_duplicates("dte", keep="last")
    rows = []
    for target in target_dtes:
        lower = curve[curve["dte"].le(target)].tail(1)
        upper = curve[curve["dte"].ge(target)].head(1)
        if lower.empty or upper.empty:
            continue
        near, far = lower.iloc[0], upper.iloc[0]
        if near["dte"] == far["dte"]:
            weight = 0.0
            log_price = math.log(float(near["settlement"]))
        else:
            weight = (float(target) - float(near["dte"])) / (float(far["dte"]) - float(near["dte"]))
            log_price = (
                (1.0 - weight) * math.log(float(near["settlement"]))
                + weight * math.log(float(far["settlement"]))
            )
        price = math.exp(log_price)
        spot = float(near["spot"])
        tau = float(target) / 365.25
        rate = (
            float(near["rate_proxy"])
            if near["dte"] == far["dte"]
            else (1.0 - weight) * float(near["rate_proxy"]) + weight * float(far["rate_proxy"])
        )
        dividend = float(near["dividend_yield_proxy"])
        annualized = math.log(price / spot) / tau
        quality = set(filter(None, str(near.get("quality_flags", "")).split(";")))
        quality.update(filter(None, str(far.get("quality_flags", "")).split(";")))
        rows.append({
            "asof_timestamp": near["asof_timestamp"],
            "root_symbol": near["root_symbol"],
            "target_dte": int(target),
            "interpolated_log_price": log_price,
            "interpolated_price": price,
            "basis": price / spot - 1.0,
            "annualized_basis": annualized,
            "rate_proxy": rate,
            "dividend_yield_proxy": dividend,
            "carry_residual": annualized - (rate - dividend),
            "near_contract": near["raw_symbol"],
            "far_contract": far["raw_symbol"],
            "interpolation_weight": weight,
            "near_dte": float(near["dte"]),
            "far_dte": float(far["dte"]),
            "interpolation_span": float(far["dte"] - near["dte"]),
            "quality_flags": ";".join(sorted(quality)),
        })
    return pd.DataFrame(rows).sort_values("target_dte").reset_index(drop=True) if rows else pd.DataFrame()

