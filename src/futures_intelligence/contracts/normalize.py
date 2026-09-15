from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STAT_TYPE_MAP = {
    "SETTLEMENT": 3,
    "SETTLEMENT_PRICE": 3,
    "CLEARED_VOLUME": 6,
    "VOLUME": 6,
    "OPEN_INTEREST": 9,
}


def _column(frame: pd.DataFrame, *names: str, default: Any = None) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _is_future(value: Any) -> bool:
    text = str(value).upper()
    return text in {"F", "FUTURE", "INSTRUMENTCLASS.FUTURE"}


def normalize_definitions(frame: pd.DataFrame, *, provider: str = "databento",
                          dataset: str = "GLBX.MDP3", root_symbol: str = "ES") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "provider", "dataset", "publisher_id", "instrument_id", "root_symbol",
            "raw_symbol", "instrument_class", "listing_timestamp", "expiration_timestamp",
            "tick_size", "multiplier", "currency", "event_timestamp", "available_at",
        ])
    out = pd.DataFrame(index=frame.index)
    out["provider"] = provider
    out["dataset"] = dataset
    out["publisher_id"] = _column(frame, "publisher_id")
    out["instrument_id"] = pd.to_numeric(_column(frame, "instrument_id"), errors="coerce")
    out["root_symbol"] = _column(frame, "asset", "root_symbol", default=root_symbol).fillna(root_symbol)
    out["raw_symbol"] = _column(frame, "raw_symbol", "symbol").astype(str).str.strip()
    out["instrument_class"] = _column(frame, "instrument_class").astype(str)
    out["listing_timestamp"] = _utc(_column(frame, "activation", "listing", "listing_timestamp"))
    out["expiration_timestamp"] = _utc(_column(frame, "expiration", "expiration_timestamp"))
    out["tick_size"] = pd.to_numeric(
        _column(frame, "min_price_increment", "tick_size"), errors="coerce"
    )
    out["multiplier"] = pd.to_numeric(
        _column(frame, "contract_multiplier", "multiplier"), errors="coerce"
    )
    out["currency"] = _column(frame, "currency", default="USD").fillna("USD")
    out["event_timestamp"] = _utc(_column(frame, "ts_event", "event_timestamp"))
    out["available_at"] = _utc(
        _column(frame, "ts_recv", "received_timestamp", "ts_event", "event_timestamp")
    )
    out = out[
        out["instrument_class"].map(_is_future)
        & out["root_symbol"].astype(str).str.upper().eq(root_symbol.upper())
        & out["instrument_id"].notna()
        & out["expiration_timestamp"].notna()
    ]
    return out.sort_values(["available_at", "instrument_id"]).reset_index(drop=True)


def _stat_code(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).upper().split(".")[-1].replace(" ", "_")
        return STAT_TYPE_MAP.get(text)


def normalize_statistics(frame: pd.DataFrame, *, provider: str = "databento",
                         dataset: str = "GLBX.MDP3") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "provider", "dataset", "instrument_id", "raw_symbol",
            "trading_reference_date", "stat_type", "price", "quantity",
            "stat_flags", "is_final", "is_actual", "event_timestamp",
            "received_timestamp", "available_at", "sequence",
        ])
    flags = pd.to_numeric(_column(frame, "stat_flags", default=0), errors="coerce").fillna(0).astype(int)
    raw_price = pd.to_numeric(_column(frame, "price"), errors="coerce")
    # Databento's DataFrame conversion normally scales prices. Raw integer inputs do not.
    if raw_price.dropna().abs().median() > 1e7:
        raw_price = raw_price / 1e9
    out = pd.DataFrame(index=frame.index)
    out["provider"] = provider
    out["dataset"] = dataset
    out["instrument_id"] = pd.to_numeric(_column(frame, "instrument_id"), errors="coerce")
    out["raw_symbol"] = _column(frame, "raw_symbol", "symbol").astype(str).str.strip()
    ref = _utc(_column(frame, "ts_ref", "trading_reference_date"))
    out["trading_reference_date"] = ref.dt.date
    out["stat_type"] = _column(frame, "stat_type").map(_stat_code).astype("Int64")
    out["price"] = raw_price
    out["quantity"] = pd.to_numeric(_column(frame, "quantity"), errors="coerce")
    out["stat_flags"] = flags
    out["is_final"] = flags.map(lambda value: bool(value & 1))
    out["is_actual"] = flags.map(lambda value: bool(value & 2))
    out["event_timestamp"] = _utc(_column(frame, "ts_event", "event_timestamp"))
    out["received_timestamp"] = _utc(_column(frame, "ts_recv", "received_timestamp"))
    out["available_at"] = out["received_timestamp"].fillna(out["event_timestamp"])
    out["sequence"] = pd.to_numeric(_column(frame, "sequence", default=0), errors="coerce").fillna(0).astype("int64")
    return out[
        out["instrument_id"].notna()
        & out["stat_type"].isin([3, 6, 9])
        & out["available_at"].notna()
    ].sort_values(["available_at", "sequence"]).reset_index(drop=True)

