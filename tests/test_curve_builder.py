from datetime import date

import numpy as np
import pandas as pd

from futures_intelligence.curves import build_contract_curve, standardize_curve


ASOF = pd.Timestamp("2018-06-01T22:00:00Z")


def _definitions():
    expirations = ["2018-06-15", "2018-09-21", "2018-12-21", "2019-03-15", "2019-09-20"]
    symbols = ["ESM8", "ESU8", "ESZ8", "ESH9", "ESU9"]
    rows = []
    for instrument, (symbol, expiry) in enumerate(zip(symbols, expirations), 1):
        rows.append({
            "provider": "fixture", "dataset": "x", "publisher_id": 1,
            "instrument_id": instrument, "root_symbol": "ES", "raw_symbol": symbol,
            "instrument_class": "FUTURE", "listing_timestamp": pd.Timestamp("2017-01-01", tz="UTC"),
            "expiration_timestamp": pd.Timestamp(expiry, tz="UTC"),
            "tick_size": 0.25, "multiplier": 50, "currency": "USD",
            "event_timestamp": pd.Timestamp("2017-01-01", tz="UTC"),
            "available_at": pd.Timestamp("2017-01-01", tz="UTC"),
        })
    return pd.DataFrame(rows)


def _statistics():
    rows = []
    for instrument, price in enumerate([2750, 2760, 2770, 2780, 2800], 1):
        rows.extend([
            {"instrument_id": instrument, "raw_symbol": "", "trading_reference_date": date(2018, 6, 1),
             "stat_type": 3, "price": price, "quantity": np.nan, "stat_flags": 2,
             "is_final": False, "is_actual": True,
             "event_timestamp": pd.Timestamp("2018-06-01T20:00:00Z"),
             "received_timestamp": pd.Timestamp("2018-06-01T20:00:01Z"),
             "available_at": pd.Timestamp("2018-06-01T20:00:01Z"), "sequence": 1},
            # Later final revision must not enter the 18:00 ET snapshot.
            {"instrument_id": instrument, "raw_symbol": "", "trading_reference_date": date(2018, 6, 1),
             "stat_type": 3, "price": price + 99, "quantity": np.nan, "stat_flags": 3,
             "is_final": True, "is_actual": True,
             "event_timestamp": pd.Timestamp("2018-06-02T00:01:00Z"),
             "received_timestamp": pd.Timestamp("2018-06-02T00:01:01Z"),
             "available_at": pd.Timestamp("2018-06-02T00:01:01Z"), "sequence": 2},
            {"instrument_id": instrument, "raw_symbol": "", "trading_reference_date": date(2018, 5, 31),
             "stat_type": 6, "price": np.nan, "quantity": instrument * 1000, "stat_flags": 0,
             "is_final": False, "is_actual": False,
             "event_timestamp": pd.Timestamp("2018-06-01T10:00:00Z"),
             "received_timestamp": pd.Timestamp("2018-06-01T10:00:01Z"),
             "available_at": pd.Timestamp("2018-06-01T10:00:01Z"), "sequence": 1},
            {"instrument_id": instrument, "raw_symbol": "", "trading_reference_date": date(2018, 5, 31),
             "stat_type": 9, "price": np.nan, "quantity": instrument * 2000, "stat_flags": 0,
             "is_final": False, "is_actual": False,
             "event_timestamp": pd.Timestamp("2018-06-01T10:00:00Z"),
             "received_timestamp": pd.Timestamp("2018-06-01T10:00:01Z"),
             "available_at": pd.Timestamp("2018-06-01T10:00:01Z"), "sequence": 1},
        ])
    return pd.DataFrame(rows)


def test_asof_revision_and_liquidity_join():
    curve = build_contract_curve(
        _definitions(), _statistics(), asof_timestamp=ASOF,
        session_date="2018-06-01", spot=2734.62,
        rate_by_dte=lambda _: 0.02, dividend_yield=0.018,
    )
    assert curve["settlement"].tolist() == [2750, 2760, 2770, 2780, 2800]
    assert curve["settlement_status"].eq("preliminary_actual").all()
    assert curve.loc[0, "open_interest"] == 2000
    assert curve["settlement_available_at"].max() <= ASOF


def test_interpolation_and_no_extrapolation():
    contract = build_contract_curve(
        _definitions(), _statistics(), asof_timestamp=ASOF,
        session_date="2018-06-01", spot=2734.62,
        rate_by_dte=lambda _: 0.02, dividend_yield=0.018,
    )
    standardized = standardize_curve(contract, [1, 30, 60, 90, 180, 270, 365, 900])
    assert standardized["target_dte"].tolist() == [30, 60, 90, 180, 270, 365]
    assert standardized["interpolated_price"].gt(0).all()
    assert standardized["interpolation_weight"].between(0, 1).all()

