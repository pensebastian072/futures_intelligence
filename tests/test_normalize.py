import pandas as pd

from futures_intelligence.contracts import normalize_definitions, normalize_statistics


def test_definition_normalization_excludes_spreads():
    raw = pd.DataFrame([
        {"instrument_id": 1, "asset": "ES", "raw_symbol": "ESM8",
         "instrument_class": "FUTURE", "expiration": "2018-06-15T13:30:00Z",
         "ts_event": "2018-01-01T00:00:00Z"},
        {"instrument_id": 2, "asset": "ES", "raw_symbol": "ESM8-ESU8",
         "instrument_class": "FUTURE_SPREAD", "expiration": "2018-06-15T13:30:00Z",
         "ts_event": "2018-01-01T00:00:00Z"},
        {"instrument_id": 3, "asset": "NQ", "raw_symbol": "NQM8",
         "instrument_class": "FUTURE", "expiration": "2018-06-15T13:30:00Z",
         "ts_event": "2018-01-01T00:00:00Z"},
    ])
    result = normalize_definitions(raw)
    assert result["raw_symbol"].tolist() == ["ESM8"]


def test_statistics_flags_and_raw_price_scaling():
    raw = pd.DataFrame([{
        "instrument_id": 1, "symbol": "ESM8", "ts_ref": "2018-06-01",
        "stat_type": 3, "price": 2_750_250_000_000,
        "quantity": None, "stat_flags": 3, "ts_event": "2018-06-01T20:01:00Z",
        "ts_recv": "2018-06-01T20:01:01Z", "sequence": 2,
    }])
    result = normalize_statistics(raw)
    assert result.loc[0, "price"] == 2750.25
    assert bool(result.loc[0, "is_final"])
    assert bool(result.loc[0, "is_actual"])

