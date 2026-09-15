import pandas as pd

from futures_intelligence.providers.cftc import normalize_cftc


def test_cftc_historical_availability_is_conservatively_lagged():
    result = normalize_cftc([{
        "market_and_exchange_names": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
        "report_date_as_yyyy_mm_dd": "2026-08-11T00:00:00.000",
        "open_interest_all": "123456",
        "dealer_positions_long_all": "10",
        "dealer_positions_short_all": "20",
    }])
    assert result.loc[0, "open_interest_all"] == 123456
    assert result.loc[0, "available_at"] == pd.Timestamp("2026-08-18T22:00:00Z")
    assert not result.loc[0, "market_and_exchange_names"].startswith("MICRO")
