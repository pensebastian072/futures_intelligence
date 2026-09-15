from futures_intelligence.providers.cme import parse_daily_settlement_text


def _line(values):
    widths = (10, 13, 13, 13, 13, 12, 13, 12, 15, 12, 12)
    return "".join(str(value).ljust(width)[:width] for value, width in zip(values, widths))


def test_fixed_width_settlement_parser():
    text = "HEADER\n" + _line([
        "JUN 26", "6000.25", "6010.00B", "5980.00A", "6005.25",
        "6004.50", "10.25", "123456", "5994.25", "100000", "250000",
    ])
    result = parse_daily_settlement_text(text, trade_date="2026-08-18")
    assert len(result) == 1
    assert result.loc[0, "settlement"] == 6004.5
    assert result.loc[0, "open_interest"] == 250000
    assert result.loc[0, "root_symbol"] == "ES"

