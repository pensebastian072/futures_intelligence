from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from futures_intelligence.config import Settings, secret_value
from futures_intelligence.providers.base import PaidDownloadBlocked, ProviderUnavailable
from futures_intelligence.providers.http import SafeHttpClient


class CmeDataMineProvider:
    """Authenticated DataMine file discovery; downloads are separately gated."""

    def __init__(self, settings: Settings, *, client: SafeHttpClient | None = None):
        self.api_id = secret_value("CME_DATAMINE_API_ID")
        self.password = secret_value("CME_DATAMINE_API_PASSWORD")
        self.auth_url = settings.raw["cme"]["auth_url"]
        self.list_url = settings.raw["cme"]["list_url"]
        self.http = client or SafeHttpClient()

    def _token(self) -> str:
        if not self.api_id or not self.password:
            raise ProviderUnavailable("CME DataMine credentials are not configured")
        response = self.http.post(
            self.auth_url, data={"grant_type": "client_credentials"},
            auth=(self.api_id, self.password),
        )
        token = response.json().get("access_token")
        if not token:
            raise ProviderUnavailable("CME DataMine did not return an access token")
        return str(token)

    def list_entitled_files(self, *, period_date: str,
                            product_code: str = "ES") -> dict[str, Any]:
        token = self._token()
        return self.http.get(
            self.list_url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "futures-intelligence/0.1"},
            params={"category_code": "EOD", "exchange_code": "XCME",
                    "product_code": product_code, "foi_indicator": "FUT",
                    "period_date": period_date, "limit": 1000},
        ).json()

    def download_file(self, url: str, *, allow_download: bool = False) -> bytes:
        if not (allow_download and os.environ.get("FI_ALLOW_CME_DOWNLOAD") == "YES"):
            raise PaidDownloadBlocked(
                "CME file download requires --allow-download and FI_ALLOW_CME_DOWNLOAD=YES"
            )
        token = self._token()
        return self.http.get(url, headers={"Authorization": f"Bearer {token}"}).content


_PRICE_MARKER = re.compile(r"([0-9.]+)([AB])?$")


def _number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned in {"-", "UNCH"}:
        return None
    match = _PRICE_MARKER.match(cleaned)
    if not match:
        return None
    return float(match.group(1))


def parse_daily_settlement_text(text: str, *, trade_date: str,
                                root_symbol: str = "ES") -> pd.DataFrame:
    """Parse the documented 11-column CME fixed-width futures layout.

    The parser intentionally accepts only monthly futures rows. Titles, option strikes,
    spreads, and malformed lines are ignored rather than guessed.
    """
    rows: list[dict[str, Any]] = []
    widths = (10, 13, 13, 13, 13, 12, 13, 12, 15, 12, 12)
    cuts = []
    total = 0
    for width in widths:
        cuts.append((total, total + width))
        total += width
    month_pattern = re.compile(r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{2}$")
    for line in io.StringIO(text):
        if len(line.rstrip("\n\r")) < sum(widths[:8]):
            continue
        fields = [line[a:b].strip() for a, b in cuts]
        if not month_pattern.match(fields[0].upper()):
            continue
        rows.append({
            "provider": "cme_datamine",
            "root_symbol": root_symbol,
            "trade_date": pd.Timestamp(trade_date).date(),
            "contract_month": fields[0].upper(),
            "open": _number(fields[1]), "high": _number(fields[2]),
            "low": _number(fields[3]), "last": _number(fields[4]),
            "settlement": _number(fields[5]), "change": _number(fields[6]),
            "estimated_volume": _number(fields[7]),
            "prior_settlement": _number(fields[8]),
            "prior_volume": _number(fields[9]),
            "open_interest": _number(fields[10]),
        })
    return pd.DataFrame(rows)

