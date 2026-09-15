from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from futures_intelligence.providers.http import SafeHttpClient
from futures_intelligence.storage import ImmutableRawStore, ManifestStore
from futures_intelligence.types import DataRequest


TFF_FUTURES_ONLY = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"


class CftcProvider:
    """Public CFTC Traders in Financial Futures, futures-only adapter."""

    def __init__(self, data_root: Path | str, *, client: SafeHttpClient | None = None):
        self.http = client or SafeHttpClient()
        self.raw = ImmutableRawStore(data_root)
        self.manifest = ManifestStore(data_root)

    def fetch_es(self, request: DataRequest) -> pd.DataFrame:
        request = replace(
            request,
            parameters={**request.parameters, "market_filter": "exact_es_v1"},
        )
        prior = self.manifest.successful(request.request_hash)
        if prior and Path(prior["raw_path"]).exists():
            rows = json.loads(Path(prior["raw_path"]).read_text(encoding="utf-8"))
            return normalize_cftc(rows)
        params = {
            "$limit": min(int(request.parameters.get("limit", 50000)), 50000),
            "$order": "report_date_as_yyyy_mm_dd",
            "$where": (
                "report_date_as_yyyy_mm_dd >= '" + request.start[:10] + "T00:00:00.000' "
                "AND report_date_as_yyyy_mm_dd < '" + request.end[:10] + "T00:00:00.000' "
                "AND upper(market_and_exchange_names) = "
                "'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE'"
            ),
        }
        self.manifest.begin(request)
        try:
            response = self.http.get(TFF_FUTURES_ONLY, params=params)
            result = self.raw.write(request, response.content, suffix=".json")
            self.manifest.complete(result)
            return normalize_cftc(response.json())
        except Exception as exc:
            self.manifest.fail(request.request_hash, exc)
            raise


def normalize_cftc(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    numeric = [
        "open_interest_all", "dealer_positions_long_all", "dealer_positions_short_all",
        "asset_mgr_positions_long", "asset_mgr_positions_short",
        "lev_money_positions_long", "lev_money_positions_short",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["report_date"] = pd.to_datetime(frame["report_date_as_yyyy_mm_dd"], errors="coerce")
    # TFF reports describe Tuesday positions and are normally released Friday. A full
    # seven-calendar-day lag is deliberately conservative for historical holiday weeks.
    ny = ZoneInfo("America/New_York")
    frame["available_at"] = frame["report_date"].map(
        lambda value: pd.Timestamp(
            datetime.combine((value + pd.Timedelta(days=7)).date(), time(18, 0), tzinfo=ny)
        ).tz_convert("UTC") if pd.notna(value) else pd.NaT
    )
    frame["provider"] = "cftc"
    return frame.sort_values("report_date").reset_index(drop=True)
