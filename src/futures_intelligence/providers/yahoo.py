from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from futures_intelligence.storage import ImmutableRawStore, ManifestStore
from futures_intelligence.types import DataRequest, FetchResult


class YahooProvider:
    def __init__(self, data_root: Path | str):
        self.raw = ImmutableRawStore(data_root)
        self.manifest = ManifestStore(data_root)

    def fetch_bars(self, request: DataRequest) -> pd.DataFrame:
        if len(request.symbols) != 1:
            raise ValueError("Yahoo requests require exactly one symbol")
        prior = self.manifest.successful(request.request_hash)
        if prior and Path(prior["raw_path"]).exists():
            records = json.loads(Path(prior["raw_path"]).read_text(encoding="utf-8"))
            return pd.DataFrame.from_records(records)

        import yfinance as yf

        self.manifest.begin(request)
        try:
            frame = yf.download(
                request.symbols[0], start=request.start, end=request.end,
                auto_adjust=False, actions=False, progress=False, threads=False,
            )
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame.reset_index()
            frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
            frame["vendor"] = "yahoo"
            frame["symbol"] = request.symbols[0]
            records = frame.to_dict(orient="records")
            payload = json.dumps(records, default=str, separators=(",", ":")).encode("utf-8")
            result = self.raw.write(request, payload, suffix=".json")
            self.manifest.complete(result)
            return frame
        except Exception as exc:
            self.manifest.fail(request.request_hash, exc)
            raise

