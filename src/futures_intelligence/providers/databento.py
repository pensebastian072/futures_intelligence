from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from futures_intelligence.config import secret_value
from futures_intelligence.providers.base import PaidDownloadBlocked, ProviderUnavailable
from futures_intelligence.providers.http import SafeHttpClient
from futures_intelligence.storage import ImmutableRawStore, ManifestStore
from futures_intelligence.types import DataRequest, FetchResult


HIST_BASE = "https://hist.databento.com/v0"


class DatabentoProvider:
    """Fail-closed Databento historical adapter.

    Metadata and cost endpoints are read-only. Time-series bytes require both the
    method argument and FI_ALLOW_PAID_DOWNLOAD=YES, preventing accidental spend.
    """

    def __init__(self, data_root: Path | str, *, client: SafeHttpClient | None = None):
        self.key = secret_value("DATABENTO_API_KEY")
        self.http = client or SafeHttpClient()
        self.raw = ImmutableRawStore(data_root)
        self.manifest = ManifestStore(data_root)

    def _auth(self):
        if not self.key:
            raise ProviderUnavailable("Databento credential is not configured")
        return (self.key, "")

    def list_schemas(self, dataset: str) -> list[str]:
        response = self.http.get(
            f"{HIST_BASE}/metadata.list_schemas",
            params={"dataset": dataset}, auth=self._auth(),
        )
        value = response.json()
        return [str(v) for v in value]

    def dataset_range(self, dataset: str) -> dict[str, Any]:
        return self.http.get(
            f"{HIST_BASE}/metadata.get_dataset_range",
            params={"dataset": dataset}, auth=self._auth(),
        ).json()

    @staticmethod
    def _params(request: DataRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "dataset": request.dataset,
            "symbols": ",".join(request.symbols),
            "schema": request.schema,
            "start": request.start,
            "end": request.end,
            "stype_in": request.input_symbology,
        }
        params.update(request.parameters)
        return params

    def estimate_cost(self, request: DataRequest) -> float:
        response = self.http.get(
            f"{HIST_BASE}/metadata.get_cost",
            params=self._params(request), auth=self._auth(),
        )
        cost = float(response.json())
        self.manifest.begin(request, cost_estimate_usd=cost)
        return cost

    def fetch_raw(self, request: DataRequest, *, allow_paid: bool = False,
                  suffix: str = ".dbn.zst") -> FetchResult:
        prior = self.manifest.successful(request.request_hash)
        if prior and Path(prior["raw_path"]).exists():
            return FetchResult(
                request.request_hash, prior["raw_path"], prior["content_hash"],
                int(prior["bytes_written"]), True,
            )
        enabled = os.environ.get("FI_ALLOW_PAID_DOWNLOAD") == "YES"
        if not (allow_paid and enabled):
            raise PaidDownloadBlocked(
                "Databento time-series download requires --allow-paid and "
                "FI_ALLOW_PAID_DOWNLOAD=YES"
            )
        cost = self.estimate_cost(request)
        self.manifest.begin(request, cost_estimate_usd=cost)
        try:
            params = self._params(request) | {"encoding": "dbn", "compression": "zstd"}
            response = self.http.get(
                f"{HIST_BASE}/timeseries.get_range", params=params, auth=self._auth()
            )
            result = self.raw.write(request, response.content, suffix=suffix)
            self.manifest.complete(result)
            return result
        except Exception as exc:
            self.manifest.fail(request.request_hash, exc)
            raise

    @staticmethod
    def dbn_to_frame(path: Path | str) -> pd.DataFrame:
        try:
            import databento as db
        except ImportError as exc:
            raise ProviderUnavailable(
                "install the optional 'databento' dependency to decode DBN"
            ) from exc
        return db.DBNStore.from_file(path).to_df().reset_index()

    def fetch_definitions(self, request: DataRequest) -> pd.DataFrame:
        if request.schema != "definition":
            raise ValueError("definition request must use schema='definition'")
        result = self.fetch_raw(request)
        return self.dbn_to_frame(result.raw_path)

    def fetch_statistics(self, request: DataRequest) -> pd.DataFrame:
        if request.schema != "statistics":
            raise ValueError("statistics request must use schema='statistics'")
        result = self.fetch_raw(request)
        return self.dbn_to_frame(result.raw_path)

