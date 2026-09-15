from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from futures_intelligence.config import Settings, secret_value
from futures_intelligence.providers.base import ProviderUnavailable
from futures_intelligence.providers.http import SafeHttpClient
from futures_intelligence.storage import ImmutableRawStore, ManifestStore
from futures_intelligence.types import DataRequest, FetchResult


class LseProvider:
    def __init__(self, settings: Settings, *, client: SafeHttpClient | None = None):
        legacy = Path(settings.raw["lse"]["legacy_secret_file"])
        self.key = secret_value("LSE_API_KEY", json_file=legacy)
        self.base_url = settings.raw["lse"]["base_url"].rstrip("/")
        self.http = client or SafeHttpClient()
        self.raw = ImmutableRawStore(settings.data_root)
        self.manifest = ManifestStore(settings.data_root)

    def _headers(self) -> dict[str, str]:
        if not self.key:
            raise ProviderUnavailable("LSE credential is not configured")
        return {"x-api-key": self.key, "Accept": "application/json"}

    def get_json(self, request: DataRequest, endpoint: str,
                 params: dict[str, Any] | None = None) -> tuple[Any, FetchResult]:
        prior = self.manifest.successful(request.request_hash)
        if prior and Path(prior["raw_path"]).exists():
            payload = Path(prior["raw_path"]).read_bytes()
            result = FetchResult(
                request.request_hash, prior["raw_path"], prior["content_hash"],
                int(prior["bytes_written"]), True,
            )
            return json.loads(payload), result
        self.manifest.begin(request)
        try:
            response = self.http.get(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                headers=self._headers(), params=params,
            )
            payload = response.content
            result = self.raw.write(request, payload, suffix=".json")
            self.manifest.complete(result)
            return response.json(), result
        except Exception as exc:
            self.manifest.fail(request.request_hash, exc)
            raise

    def audit(self) -> dict[str, Any]:
        meta_request = DataRequest("lse", "meta", tuple(), "json", "", "")
        catalog_request = DataRequest(
            "lse", "catalog", ("ES.F",), "json", "", "",
            parameters={"dataset": "futures"},
        )
        meta, meta_result = self.get_json(meta_request, "/meta")
        catalog, catalog_result = self.get_json(
            catalog_request, "/catalog", params={"dataset": "futures"}
        )
        es_rows = [row for row in catalog if row.get("symbol") == "ES.F"]
        return {
            "configured": True,
            "meta_cached": meta_result.reused,
            "catalog_cached": catalog_result.reused,
            "datasets": sorted(meta.get("datasets", [])) if isinstance(meta, dict) else [],
            "es_rows": es_rows,
            "dated_contracts_present": any(
                re.fullmatch(r"ES[HMUZ]\d{1,4}", str(row.get("symbol", "")))
                for row in catalog
            ),
        }

    def fetch_series(self, request: DataRequest) -> pd.DataFrame:
        if len(request.symbols) != 1:
            raise ValueError("LSE series requests require exactly one symbol")
        params = {
            "symbol": request.symbols[0], "start": request.start, "end": request.end
        }
        rows, _ = self.get_json(request, "/series", params=params)
        return pd.DataFrame(rows)
