from __future__ import annotations

from typing import Protocol

import pandas as pd

from futures_intelligence.types import DataRequest


class ContractSource(Protocol):
    def fetch_definitions(self, request: DataRequest) -> pd.DataFrame: ...


class StatisticsSource(Protocol):
    def fetch_statistics(self, request: DataRequest) -> pd.DataFrame: ...


class SpotSource(Protocol):
    def fetch_bars(self, request: DataRequest) -> pd.DataFrame: ...


class ReferenceSource(Protocol):
    def fetch_series(self, request: DataRequest) -> pd.DataFrame: ...


class ProviderUnavailable(RuntimeError):
    pass


class PaidDownloadBlocked(RuntimeError):
    pass

