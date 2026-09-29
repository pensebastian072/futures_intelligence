from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path

from futures_intelligence.config import Settings, secret_status, secret_value


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def environment_audit(settings: Settings) -> dict:
    lse_file = Path(settings.raw["lse"]["legacy_secret_file"])
    return {
        "mode": settings.mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data_root": str(settings.data_root),
        "data_root_exists": settings.data_root.exists(),
        "credentials": {
            "lse": secret_status(secret_value("LSE_API_KEY", json_file=lse_file)),
            "databento": secret_status(secret_value("DATABENTO_API_KEY")),
            "cme_datamine": "configured" if (
                os.environ.get("CME_DATAMINE_API_ID")
                and os.environ.get("CME_DATAMINE_API_PASSWORD")
            ) else "missing",
        },
        "packages": {name: _version(name) for name in (
            "numpy", "pandas", "pyarrow", "duckdb", "requests", "yfinance",
            "scipy", "scikit-learn", "statsmodels", "databento", "xgboost", "pytest",
        )},
        "cutoff": f"{settings.cutoff_time.isoformat()} {settings.timezone}",
        "target_dtes": list(settings.target_dtes),
        "validation_root_exists": settings.validation_root.exists(),
        "broker_execution": "disabled",
        "promoted": False,
    }
