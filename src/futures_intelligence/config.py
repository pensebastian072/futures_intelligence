from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "base.toml"


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    mode: str
    timezone: str
    cutoff_time: time
    data_root: Path
    validation_root: Path
    target_dtes: tuple[int, ...]
    primary_dtes: tuple[int, ...]
    day_count: float
    raw: dict

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def ensure_shadow(self) -> None:
        if self.mode.upper() != "SHADOW":
            raise ConfigurationError("futures_intelligence must run in SHADOW mode")


def load_settings(path: Path | str = DEFAULT_CONFIG) -> Settings:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    project = raw["project"]
    curve = raw["curve"]
    hh, mm = (int(part) for part in project["cutoff_time"].split(":"))
    settings = Settings(
        mode=str(project["mode"]),
        timezone=str(project["timezone"]),
        cutoff_time=time(hh, mm),
        data_root=Path(project["data_root"]),
        validation_root=Path(project["canonical_validation_root"]),
        target_dtes=tuple(int(v) for v in curve["target_dtes"]),
        primary_dtes=tuple(int(v) for v in curve["primary_dtes"]),
        day_count=float(curve["day_count"]),
        raw=raw,
    )
    settings.ensure_shadow()
    return settings


def secret_value(env_name: str, *, json_file: Path | None = None,
                 json_key: str = "api_key") -> str | None:
    """Resolve a secret without logging, returning, or persisting its source metadata."""
    value = os.environ.get(env_name)
    if value:
        return value.strip()
    if json_file and json_file.exists():
        try:
            candidate = json.loads(json_file.read_text(encoding="utf-8")).get(json_key)
        except (OSError, ValueError, TypeError):
            return None
        return str(candidate).strip() if candidate else None
    return None


def secret_status(value: str | None) -> str:
    return "configured" if value else "missing"

