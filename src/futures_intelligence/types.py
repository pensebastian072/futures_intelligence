from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DataRequest:
    provider: str
    dataset: str
    symbols: tuple[str, ...]
    schema: str
    start: str
    end: str
    input_symbology: str = "raw_symbol"
    request_time: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    parameters: dict[str, Any] = field(default_factory=dict)

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("request_time", None)
        forbidden = {"api_key", "key", "password", "secret", "authorization"}
        lowered = {str(k).lower() for k in payload["parameters"]}
        if forbidden & lowered:
            raise ValueError("request parameters must not contain credentials")
        return payload

    @property
    def request_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def safe_dict(self) -> dict[str, Any]:
        value = self.canonical_payload()
        value["request_time"] = self.request_time
        value["request_hash"] = self.request_hash
        return value


@dataclass(frozen=True)
class FetchResult:
    request_hash: str
    raw_path: str
    content_hash: str
    bytes_written: int
    reused: bool

