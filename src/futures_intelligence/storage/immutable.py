from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from futures_intelligence.types import DataRequest, FetchResult


class ImmutableRawStore:
    """Content-addressed raw storage. Existing bytes are never overwritten."""

    def __init__(self, data_root: Path | str):
        self.root = Path(data_root) / "raw"

    def path_for(self, request: DataRequest, content_hash: str, suffix: str) -> Path:
        stamp = datetime.fromisoformat(request.request_time).astimezone(timezone.utc)
        clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return (
            self.root
            / request.provider.lower()
            / request.dataset.replace("/", "_")
            / f"{stamp.year:04d}"
            / f"{stamp.month:02d}"
            / f"{request.request_hash}_{content_hash[:16]}{clean_suffix}"
        )

    def write(self, request: DataRequest, payload: bytes, *, suffix: str) -> FetchResult:
        digest = hashlib.sha256(payload).hexdigest()
        target = self.path_for(request, digest, suffix)
        if target.exists():
            existing = hashlib.sha256(target.read_bytes()).hexdigest()
            if existing != digest:
                raise RuntimeError(f"immutable raw collision at {target}")
            return FetchResult(request.request_hash, str(target), digest, len(payload), True)

        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".partial-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_name, target)
                os.unlink(temp_name)
            except OSError:
                if target.exists():
                    os.unlink(temp_name)
                else:
                    os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return FetchResult(request.request_hash, str(target), digest, len(payload), False)

