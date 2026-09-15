from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from futures_intelligence.storage.manifest import ManifestStore


class ParquetStore:
    """Versioned normalized partitions; existing files are never overwritten."""

    def __init__(self, data_root: Path | str):
        self.root = Path(data_root) / "normalized"
        self.manifest = ManifestStore(data_root)

    def write(self, frame: pd.DataFrame, *, table_name: str, partition_key: str,
              feature_version: str | None = None,
              code_version: str | None = None) -> Path:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="zstd")
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / table_name / partition_key / f"{digest}.parquet"
        if not target.exists():
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
        self.manifest.record_partition(
            table_name=table_name, partition_key=partition_key, path=str(target),
            row_count=len(frame), content_hash=digest,
            feature_version=feature_version, code_version=code_version,
        )
        return target
