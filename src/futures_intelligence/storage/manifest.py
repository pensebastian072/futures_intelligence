from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from futures_intelligence.types import DataRequest, FetchResult


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS request_manifest (
    request_hash VARCHAR PRIMARY KEY,
    provider VARCHAR NOT NULL,
    dataset VARCHAR NOT NULL,
    schema_name VARCHAR NOT NULL,
    symbols_json VARCHAR NOT NULL,
    start_value VARCHAR NOT NULL,
    end_value VARCHAR NOT NULL,
    input_symbology VARCHAR NOT NULL,
    request_json VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    raw_path VARCHAR,
    content_hash VARCHAR,
    bytes_written BIGINT,
    cost_estimate_usd DOUBLE,
    error_type VARCHAR,
    created_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS partition_manifest (
    table_name VARCHAR NOT NULL,
    partition_key VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    row_count BIGINT NOT NULL,
    content_hash VARCHAR NOT NULL,
    feature_version VARCHAR,
    code_version VARCHAR,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (table_name, partition_key, content_hash)
);
CREATE TABLE IF NOT EXISTS experiment_manifest (
    experiment_id VARCHAR PRIMARY KEY,
    manifest_json VARCHAR NOT NULL,
    promoted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL
);
"""


class ManifestStore:
    def __init__(self, data_root: Path | str):
        root = Path(data_root) / "manifests"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "futures_intelligence.duckdb"
        with self.connect() as con:
            con.execute(SCHEMA_SQL)

    def connect(self):
        return duckdb.connect(str(self.path))

    def successful(self, request_hash: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT raw_path, content_hash, bytes_written FROM request_manifest "
                "WHERE request_hash = ? AND status = 'success'", [request_hash]
            ).fetchone()
        if not row:
            return None
        return {"raw_path": row[0], "content_hash": row[1], "bytes_written": row[2]}

    def begin(self, request: DataRequest, *, cost_estimate_usd: float | None = None) -> None:
        now = datetime.now(timezone.utc)
        safe = request.safe_dict()
        values = [
            request.request_hash, request.provider, request.dataset, request.schema,
            json.dumps(request.symbols), request.start, request.end,
            request.input_symbology, json.dumps(safe, sort_keys=True, default=str),
            "pending", cost_estimate_usd, now,
        ]
        with self.connect() as con:
            con.execute(
                """INSERT INTO request_manifest
                (request_hash, provider, dataset, schema_name, symbols_json, start_value,
                 end_value, input_symbology, request_json, status, cost_estimate_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (request_hash) DO UPDATE SET
                    cost_estimate_usd = COALESCE(excluded.cost_estimate_usd,
                                                 request_manifest.cost_estimate_usd)""",
                values,
            )

    def complete(self, result: FetchResult) -> None:
        with self.connect() as con:
            con.execute(
                """UPDATE request_manifest SET status='success', raw_path=?, content_hash=?,
                bytes_written=?, error_type=NULL, completed_at=? WHERE request_hash=?""",
                [result.raw_path, result.content_hash, result.bytes_written,
                 datetime.now(timezone.utc), result.request_hash],
            )

    def fail(self, request_hash: str, exc: BaseException) -> None:
        with self.connect() as con:
            con.execute(
                """UPDATE request_manifest SET status='failed', error_type=?, completed_at=?
                WHERE request_hash=?""",
                [type(exc).__name__, datetime.now(timezone.utc), request_hash],
            )

    def record_experiment(self, experiment_id: str, manifest: dict[str, Any]) -> None:
        safe = dict(manifest)
        safe["promoted"] = False
        with self.connect() as con:
            con.execute(
                """INSERT INTO experiment_manifest VALUES (?, ?, FALSE, ?)
                ON CONFLICT (experiment_id) DO NOTHING""",
                [experiment_id, json.dumps(safe, sort_keys=True, default=str),
                 datetime.now(timezone.utc)],
            )

    def record_partition(self, *, table_name: str, partition_key: str, path: str,
                         row_count: int, content_hash: str,
                         feature_version: str | None = None,
                         code_version: str | None = None) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO partition_manifest
                (table_name, partition_key, path, row_count, content_hash,
                 feature_version, code_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING""",
                [table_name, partition_key, path, int(row_count), content_hash,
                 feature_version, code_version, datetime.now(timezone.utc)],
            )
