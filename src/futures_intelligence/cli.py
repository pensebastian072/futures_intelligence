from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from futures_intelligence.backtest import non_overlapping_forecast_backtest
from futures_intelligence.config import load_settings
from futures_intelligence.contracts import normalize_definitions, normalize_statistics
from futures_intelligence.curves import build_contract_curve, standardize_curve
from futures_intelligence.features import (
    SPOT_CONTEXT_COLUMNS,
    build_forward_open_returns,
    build_spot_context_features,
)
from futures_intelligence.labels import build_forward_targets
from futures_intelligence.models import XgbResearchConfig, probe_xgboost_device, walk_forward_xgb
from futures_intelligence.providers.cme import CmeDataMineProvider
from futures_intelligence.providers.databento import DatabentoProvider
from futures_intelligence.providers.lse import LseProvider
from futures_intelligence.providers.yahoo import YahooProvider
from futures_intelligence.reports import environment_audit
from futures_intelligence.rl import check_rl_eligibility
from futures_intelligence.storage import ManifestStore, ParquetStore
from futures_intelligence.types import DataRequest
from futures_intelligence.validation import canonical_gate


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _sample_request(settings, schema: str) -> DataRequest:
    cfg = settings.raw["databento"]
    return DataRequest(
        provider="databento", dataset=cfg["dataset"],
        symbols=(cfg["parent_symbol"],), schema=schema,
        start=cfg["sample_start"], end=cfg["sample_end"],
        input_symbology=cfg["input_symbology"],
    )


def command_audit(args) -> int:
    settings = load_settings(args.config)
    result = environment_audit(settings)
    if args.live_lse:
        try:
            result["lse_live"] = LseProvider(settings).audit()
        except Exception as exc:
            result["lse_live"] = {
                "available": False, "error_type": type(exc).__name__
            }
    if args.output:
        _write_json(args.output, result)
    _print(result)
    return 0


def command_databento_cost(args) -> int:
    settings = load_settings(args.config)
    provider = DatabentoProvider(settings.data_root)
    schemas = provider.list_schemas(settings.raw["databento"]["dataset"])
    result = {"available_schemas": schemas, "requests": []}
    for schema in ("definition", "statistics"):
        request = _sample_request(settings, schema)
        result["requests"].append({
            "schema": schema,
            "request_hash": request.request_hash,
            "cost_usd": provider.estimate_cost(request),
        })
    result["total_cost_usd"] = sum(v["cost_usd"] for v in result["requests"])
    result["downloaded"] = False
    _print(result)
    return 0


def command_databento_sample(args) -> int:
    settings = load_settings(args.config)
    provider = DatabentoProvider(settings.data_root)
    results = []
    for schema in ("definition", "statistics"):
        request = _sample_request(settings, schema)
        fetched = provider.fetch_raw(request, allow_paid=args.allow_paid)
        results.append({
            "schema": schema, "request_hash": request.request_hash,
            "raw_path": fetched.raw_path, "content_hash": fetched.content_hash,
            "bytes": fetched.bytes_written, "reused": fetched.reused,
        })
    _print({"mode": "SHADOW", "promoted": False, "results": results})
    return 0


def command_databento_verify(args) -> int:
    settings = load_settings(args.config)
    provider = DatabentoProvider(settings.data_root)
    cached = {}
    for schema in ("definition", "statistics"):
        request = _sample_request(settings, schema)
        prior = provider.manifest.successful(request.request_hash)
        if not prior or not Path(prior["raw_path"]).exists():
            raise RuntimeError(
                f"cached {schema} sample is missing; run cost review and the gated sample first"
            )
        cached[schema] = provider.dbn_to_frame(prior["raw_path"])
    definitions = normalize_definitions(cached["definition"])
    statistics = normalize_statistics(cached["statistics"])
    local_cutoff = pd.Timestamp("2018-06-01 18:00:00", tz=settings.tz)
    contract_curve = build_contract_curve(
        definitions, statistics, asof_timestamp=local_cutoff.tz_convert("UTC"),
        session_date="2018-06-01", spot=float(args.spot),
        rate_by_dte=0.0, dividend_yield=0.0, day_count=settings.day_count,
    )
    standardized = standardize_curve(contract_curve, settings.target_dtes)
    store = ParquetStore(settings.data_root)
    paths = {
        "instrument_definitions": str(store.write(
            definitions, table_name="instrument_definitions", partition_key="year=2018"
        )),
        "contract_statistics_raw": str(store.write(
            statistics, table_name="contract_statistics_raw", partition_key="year=2018"
        )),
        "curve_contract_daily": str(store.write(
            contract_curve, table_name="curve_contract_daily", partition_key="year=2018"
        )),
        "constant_maturity_curve": str(store.write(
            standardized, table_name="constant_maturity_curve", partition_key="year=2018",
            feature_version="es_curve_v1",
        )),
    }
    usable = len(contract_curve)
    bracketing = len(standardized)
    passed = usable >= 2 and bracketing >= 1
    _print({
        "acceptance_passed": passed,
        "asof_timestamp": local_cutoff.tz_convert("UTC"),
        "outright_contracts": usable,
        "standardized_maturities": standardized.get("target_dte", pd.Series(dtype=int)).tolist(),
        "post_cutoff_rows_used": bool(
            not contract_curve.empty
            and contract_curve["settlement_available_at"].gt(local_cutoff.tz_convert("UTC")).any()
        ),
        "paths": paths,
        "mode": "SHADOW",
        "promoted": False,
    })
    return 0 if passed else 2


def command_cme_list(args) -> int:
    settings = load_settings(args.config)
    result = CmeDataMineProvider(settings).list_entitled_files(
        period_date=args.date.replace("-", ""), product_code="ES"
    )
    safe = {
        "date": args.date,
        "entries": len(result.get("data", [])),
        "data": result.get("data", []),
    }
    _print(safe)
    return 0


def command_gpu_audit(args) -> int:
    try:
        version = importlib.metadata.version("xgboost")
    except importlib.metadata.PackageNotFoundError:
        version = "not_installed"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        fields = [value.strip() for value in completed.stdout.strip().split(",")]
        gpu = dict(zip(
            ["name", "driver_version", "memory_total_mib", "memory_free_mib", "compute_capability"],
            fields,
        ))
    except (OSError, subprocess.SubprocessError):
        gpu = {"available": False}
    result = {"gpu": gpu, "xgboost": version, "mode": "SHADOW", "promoted": False}
    if version != "not_installed" and args.probe:
        try:
            result["probe"] = probe_xgboost_device(args.device)
        except Exception as exc:
            result["probe"] = {"available": False, "error_type": type(exc).__name__}
    _print(result)
    return 0


def _synthetic_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2010-06-07", "2018-12-31")
    rng = np.random.default_rng(1729)
    level = rng.normal(size=len(dates))
    curve = rng.normal(size=len(dates))
    target = 0.02 * level + 0.03 * curve + rng.normal(scale=0.2, size=len(dates))
    return pd.DataFrame({
        "date": dates, "spot_level": level, "spot_vol": np.abs(level),
        "curve_slope": curve, "target": target,
    })


def command_ml_smoke(args) -> int:
    probe = probe_xgboost_device(args.device)
    predictions, metrics = walk_forward_xgb(
        _synthetic_frame(), date_col="date", target_col="target",
        feature_sets={
            "D_spot": ["spot_level", "spot_vol"],
            "E_spot_curve": ["spot_level", "spot_vol", "curve_slope"],
        },
        horizon=5, device=probe["selected"],
        config=XgbResearchConfig(n_estimators=20, max_depth=2),
        first_test_year=2016, last_test_year=2018,
    )
    _print({
        "synthetic": True,
        "research_claim": False,
        "probe": probe,
        "prediction_rows": len(predictions),
        "fold_metrics": metrics.to_dict(orient="records"),
        "mode": "SHADOW",
        "promoted": False,
    })
    return 0


def _yahoo_request(symbol: str) -> DataRequest:
    return DataRequest(
        provider="yahoo", dataset="daily", symbols=(symbol,), schema="ohlcv",
        start="2010-06-07", end="2025-01-01", input_symbology="ticker",
    )


def command_spot_baseline(args) -> int:
    if int(args.n_trials) < 17:
        raise ValueError(
            "n_trials must include 11 formulas, one feature family, three horizons, "
            "one model, and one hyperparameter configuration (minimum 17)"
        )
    settings = load_settings(args.config)
    probe = probe_xgboost_device(args.device)
    provider = YahooProvider(settings.data_root)
    spx = provider.fetch_bars(_yahoo_request("^GSPC"))
    vix = provider.fetch_bars(_yahoo_request("^VIX"))
    spy = provider.fetch_bars(_yahoo_request("SPY"))
    features = build_spot_context_features(spx, vix)
    targets = build_forward_targets(
        spx[["date", "close"]], return_horizons=(5, 21, 63), vol_horizons=()
    )
    execution = build_forward_open_returns(spy, horizons=(5, 21, 63))
    frame = features.merge(targets, on="date", validate="one_to_one").merge(
        execution, on="date", validate="one_to_one"
    )
    all_predictions = []
    all_metrics = []
    all_episodes = []
    gates = {}
    backtests = {}
    config = XgbResearchConfig()
    for horizon in (5, 21, 63):
        predictions, metrics = walk_forward_xgb(
            frame, date_col="date", target_col=f"return_{horizon}d",
            feature_sets={"D_spot_context": SPOT_CONTEXT_COLUMNS},
            horizon=horizon, device=probe["selected"], config=config,
            first_test_year=2016, last_test_year=2024,
        )
        predictions = predictions.merge(
            frame[["date", f"spy_open_return_{horizon}d"]], on="date", how="left",
            validate="many_to_one",
        )
        episodes, backtest = non_overlapping_forecast_backtest(
            predictions, horizon=horizon, return_col=f"spy_open_return_{horizon}d",
            cost_fraction=float(args.cost_bps) / 10_000.0,
        )
        gate = canonical_gate(
            episodes["strategy_log_return"].to_numpy(float),
            n_trials=int(args.n_trials), validation_root=settings.validation_root,
        )
        gates[str(horizon)] = gate
        backtests[str(horizon)] = backtest
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        episodes["horizon"] = horizon
        all_episodes.append(episodes)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    metric_frame = pd.concat(all_metrics, ignore_index=True)
    episode_frame = pd.concat(all_episodes, ignore_index=True)
    store = ParquetStore(settings.data_root)
    paths = {
        "predictions": str(store.write(
            prediction_frame, table_name="oos_predictions",
            partition_key="experiment=spot_context_xgb_v1", feature_version="spot_context_v1",
        )),
        "metrics": str(store.write(
            metric_frame, table_name="experiment_metrics",
            partition_key="experiment=spot_context_xgb_v1", feature_version="spot_context_v1",
        )),
        "backtest_episodes": str(store.write(
            episode_frame, table_name="backtest_episodes",
            partition_key="experiment=spot_context_xgb_v1", feature_version="spot_context_v1",
        )),
    }
    summary_metrics = metric_frame.groupby(["model", "horizon"])[
        ["rmse", "correlation", "rank_ic"]
    ].agg(["mean", "median"]).reset_index()
    summary_metrics.columns = [
        "model", "horizon", "rmse_mean", "rmse_median",
        "correlation_mean", "correlation_median", "rank_ic_mean", "rank_ic_median",
    ]
    manifest = {
        "experiment_id": f"spot_context_xgb_v1_{probe['selected']}_t{int(args.n_trials)}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": False,
        "scope": "spot/context baseline only; no futures curve features",
        "data_end_exclusive": "2025-01-01",
        "locked_holdout_touched": False,
        "device": probe,
        "parameters": config.model_params(device=probe["selected"]),
        "horizons": [5, 21, 63],
        "n_trials": int(args.n_trials),
        "trial_count_basis": (
            "11 formulas + 1 feature family + 3 horizons + 1 model + "
            "1 hyperparameter configuration + retries"
        ),
        "gates": gates,
        "all_horizons_pass": all(bool(v.get("passes", False)) for v in gates.values()),
        "backtests": backtests,
        "fold_summary": summary_metrics.to_dict(orient="records"),
        "paths": paths,
        "mode": "SHADOW",
        "promoted": False,
    }
    manifest["rl_eligibility"] = check_rl_eligibility(
        has_dated_contract_curve=False,
        supervised_all_horizons_pass=False,
        oos_transitions=len(episode_frame),
        transaction_costs_enabled=True,
        holdout_touched=False,
    ).as_dict()
    ManifestStore(settings.data_root).record_experiment(manifest["experiment_id"], manifest)
    output = args.output or settings.data_root / "reports" / f"{manifest['experiment_id']}.json"
    _write_json(output, manifest)
    _print(manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="futures-intelligence",
        description="Research-only futures curve engine (always SHADOW).",
    )
    parser.add_argument("--config", type=Path, default=Path("config/base.toml"))
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-sources")
    audit.add_argument("--live-lse", action="store_true")
    audit.add_argument("--output", type=Path)
    audit.set_defaults(func=command_audit)

    cost = sub.add_parser("databento-cost")
    cost.set_defaults(func=command_databento_cost)

    sample = sub.add_parser("databento-sample")
    sample.add_argument("--allow-paid", action="store_true")
    sample.set_defaults(func=command_databento_sample)

    verify = sub.add_parser("databento-verify")
    verify.add_argument("--spot", type=float, required=True,
                        help="Unadjusted SPX close for 2018-06-01")
    verify.set_defaults(func=command_databento_verify)

    cme = sub.add_parser("cme-list")
    cme.add_argument("--date", required=True, help="YYYY-MM-DD")
    cme.set_defaults(func=command_cme_list)

    gpu = sub.add_parser("gpu-audit")
    gpu.add_argument("--probe", action="store_true")
    gpu.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    gpu.set_defaults(func=command_gpu_audit)

    smoke = sub.add_parser("ml-smoke")
    smoke.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    smoke.set_defaults(func=command_ml_smoke)

    baseline = sub.add_parser("spot-baseline")
    baseline.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    baseline.add_argument("--cost-bps", type=float, default=5.0)
    baseline.add_argument("--n-trials", type=int, default=17)
    baseline.add_argument("--output", type=Path)
    baseline.set_defaults(func=command_spot_baseline)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
