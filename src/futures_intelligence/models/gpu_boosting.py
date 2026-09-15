from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .baselines import annual_walk_forward_splits


@dataclass(frozen=True)
class XgbResearchConfig:
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 10.0
    min_child_weight: float = 10.0
    random_state: int = 1729

    def model_params(self, *, device: str) -> dict:
        return {
            **asdict(self),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": device,
            "n_jobs": 0,
            "verbosity": 0,
        }


def _xgb_regressor(config: XgbResearchConfig, *, device: str):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("install the gpu extra: pip install -e .[gpu]") from exc
    return XGBRegressor(**config.model_params(device=device))


def _trained_device(model) -> str:
    payload = json.loads(model.get_booster().save_config())
    return str(payload["learner"]["generic_param"]["device"])


def probe_xgboost_device(requested: str = "cuda") -> dict:
    """Fit a tiny model and report the device XGBoost actually retained."""
    if requested not in {"cuda", "cpu", "auto"}:
        raise ValueError("device must be cuda, cpu, or auto")
    candidates = ["cuda", "cpu"] if requested == "auto" else [requested]
    errors: list[dict[str, str]] = []
    rng = np.random.default_rng(1729)
    x = rng.normal(size=(128, 8))
    y = x[:, 0] - 0.5 * x[:, 1]
    for candidate in candidates:
        try:
            model = _xgb_regressor(
                XgbResearchConfig(n_estimators=2, max_depth=2), device=candidate
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x, y)
            actual = _trained_device(model)
            if candidate == "cuda" and not actual.startswith("cuda"):
                raise RuntimeError(f"XGBoost fell back to {actual}")
            return {
                "requested": requested,
                "selected": candidate,
                "actual": actual,
                "warnings": [str(item.message) for item in caught],
            }
        except Exception as exc:
            errors.append({"device": candidate, "error_type": type(exc).__name__})
    raise RuntimeError(f"no requested XGBoost device is available: {errors}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    variable = len(actual) > 1 and np.std(actual) > 1e-15 and np.std(predicted) > 1e-15
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]) if variable else np.nan,
        "rank_ic": float(spearmanr(actual, predicted).statistic) if variable else np.nan,
    }


def walk_forward_xgb(
    frame: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    feature_sets: Mapping[str, list[str]],
    horizon: int,
    device: str,
    config: XgbResearchConfig | None = None,
    first_test_year: int = 2016,
    last_test_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate strictly outer-fold OOS predictions on identical model rows."""
    if not feature_sets:
        raise ValueError("at least one feature set is required")
    config = config or XgbResearchConfig()
    all_features = list(dict.fromkeys(c for values in feature_sets.values() for c in values))
    required = [date_col, target_col, *all_features]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"missing model columns: {sorted(missing)}")
    data = frame[required].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    splits = annual_walk_forward_splits(
        data[date_col], first_test_year=first_test_year,
        last_test_year=last_test_year, horizon=horizon,
    )
    predictions: list[dict] = []
    metrics: list[dict] = []
    for split in splits:
        y_train = data.loc[split.train, target_col].to_numpy(float)
        y_test = data.loc[split.test, target_col].to_numpy(float)
        for model_name, columns in feature_sets.items():
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(data.loc[split.train, columns])
            x_test = imputer.transform(data.loc[split.test, columns])
            model = _xgb_regressor(config, device=device)
            model.fit(x_train, y_train)
            actual_device = _trained_device(model)
            if device == "cuda" and not actual_device.startswith("cuda"):
                raise RuntimeError(f"XGBoost requested CUDA but trained on {actual_device}")
            predicted = model.predict(x_test)
            metrics.append({
                "test_year": split.test_year,
                "model": model_name,
                "horizon": int(horizon),
                "n_test": len(y_test),
                "device": actual_device,
                **_metrics(y_test, predicted),
            })
            for row_index, observed, estimate in zip(split.test, y_test, predicted):
                predictions.append({
                    "date": data.loc[row_index, date_col],
                    "test_year": split.test_year,
                    "model": model_name,
                    "horizon": int(horizon),
                    "actual": float(observed),
                    "prediction": float(estimate),
                    "device": actual_device,
                    "mode": "SHADOW",
                    "promoted": False,
                })
    return pd.DataFrame(predictions), pd.DataFrame(metrics)
