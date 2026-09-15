from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AnnualSplit:
    train: np.ndarray
    validate: np.ndarray
    test: np.ndarray
    test_year: int


def annual_walk_forward_splits(dates: Iterable, *, first_test_year: int = 2016,
                               last_test_year: int = 2024,
                               horizon: int = 21) -> list[AnnualSplit]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    splits = []
    for year in range(first_test_year, last_test_year + 1):
        train_candidates = np.flatnonzero(dates.year <= year - 2)
        validate = np.flatnonzero(dates.year == year - 1)
        test = np.flatnonzero(dates.year == year)
        if not len(train_candidates) or not len(validate) or not len(test):
            continue
        cutoff = max(0, len(train_candidates) - int(horizon))
        train = train_candidates[:cutoff]
        if len(train):
            splits.append(AnnualSplit(train, validate, test, year))
    return splits


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    variable = (
        len(actual) > 1
        and np.std(actual) > 1e-15
        and np.std(predicted) > 1e-15
    )
    pearson = np.corrcoef(actual, predicted)[0, 1] if variable else np.nan
    rank = spearmanr(actual, predicted).statistic if variable else np.nan
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "correlation": float(pearson),
        "rank_ic": float(rank),
    }


def _ridge() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def compare_models(frame: pd.DataFrame, *, date_col: str, target_col: str,
                   momentum_columns: list[str], volatility_columns: list[str],
                   spot_columns: list[str], curve_columns: list[str],
                   horizon: int, first_test_year: int = 2016,
                   last_test_year: int = 2024) -> pd.DataFrame:
    """Compare preregistered Models A-E on identical rows and outer folds."""
    columns = list(dict.fromkeys([
        date_col, target_col, *momentum_columns, *volatility_columns,
        *spot_columns, *curve_columns,
    ]))
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    splits = annual_walk_forward_splits(
        data[date_col], first_test_year=first_test_year,
        last_test_year=last_test_year, horizon=horizon,
    )
    rows = []
    for split in splits:
        y_train = data.loc[split.train, target_col].to_numpy(float)
        y_test = data.loc[split.test, target_col].to_numpy(float)
        predictions = {"A_mean": np.repeat(y_train.mean(), len(split.test))}
        for name, feature_columns in {
            "B_momentum": momentum_columns,
            "C_momentum_volatility": [*momentum_columns, *volatility_columns],
            "D_spot": spot_columns,
            "E_spot_curve": [*spot_columns, *curve_columns],
        }.items():
            model = _ridge()
            model.fit(data.loc[split.train, feature_columns], y_train)
            predictions[name] = model.predict(data.loc[split.test, feature_columns])
        for model_name, prediction in predictions.items():
            rows.append({"test_year": split.test_year, "model": model_name,
                         "n_test": len(y_test), **_metrics(y_test, prediction)})
    return pd.DataFrame(rows)
