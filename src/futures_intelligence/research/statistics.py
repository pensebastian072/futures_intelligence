from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def benjamini_hochberg(p_values, alpha: float = 0.05) -> pd.DataFrame:
    values = np.asarray(p_values, dtype=float)
    valid = np.isfinite(values)
    rejected = np.zeros(len(values), dtype=bool)
    adjusted = np.full(len(values), np.nan)
    if valid.any():
        reject_valid, adjusted_valid, _, _ = multipletests(
            values[valid], alpha=alpha, method="fdr_bh"
        )
        rejected[valid] = reject_valid
        adjusted[valid] = adjusted_valid
    return pd.DataFrame({"p_value": values, "p_adjusted": adjusted, "rejected": rejected})


def analyze_feature(frame: pd.DataFrame, *, feature: str, target: str,
                    training_end=None, bins: int = 10) -> dict[str, Any]:
    data = frame[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < max(20, bins * 2):
        return {"available": False, "reason": "insufficient_rows", "n": len(data)}
    pearson = stats.pearsonr(data[feature], data[target])
    spearman = stats.spearmanr(data[feature], data[target])
    if training_end is not None and isinstance(frame.index, pd.DatetimeIndex):
        training = frame.loc[:training_end, feature].dropna()
    else:
        training = data[feature]
    edges = np.unique(training.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        deciles = []
    else:
        bucket = pd.cut(data[feature], edges, labels=False, include_lowest=True)
        grouped = data.assign(bucket=bucket).dropna(subset=["bucket"]).groupby("bucket")[target]
        deciles = grouped.agg(["count", "mean", "median", "std"]).reset_index().to_dict("records")
    return {
        "available": True,
        "n": len(data),
        "pearson": {"correlation": float(pearson.statistic), "p_value": float(pearson.pvalue)},
        "spearman": {"correlation": float(spearman.statistic), "p_value": float(spearman.pvalue)},
        "bin_edges": edges.tolist(),
        "deciles": deciles,
    }

