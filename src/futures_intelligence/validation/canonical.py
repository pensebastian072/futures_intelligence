from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _module(validation_root: Path | str):
    root = str(Path(validation_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from copper_brain import validate
    return validate


def canonical_gate(pnls, *, n_trials: int, validation_root: Path | str) -> dict:
    """Reuse the canonical gate; import failure is a hard SHADOW result."""
    try:
        validate = _module(validation_root)
        result = validate.evaluate_gate(np.asarray(pnls, dtype=float), n_trials=n_trials)
    except Exception as exc:
        return {
            "available": False,
            "passes": False,
            "promoted": False,
            "error_type": type(exc).__name__,
            "reasons": ["canonical validation unavailable; remain SHADOW"],
        }
    result = dict(result)
    result["available"] = True
    result["promoted"] = False
    result["mode"] = "SHADOW"
    return result


def canonical_splits(n: int, *, n_splits: int, horizon: int,
                     validation_root: Path | str):
    validate = _module(validation_root)
    return list(validate.purged_kfold(n, n_splits, horizon, embargo_pct=horizon / max(n, 1)))

