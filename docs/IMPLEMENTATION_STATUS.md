# Implementation status

## Complete

- Isolated Python 3.11 repository and D: data root.
- Secret-free configuration and SHADOW enforcement.
- Immutable raw store, normalized Parquet partitions, DuckDB request/partition/experiment manifests.
- LSE, Databento, CME DataMine, Yahoo, and CFTC provider adapters.
- Double opt-in for any Databento or CME file download.
- Databento definition/statistics normalization and outright filtering.
- Revision-aware 18:00 ET contract curves and no-extrapolation constant maturities.
- Carry, curve-shape, target, statistical, baseline A-E, canonical gate, and offline backtest modules.
- Bounded live LSE audit and public CFTC sample verification.
- Automated tests and Windows-native-TLS bootstrap.
- GPU XGBoost regression with verified device retention, annual OOS predictions, and
  identical-row Model D/E support.
- Delayed, non-overlapping costed forecast backtest and fail-closed RL eligibility gate.
- Real 2010-2024 Yahoo Model D spot/context baseline executed on CUDA; all canonical
  primary-horizon gates failed and `promoted=false` was preserved.

## Blocked external acceptance checks

- Databento metadata/cost/sample: `DATABENTO_API_KEY` missing.
- CME DataMine entitlement list: API ID/password missing.
- The LSE key was exposed in chat and should be rotated outside this repository.
- Model E and RL research remain blocked on the point-in-time dated ES history plus a
  successful supervised D-versus-E gate.

No paid data was downloaded and no trade/account action was attempted.
