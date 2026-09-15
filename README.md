# Futures Curve Intelligence Engine

Research-only, point-in-time ES futures-curve engine. The repository is permanently
SHADOW by default and contains no broker, account, position, order, or execution code.

## Current implementation

- Provider-neutral request contracts with secret-free deterministic hashes.
- Immutable content-addressed raw storage and DuckDB request/experiment manifests.
- Double-gated Databento sample downloads; metadata and cost estimation are separate.
- LSE supplementary-data adapter using the existing gitignored credential without
  copying it into this repository.
- CME DataMine entitlement discovery and a parser for the documented 11-column daily
  futures settlement layout. DataMine downloads are separately gated.
- Yahoo spot/VIX snapshot adapter with immutable raw retention.
- Public CFTC TFF futures-only ingestion with a conservative seven-day availability lag.
- Databento definition/statistics normalization, outright filtering, and revision fields.
- 18:00 ET as-of curve reconstruction, contract settlement/volume/OI joins, log-price
  constant-maturity interpolation, and strict no-extrapolation behavior.
- Basis, annualized basis, carry residual, slope, curvature, dynamics, targets,
  annual walk-forward baselines, research backtest, and canonical `copper_brain` gate.
- Fixed-parameter XGBoost regression with verified CUDA/CPU device reporting, identical
  Model D/E rows, train-only imputation, and annual out-of-sample predictions.
- Non-overlapping SPY next-session-open SHADOW backtests with two-sided costs and a
  fail-closed RL eligibility gate.

No historical provider data has been purchased or bulk-downloaded by setup or tests.

## Setup on this Windows host

```powershell
cd C:\Users\<your-user>\futures_intelligence
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
.venv\Scripts\python.exe -m futures_intelligence.cli audit-sources
```

The bootstrap uses uv system certificates and the dedicated Python 3.11 environment. Never use
`--trusted-host` and never use the Microsoft Store `python` stub.

## Credentials

Credentials are read from environment variables. LSE can also read the existing
`company_lab\secrets\lse.json` file referenced by `config/base.toml`. Do not copy that
file into this repository.

```text
DATABENTO_API_KEY
LSE_API_KEY
CME_DATAMINE_API_ID
CME_DATAMINE_API_PASSWORD
```

The LSE credential previously pasted into chat should be rotated. The software never
prints a key or includes one in a request hash or manifest.

## Phase 0 commands

Safe environment audit, with no live network calls:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli audit-sources
```

Small LSE metadata/catalog audit, cached immutably:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli audit-sources --live-lse
```

Persist a secret-free audit artifact:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli audit-sources --live-lse `
  --output D:\futures_intelligence_data\reports\phase0_environment_audit.json
```

Databento cost estimate only (requires a key, downloads no time-series bytes):

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli databento-cost
```

The one-day Databento sample requires both explicit controls:

```powershell
$env:FI_ALLOW_PAID_DOWNLOAD='YES'
.venv\Scripts\python.exe -m futures_intelligence.cli databento-sample --allow-paid
Remove-Item Env:FI_ALLOW_PAID_DOWNLOAD
```

Do not set this opt-in until the displayed definition + statistics cost has been
reviewed and approved.

After the raw sample is cached, verify and normalize it without another provider call:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli databento-verify --spot <SPX_CLOSE>
```

The verifier fails unless it reconstructs at least two dated ES outrights and one
bracketed constant maturity using no post-cutoff settlement revision.

CME entitlement discovery is read-only but requires DataMine credentials:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli cme-list --date 2026-08-18
```

The CME website reports every eligible contract in its daily settlement reports, but
final equity-index values arrive at 18:00 CT and displayed open interest is normally
from the prior trading day. This project therefore stores both the referenced session
and actual availability timestamp. Website pages, Barchart, TradingView, and
Investing.com are validation surfaces only; the project does not scrape them.

## GPU ML and backtesting

Inspect the installed NVIDIA/XGBoost stack and verify the device XGBoost actually uses:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli gpu-audit --probe --device cuda
```

Run a synthetic end-to-end smoke test. Its output is explicitly marked synthetic and
must not be interpreted as research evidence:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli ml-smoke --device cuda
```

Run the real pre-holdout spot/context baseline using immutable Yahoo snapshots. The
forecast target is SPX close-to-close return; the backtest uses SPY next-session-open
to future-open returns and charges costs on entry and exit:

```powershell
.venv\Scripts\python.exe -m futures_intelligence.cli spot-baseline `
  --device cuda --cost-bps 5 --n-trials 17
```

Increase `--n-trials` for every formula, model, horizon, parameter configuration,
retry, or resumed research run already attempted. The command stops at 2024-12-31;
it never reads the locked 2025 holdout.

This baseline is Model D only. It does not make a futures-curve claim. Model E uses
the same walk-forward implementation after point-in-time dated ES contracts are
available. RL remains disabled until the real D-versus-E comparison clears every
primary horizon, has enough out-of-sample transitions, includes costs, and leaves the
holdout untouched.

## Data layout

```text
D:\futures_intelligence_data\
  raw\          # provider-native immutable bytes
  normalized\   # versioned Parquet partitions
  manifests\    # DuckDB request/experiment catalog
  reports\      # reproducible research outputs
```

Successful metered requests are replayed by request hash. Transformations never delete
or overwrite their raw source.

## Promotion policy

All experiment manifests force `promoted=false`. The canonical PBO/Deflated-Sharpe
implementation is imported from `C:\Users\<your-user>\copper_brain`; an unavailable gate is
a failure, never a pass. Implemented backtests are offline research utilities and have
no execution integration.
