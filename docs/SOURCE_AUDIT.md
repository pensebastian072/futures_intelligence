# Source audit

Observed 2026-08-18. Re-run `audit-sources --live-lse` to refresh current status.

## Contract curves

### London Strategic Edge

The authenticated futures-only catalog returns `ES.F` but no symbol matching a dated
quarterly ES contract (`ES[HMUZ]<year>`). `ES.F` begins in May 2016 and has candles and
volume, but no expiry or per-contract open interest. It is diagnostic only and is never
used to reconstruct a curve.

Verdict: **not sufficient for historical ES curves**.

### Databento

The implementation targets `GLBX.MDP3`, `ES.FUT`, parent symbology, and the `definition`
and `statistics` schemas. Parent data must be filtered to outright futures because it
also includes calendar spreads. Metadata/cost estimation is implemented, but this box
has no Databento credential, so entitlement, cost, and the 2018-06-01 acceptance sample
remain unverified.

Verdict: **documented as sufficient; locally blocked on credential and cost approval**.

### CME DataMine

CME states that daily settlement reports cover every eligible contract and documents
settlement, volume, prior-day settlement/volume, and open interest fields. Its List API
requires an entitled API ID. Entitlement discovery and fixed-width settlement parsing
are implemented, but this box has no CME DataMine credential.

CME reports final equity-index settlements at 18:00 CT. That is later than the research
cutoff of 18:00 ET, so final files are suitable for a next-day immutable collection but
must not be retrospectively treated as available at the 18:00 ET decision timestamp.

Verdict: **authoritative forward collector/fallback, locally blocked on credentials**.

### Barchart and TradingView

Both can display expired contracts, and TradingView can manually export loaded chart
data. Neither is scraped or automated here. They are manual validation sources unless
their licensing and machine-readable access are explicitly approved.

## Context data

- CFTC public TFF futures-only API is implemented and live-verified against exactly
  `E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE`; Micro ES is excluded.
- Historical CFTC observations receive a conservative seven-calendar-day availability
  lag to avoid treating Tuesday positions as known before publication.
- Yahoo is the isolated long-history fallback for unadjusted `^GSPC`, SPY, and `^VIX`.
- LSE provides supplementary rates and COT data. LSE rate observations are lagged one
  New York business day because their rows do not expose publication timestamps.

## Authoritative references

- https://www.cmegroup.com/trading/about-settlements.html
- https://www.cmegroup.com/market-data/files/settlement-file-layout.pdf
- https://www.cmegroup.com/datamine/datamine-list-api.html
- https://databento.com/docs/examples/symbology/parent-symbology
- https://databento.com/docs/schemas-and-data-formats/cmbp-1
- https://publicreporting.cftc.gov/Commitments-of-Traders/TFF-Futures-Only/gpe5-46if

