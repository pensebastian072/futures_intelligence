from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RlEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    mode: str = "SHADOW"
    promoted: bool = False

    def as_dict(self) -> dict:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "mode": self.mode,
            "promoted": self.promoted,
        }


def check_rl_eligibility(
    *,
    has_dated_contract_curve: bool,
    supervised_all_horizons_pass: bool,
    oos_transitions: int,
    transaction_costs_enabled: bool,
    holdout_touched: bool,
    minimum_oos_transitions: int = 1_500,
) -> RlEligibility:
    """Fail-closed gate before an RL package or policy is introduced."""
    reasons: list[str] = []
    if not has_dated_contract_curve:
        reasons.append("point-in-time dated-contract curve is unavailable")
    if not supervised_all_horizons_pass:
        reasons.append("supervised D-versus-E gate has not passed every primary horizon")
    if oos_transitions < minimum_oos_transitions:
        reasons.append(
            f"only {oos_transitions} OOS transitions; require {minimum_oos_transitions}"
        )
    if not transaction_costs_enabled:
        reasons.append("transaction costs are not enabled")
    if holdout_touched:
        reasons.append("locked holdout was already touched")
    return RlEligibility(eligible=not reasons, reasons=tuple(reasons))
