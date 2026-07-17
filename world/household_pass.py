"""Household transponder inheritance — the sealed pre-M4 pass re-model
(`docs/DECISION_M4_HAS_PASS_GATE.md`, Option B, sealed 2026-07-15).

The pass is a HOUSEHOLD attribute: drawn once per household through the CRN
layer under the sealed fresh site key ``"{namespace}:{household_id}:hh_pass"``,
gated on the household having >=1 licensed adult AND >=1 vehicle. Every member
inherits the household pass — the CHARGE-side restriction (pass discount for
car trips in the household vehicle only, never for ride/hired trips) is
enforced where the charge is computed (`world.bridge.corridor_travelers_of_day`),
not here.

This module is mechanism only: it holds the draw and the inheritance join.
Eligibility gating and the segment pass-rate re-weighting are CALIBRATION
inputs (fitted once, frozen in `runs/m4_prep/has_pass_household/manifest.json`
by `calibration.m4_gates_fit`) and arrive here as the per-household draw
rates — a household absent from the mapping (ineligible, or rate 0) holds no
pass. Because the site key is fresh, every pre-existing CRN stream
(``:vot``, ``:{day}:route``, ``:{day}:pattern``, ``:has_pass``) is
bit-identical to before this decision, and twin-world pairing is preserved by
construction.

The draw namespace is the SEEDING namespace (default ``"seed"``), matching the
per-persona skeleton draw this re-model supersedes: pass holding is a stable
population attribute, not per-run ensemble noise (ensemble variation lives in
the pattern/route/vot draws).
"""
from __future__ import annotations

from typing import Dict, Mapping

from world import crn

#: The sealed fresh CRN site (decision record item 3).
HH_PASS_SITE = "hh_pass"

#: Seeding namespace for the stable household draw (mirrors the skeleton
#: ``has_pass`` draw's namespace; see module docstring).
SEED_NAMESPACE = "seed"


def hh_pass_key(namespace: str, household_id: str) -> str:
    """The sealed CRN site key ``"{namespace}:{household_id}:hh_pass"``."""
    return f"{namespace}:{household_id}:{HH_PASS_SITE}"


def draw_household_pass(
    rate_by_household: Mapping[str, float],
    namespace: str = SEED_NAMESPACE,
) -> Dict[str, bool]:
    """One CRN pass draw per household.

    ``rate_by_household`` carries the calibrated (segment re-weighted) draw
    rate for every ELIGIBLE household; ineligible households are simply absent
    (or carry rate 0.0) and hold no pass. Deterministic in
    (rate_by_household, namespace).
    """
    return {
        str(hh): bool(crn.draw(hh_pass_key(namespace, str(hh))) < float(rate))
        for hh, rate in rate_by_household.items()
    }


def persona_pass_from_households(
    persona_household: Mapping[str, str],
    hh_pass: Mapping[str, bool],
) -> Dict[str, bool]:
    """Inheritance join: every member inherits the household pass. A persona
    whose household is unknown (absent from ``hh_pass``) holds no pass."""
    return {
        str(pid): bool(hh_pass.get(str(hh), False))
        for pid, hh in persona_household.items()
    }
