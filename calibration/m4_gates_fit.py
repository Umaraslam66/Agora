"""Pre-M4 sealed-gate calibration fits (both sealed 2026-07-15):

* ``haspass`` — household transponder inheritance
  (`docs/DECISION_M4_HAS_PASS_GATE.md`, Option B): fits the per-household
  draw rates so person-level pass coverage on car trips reproduces the sealed
  segment pass-rate pins (the frozen world ``pass_prior`` applied per
  protected segment), draws the household passes through
  `world.household_pass`, and freezes the result in
  ``runs/m4_prep/has_pass_household/`` (manifest + persona_pass map).

* ``borrowedcar`` — calibrated borrowed-car availability
  (`docs/DECISION_M4_BORROWED_CAR_GATE.md`, Option B): fits the single
  per-day availability rate so the simulated car0 car-driver trip-weight
  share reproduces the observed share (the ~7.3% pin, recomputed exactly
  here), restricted to persons whose OWN seeding record shows car-driver
  trips in a zero-vehicle household, and freezes it in
  ``runs/m4_prep/borrowed_car/`` (manifest + qualifying ids).

Fit discipline (both): calibration-window data only — the pooled 2017+2019
weekday build (every input predates the wall); fitted ONCE, frozen in the
dated manifests below, never refit after any blind quantity is seen (§2 wall;
A3.4). HARNESS-SIDE ONLY: like the rest of ``calibration/``, this module must
never be imported by agent-facing code.

Weighting conventions mirror the sealed E1 comparison exactly: the observed
side is trip_weight-weighted (the truth convention), the simulated side is
day_weight-weighted expected pattern mixes (the method-arm convention) — the
same two conventions E1 compares, so the fit closes the same gap E1 scores.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agents.card_executor import BorrowedCarAccess, expected_mode_counts
from grounding import seeding
from grounding.adapters import psrc
from world.config import get_config
from world.household_pass import (
    SEED_NAMESPACE,
    draw_household_pass,
    persona_pass_from_households,
)

#: Sentinel for income-refusal households (segment None) in the preservation
#: table; they are excluded from segmented stats but hold passes in the pooled
#: population, so the fit covers them under this pooled row.
UNSEGMENTED = "__unsegmented__"

#: Implementation tolerance on the realized (post-draw) per-segment coverage,
#: as a MEASURED z-bound (a gate on the implementation, NOT a sealed bar): the
#: expected coverage matches the pin exactly by construction, and the realized
#: coverage deviates only by the once-per-household CRN draw's binomial noise,
#: whose per-segment sigma is computable exactly from the car-trip-weight
#: concentration (sqrt(sum w_h^2 q(1-q)) / W — expansion-weight skew makes the
#: effective household count far smaller than the raw count). A segment is
#: within tolerance iff |realized - expected| <= Z_TOLERANCE * sigma.
COVERAGE_Z_TOLERANCE = 2.0

#: The sealed fresh executor CRN site (decision record item 2).
CARACCESS_SITE = "caraccess"


# ---------------------------------------------------------------------------
# shared data assembly
# ---------------------------------------------------------------------------

def _index_frame() -> Tuple[psrc.PSRCDataset, pd.DataFrame]:
    dataset = psrc.load_or_build()
    idx = seeding.persona_index(dataset)
    idx["household_id"] = idx["household_id"].astype(str)
    idx["person_id"] = idx["person_id"].astype(str)
    return dataset, idx


def _car_trip_weight_by_person(dataset: psrc.PSRCDataset) -> pd.Series:
    """Observed car-driver trip weight per person_id (trip_weight-weighted)."""
    tr = dataset.weekday_trips
    car = tr[tr["mode"] == "car"]
    return car.groupby(car["person_id"].astype(str))["trip_weight"].sum()


def _trip_weight_by_person(dataset: psrc.PSRCDataset) -> pd.Series:
    tr = dataset.weekday_trips
    return tr.groupby(tr["person_id"].astype(str))["trip_weight"].sum()


# ---------------------------------------------------------------------------
# haspass — household transponder inheritance fit + draw
# ---------------------------------------------------------------------------

def fit_household_pass(
    dataset: psrc.PSRCDataset,
    idx: pd.DataFrame,
    target_rate: Optional[float] = None,
    namespace: str = SEED_NAMESPACE,
) -> dict:
    """Fit the per-household draw rates, draw the passes, and assemble the
    manifest payload. Returns a dict with ``persona_pass``, ``rate_by_household``
    and the manifest body."""
    if target_rate is None:
        target_rate = float(get_config("cityk_corridor").pass_prior)

    # Household eligibility (sealed item 1): >=1 licensed adult AND >=1 vehicle.
    # Licensed-adult status is determined over the diary-carded members (the
    # 1:1 seeded persons — the population the executor runs); recorded below.
    idx = idx.copy()
    idx["is_licensed_adult"] = (
        idx["can_drive"].fillna(False).astype(bool)
        & (pd.to_numeric(idx["age"], errors="coerce").fillna(0) >= 18)
    )
    hh = idx.groupby("household_id").agg(
        any_licensed_adult=("is_licensed_adult", "any"),
        household_cars=("household_cars", "first"),
        segment=("segment", "first"),
        n_members=("persona_id", "size"),
    )
    hh["household_cars"] = pd.to_numeric(hh["household_cars"], errors="coerce")
    hh["eligible"] = hh["any_licensed_adult"] & (hh["household_cars"].fillna(0) >= 1)
    hh["segment"] = hh["segment"].fillna(UNSEGMENTED)

    # Person-level observed car-trip weight, joined to households.
    car_w = _car_trip_weight_by_person(dataset)
    idx["car_weight"] = idx["person_id"].map(car_w).fillna(0.0)
    person_hh = idx[["persona_id", "person_id", "household_id", "car_weight"]].merge(
        hh[["eligible", "segment"]], left_on="household_id", right_index=True, how="left"
    )

    # Per-segment eligible car-trip-weight share e_s, and the re-weighted
    # household draw rate q_s = min(1, target / e_s): person-level pass
    # coverage on car trips then equals the target in expectation.
    seg_total = person_hh.groupby("segment")["car_weight"].sum()
    seg_eligible = (
        person_hh[person_hh["eligible"].fillna(False)]
        .groupby("segment")["car_weight"].sum()
        .reindex(seg_total.index)
        .fillna(0.0)
    )
    e_s = (seg_eligible / seg_total.replace(0.0, np.nan)).fillna(0.0)
    q_s: Dict[str, float] = {}
    for seg in seg_total.index:
        share = float(e_s[seg])
        q_s[str(seg)] = min(1.0, target_rate / share) if share > 0.0 else 0.0

    rate_by_household = {
        str(h): q_s[str(row.segment)]
        for h, row in hh.iterrows()
        if bool(row.eligible)
    }
    hh_pass = draw_household_pass(rate_by_household, namespace=namespace)
    persona_household = dict(zip(idx["persona_id"].astype(str), idx["household_id"]))
    persona_pass = persona_pass_from_households(persona_household, hh_pass)

    # Preservation table: expected and realized person-level pass coverage on
    # car-trip weight, per segment. A car0 segment is STRUCTURALLY ZERO under
    # the sealed model (a zero-vehicle household is ineligible by gate item 1
    # and a borrowed car is not the household vehicle, item 2), so its pin is
    # 0, not the person-level prior — that re-model IS the decision.
    person_hh["hh_pass"] = person_hh["household_id"].map(hh_pass).fillna(False)
    table = {}
    for seg in sorted(seg_total.index.astype(str)):
        seg_rows = person_hh[person_hh["segment"].astype(str) == seg]
        denom = float(seg_rows["car_weight"].sum())
        covered = float(seg_rows.loc[seg_rows["hh_pass"].astype(bool), "car_weight"].sum())
        q = q_s[seg]
        expected = q * float(e_s[seg])
        realized = covered / denom if denom > 0.0 else 0.0
        structurally_zero = float(e_s[seg]) == 0.0
        # exact draw-noise sigma of the realized coverage (household-level
        # binomial, car-trip-weight weighted)
        hw = seg_rows.groupby("household_id")["car_weight"].sum()
        sigma = (
            float(np.sqrt((hw**2).sum() * q * (1.0 - q)) / denom)
            if denom > 0.0 else 0.0
        )
        dev = abs(realized - expected)
        table[seg] = {
            "target": 0.0 if structurally_zero else target_rate,
            "structurally_zero": structurally_zero,
            "eligible_car_weight_share": float(e_s[seg]),
            "household_draw_rate": q,
            "expected_coverage": expected,
            "realized_coverage": realized,
            "draw_noise_sigma": sigma,
            "within_tolerance": bool(
                dev <= COVERAGE_Z_TOLERANCE * sigma if sigma > 0.0 else dev == 0.0
            ),
            "n_households": int((hh["segment"].astype(str) == seg).sum()),
        }

    n_minors_stripped = int(
        ((pd.to_numeric(idx["age"], errors="coerce").fillna(0) < 18) & idx["has_pass"]).sum()
    )
    n_nondrivers_stripped = int(
        ((~idx["can_drive"].fillna(False).astype(bool)) & idx["has_pass"]).sum()
    )
    manifest = {
        "decision": "docs/DECISION_M4_HAS_PASS_GATE.md (Option B, sealed 2026-07-15)",
        "date": date.today().isoformat(),
        "crn_site_key": "{namespace}:{household_id}:hh_pass",
        "draw_namespace": namespace,
        "target_rate": target_rate,
        "target_source": "frozen world pass_prior (cityk_corridor)",
        "gating_counts": {
            "households": int(len(hh)),
            "eligible_households": int(hh["eligible"].sum()),
            "ineligible_households": int((~hh["eligible"]).sum()),
            "pass_households": int(sum(hh_pass.values())),
            "personas": int(len(idx)),
            "personas_with_pass": int(sum(persona_pass.values())),
            "legacy_minor_passes_removed": n_minors_stripped,
            "legacy_nondriver_passes_removed": n_nondrivers_stripped,
        },
        "adult_licence_scope": (
            "licensed-adult status determined over the diary-carded household "
            "members (the 1:1 seeded persons)"
        ),
        "coverage_z_tolerance": COVERAGE_Z_TOLERANCE,
        "segment_preservation": table,
        "charge_semantics": (
            "pass discounts CAR trips in the household vehicle only; ride/hired "
            "trips never receive the discount (world.bridge.corridor_travelers_of_day)"
        ),
    }
    return {
        "manifest": manifest,
        "persona_pass": persona_pass,
        "rate_by_household": rate_by_household,
    }


# ---------------------------------------------------------------------------
# borrowedcar — availability-rate fit
# ---------------------------------------------------------------------------

def qualifying_persona_ids(idx: pd.DataFrame, dataset: psrc.PSRCDataset) -> FrozenSet[str]:
    """The sealed qualifying class: licensed adults in zero-vehicle households
    whose OWN weekday diary shows >=1 car-driver trip."""
    car_w = _car_trip_weight_by_person(dataset)
    hh_cars = pd.to_numeric(idx["household_cars"], errors="coerce")
    is_adult = pd.to_numeric(idx["age"], errors="coerce").fillna(0) >= 18
    licensed = idx["can_drive"].fillna(False).astype(bool)
    shows_car = idx["person_id"].map(car_w).fillna(0.0) > 0.0
    mask = (hh_cars == 0) & licensed & is_adult & shows_car
    return frozenset(idx.loc[mask, "persona_id"].astype(str))


def observed_car0_share(idx: pd.DataFrame, dataset: psrc.PSRCDataset) -> Tuple[float, float, float]:
    """(share, car_weight, total_weight): observed car-driver share of all
    car0-household trip weight (trip_weight-weighted, the truth convention).
    The sealed record pins this as ~7.3%; the exact value is recomputed here."""
    hh_cars = pd.to_numeric(idx["household_cars"], errors="coerce")
    car0_person_ids = set(idx.loc[hh_cars == 0, "person_id"].astype(str))
    tw = _trip_weight_by_person(dataset)
    cw = _car_trip_weight_by_person(dataset)
    total = float(sum(tw.get(p, 0.0) for p in car0_person_ids))
    car = float(sum(cw.get(p, 0.0) for p in car0_person_ids))
    return (car / total if total > 0.0 else 0.0), car, total


def fit_borrowed_car_rate(
    cards: Sequence[dict],
    idx: pd.DataFrame,
    dataset: psrc.PSRCDataset,
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
) -> dict:
    """Fit the per-day availability rate on a card set.

    share_sim(r) = r * N1 / D, where N1 = day-weighted expected car-trip weight
    of the qualifying personas' cards when access is ALWAYS granted, and D =
    day-weighted expected total trip weight of all car0 personas' cards (total
    trips are coercion-invariant, so D does not depend on r). The fitted rate
    is r = clamp(share_obs * D / N1, 0, 1). N1 = 0 (a card set whose car0
    cards carry no car trips — true of the pre-decision r2b build, whose
    generation-time feasibility gate stripped them) is reported as DEGENERATE:
    no rate is frozen from such a set."""
    share_obs, car_w_obs, total_w_obs = observed_car0_share(idx, dataset)
    qualifying = qualifying_persona_ids(idx, dataset)

    hh_cars_by_persona = dict(
        zip(idx["persona_id"].astype(str), pd.to_numeric(idx["household_cars"], errors="coerce"))
    )
    n1 = 0.0
    d_total = 0.0
    for card in cards:
        pid = str(card.get("persona_id"))
        if hh_cars_by_persona.get(pid) != 0:
            continue
        slot_weight = float(sum(w for _dn, w in day_slots.get(pid, [])))
        if slot_weight == 0.0:
            continue
        # Total trips are mode-invariant: use the no-access expectation for D.
        counts_off = expected_mode_counts(card, car_ok=False)
        d_total += slot_weight * float(sum(counts_off.values()))
        if pid in qualifying:
            counts_on = expected_mode_counts(card, car_ok=True)
            n1 += slot_weight * float(counts_on.get("car", 0.0))

    degenerate = n1 <= 0.0
    rate = 0.0 if degenerate else min(1.0, share_obs * d_total / n1)
    return {
        "share_obs": share_obs,
        "car_weight_obs": car_w_obs,
        "total_weight_obs": total_w_obs,
        "qualifying": qualifying,
        "n1_always_allow_car_weight": n1,
        "d_total_trip_weight": d_total,
        "rate": rate,
        "degenerate": degenerate,
    }


def borrowed_car_manifest(fit: dict, cards_path: str, check: Optional[dict] = None) -> dict:
    return {
        "decision": "docs/DECISION_M4_BORROWED_CAR_GATE.md (Option B, sealed 2026-07-15)",
        "date": date.today().isoformat(),
        "crn_site_key": "{namespace}:{persona_id}:{day_index}:caraccess",
        "fit_window": "pooled 2017+2019 weekday build (calibration window; pre-wall)",
        "cards": cards_path,
        "observed_car0_car_share": fit["share_obs"],
        "observed_car0_car_weight": fit["car_weight_obs"],
        "observed_car0_total_weight": fit["total_weight_obs"],
        "n_qualifying_personas": len(fit["qualifying"]),
        "sim_always_allow_car_weight": fit["n1_always_allow_car_weight"],
        "sim_total_trip_weight": fit["d_total_trip_weight"],
        "fitted_rate": fit["rate"],
        "degenerate": fit["degenerate"],
        "degenerate_note": (
            "N1=0: this card set's car0 cards carry no car trips (generation-time "
            "feasibility gate predates the sealed relaxation); the executor draw is "
            "inert on it and NO rate is frozen from it. The rate must be fitted on a "
            "card set built AFTER grounding.card_validation.feasibility learned the "
            "qualifying-class relaxation (the A4.1 tier rebuild)."
            if fit["degenerate"] else None
        ),
        "realized_check": check,
    }


def realized_car0_check(
    cards: Sequence[dict],
    idx: pd.DataFrame,
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
    qualifying: FrozenSet[str],
    rate: float,
    namespace: str = "m4gates_check",
) -> dict:
    """CRN-realized car0 car-trip weight share under the fitted draw (the
    manifest's 'realized car0 car-trip weight after the draw' record)."""
    from agents.card_executor import execute_days  # local: keep module import cheap

    hh_cars_by_persona = dict(
        zip(idx["persona_id"].astype(str), pd.to_numeric(idx["household_cars"], errors="coerce"))
    )
    car0_cards = [c for c in cards if hh_cars_by_persona.get(str(c.get("persona_id"))) == 0]
    access = BorrowedCarAccess(rate=rate, qualifying=qualifying)
    slots = {str(c["persona_id"]): day_slots.get(str(c["persona_id"]), []) for c in car0_cards}
    out = execute_days(car0_cards, slots, namespace, update_habits=False, car_access=access)
    car_w = 0.0
    total_w = 0.0
    for pid, days in out.items():
        for day in days:
            for t in day.trips:
                total_w += day.day_weight
                if t.mode == "car":
                    car_w += day.day_weight
    return {
        "namespace": namespace,
        "realized_car_weight": car_w,
        "realized_total_weight": total_w,
        "realized_share": car_w / total_w if total_w > 0.0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write(out_dir: Path, name: str, payload) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    hp = sub.add_parser("haspass", help="household transponder inheritance fit + draw")
    hp.add_argument("--out", default="runs/m4_prep/has_pass_household")

    bc = sub.add_parser("borrowedcar", help="borrowed-car availability rate fit")
    bc.add_argument("--cards", required=True)
    bc.add_argument("--out", default="runs/m4_prep/borrowed_car")

    args = ap.parse_args(argv)
    dataset, idx = _index_frame()

    if args.cmd == "haspass":
        res = fit_household_pass(dataset, idx)
        out = Path(args.out)
        _write(out, "manifest.json", res["manifest"])
        _write(out, "persona_pass.json", res["persona_pass"])
        _write(out, "rate_by_household.json", res["rate_by_household"])
        bad = [s for s, row in res["manifest"]["segment_preservation"].items()
               if not row["within_tolerance"]]
        print(f"haspass: {res['manifest']['gating_counts']}")
        if bad:
            print(f"haspass: segments outside tolerance (draw granularity): {bad}")
        return 0

    if args.cmd == "borrowedcar":
        import evaluation.e1 as e1
        with open(args.cards) as f:
            cards = [json.loads(line) for line in f if line.strip()]
        persona_of_person = e1.persona_of_person_map(seeding.persona_index(dataset))
        day_slots = e1.day_slots_by_persona(dataset, persona_of_person)
        fit = fit_borrowed_car_rate(cards, idx, dataset, day_slots)
        check = None
        if not fit["degenerate"]:
            check = realized_car0_check(
                cards, idx, day_slots, fit["qualifying"], fit["rate"]
            )
        out = Path(args.out)
        _write(out, "manifest.json", borrowed_car_manifest(fit, args.cards, check))
        _write(out, "qualifying_personas.json", sorted(fit["qualifying"]))
        rate_str = "DEGENERATE" if fit["degenerate"] else "%.4f" % fit["rate"]
        print(
            "borrowedcar: observed share %.4f, qualifying %d, rate %s"
            % (fit["share_obs"], len(fit["qualifying"]), rate_str)
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
