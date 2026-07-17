"""Sealed no-LLM comparator arm for BT1 (drafted for §7 Amendment A5).

A plain calibrated statistical model producing a tunnel-volume prediction
under the toll: DIARY REPLAY + the frozen corridor route-choice logit, and
nothing else. Same PSRC seeds (each persona's OBSERVED weekday diary replayed
verbatim — no cards, no LLM, no habit machinery), same world equilibrium
(logit theta frozen, VoT lognormal x income ladder), same sealed pre-M4 gates
(household pass inheritance; a zero-vehicle person's observed car trips ARE
their availability — replay needs no draw), and the SAME SR 520 calibration
anchor: the comparator's own VoT scale is bisected so its corridor drop under
the rehearsal schedule reproduces the pinned plateau midpoint. It is the
"calibrated dial without the agents": if it predicts BT1 as well as the
two-brain method, individual agent detail carried no signal — which is
exactly the comparison the A5 draft seals.

Being deterministic-behavior + CRN route draws, it has NO machinery drift, so
its prediction needs no placebo differencing: the raw drop ensemble is the
prediction (recorded as such; the E4 scorer's coverage/closeness kernels
apply unchanged at firing time).

HARNESS-SIDE: consumes the seeding diaries directly (calibration-window
data); never imports the quarantined blind-truth package. Its BT1 prediction
is produced and FROZEN before any blind number is read; scoring against
truth happens only inside the guarded firing driver.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace as dc_replace
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agents.card_executor import RealizedDay, RealizedTrip
from calibration.sr520_target import sr520_target
from world import bridge, crn
from world.config import cityk_corridor
from world.network import facility_times_from_loads, realized_facilities, solve_corridor_equilibrium

#: CRN namespace prefix for the comparator's ensemble (fresh sites; nothing
#: shared with the method arm's streams).
NS_PREFIX = "cmp_run"

ENSEMBLE = 20
N_DAYS = 20  # replay window length per arm (steady state; behavior is static)


def _pseudo_cards(persona_index) -> List[dict]:
    """Minimal card shells (persona_id + skeleton) so the ONE bridge
    population path builds the comparator population too."""
    from grounding import seeding

    out = []
    for r in persona_index.itertuples(index=False):
        row = r._asdict()
        out.append({
            "persona_id": str(row["persona_id"]),
            "skeleton": {f: row.get(f) for f in seeding.SKELETON_FIELDS},
            "patterns": [], "rules": [],
        })
    return out


def _observed_days(dataset, persona_index) -> Dict[str, List[RealizedDay]]:
    """Each persona's observed weekday diary as RealizedDay lists (verbatim
    replay; day weights carried)."""
    from evaluation import e1
    from grounding import seeding

    enriched = seeding.enriched_trips(dataset)
    enriched["person_id"] = enriched["person_id"].astype(str)
    pop_map = e1.persona_of_person_map(persona_index)
    slots = e1.day_slots_by_persona(dataset, pop_map)
    person_of = {str(r.persona_id): str(r.person_id)
                 for r in persona_index.itertuples(index=False)}
    trips_by = {pid: g for pid, g in enriched.groupby("person_id")}

    out: Dict[str, List[RealizedDay]] = {}
    for pid, slot_list in slots.items():
        person_id = person_of.get(pid)
        g = trips_by.get(person_id)
        days: List[RealizedDay] = []
        for dn, w in slot_list:
            trips: List[RealizedTrip] = []
            if g is not None:
                sub = g[g["daynum"].astype(int) == int(dn)].sort_values("tripnum")
                trips = [RealizedTrip(t.purpose, t.mode, t.band) for t in
                         sub.itertuples(index=False)]
            days.append(RealizedDay(int(dn), float(w), trips))
        out[pid] = days
    return out


def corridor_drop(
    cards: List[dict],
    observed_days: Dict[str, List[RealizedDay]],
    config,
    namespace: str,
    schedule,
    persona_pass: Optional[Dict[str, bool]] = None,
) -> float:
    """Weekday tolled-facility volume drop fraction: replay the observed
    diaries through the corridor equilibrium with the toll ON vs OFF, same
    CRN route draws (paired twin worlds), N_DAYS each."""
    population = bridge.population_from_cards(cards, config, namespace,
                                              persona_pass=persona_pass)
    row_index = bridge.persona_row_index(cards)
    table = bridge.corridor_travelers_of_day(
        observed_days, cards, config, population=population, row_index=row_index
    )
    if len(table) == 0:
        return float("nan")
    free_state = config.network_state_for_day(200)   # era2: crossing open, free
    toll_state = config.network_state_for_day(300)   # era3: crossing tolled
    if schedule is not None:
        toll_state = dc_replace(toll_state, toll_schedule=schedule)

    loads = {}
    for arm, state in (("free", free_state), ("toll", toll_state)):
        t_code = "T"
        idx = state.facility_codes.index(t_code)
        facilities = [config.facility(c) for c in state.facility_codes]
        day_loads = []
        for d in range(N_DAYS):
            eq = solve_corridor_equilibrium(
                facilities, access=table.access, vot=table.vot,
                period_codes=table.period_codes, has_pass=table.has_pass,
                state=state, theta=config.logit_theta,
            )
            keys = ["%s:%s:%d:route" % (namespace, pid, d)
                    for pid in table.persona_ids]
            choice = realized_facilities(crn.draws(keys), eq.choice_probs)
            l = np.bincount(choice, minlength=len(facilities)).astype(float)
            day_loads.append(l[idx])
        loads[arm] = float(np.mean(day_loads))
    return 1.0 - loads["toll"] / loads["free"] if loads["free"] > 0 else float("nan")


def calibrate_vot_scale(cards, observed_days, persona_pass, schedule,
                        target: Optional[float] = None, iters: int = 10,
                        log=print) -> float:
    """Bisect the comparator's own VoT scale to the SR 520 plateau midpoint
    (the same anchor and level criterion the method's A4.3 fit uses)."""
    if target is None:
        target = sr520_target().drop_midpoint
    base = cityk_corridor()

    def drop_at(scale: float) -> float:
        cfg = dc_replace(base, vot_median=base.vot_median * scale)
        return corridor_drop(cards, observed_days, cfg, "cmp_cal",
                             schedule, persona_pass)

    lo, hi = 0.05, 8.0
    d_lo, d_hi = drop_at(lo), drop_at(hi)
    log(f"comparator bisect: drop({lo})={d_lo:.3f} drop({hi})={d_hi:.3f} "
        f"target={target:.3f}")
    if d_lo < target:
        return lo
    if d_hi > target:
        return hi
    mid = lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        d = drop_at(mid)
        if d > target:
            lo = mid
        else:
            hi = mid
        if abs(d - target) < 0.005:
            break
    return mid


def predict(cards, observed_days, persona_pass, vot_scale: float,
            schedule=None, n_runs: int = ENSEMBLE) -> dict:
    """The frozen BT1-facing prediction: CRN ensemble of weekday tunnel-volume
    drops under the M4 toll schedule (schedule=None = the config's own)."""
    base = cityk_corridor()
    cfg = dc_replace(base, vot_median=base.vot_median * vot_scale)
    drops = [
        corridor_drop(cards, observed_days, cfg, f"{NS_PREFIX}{k}",
                      schedule, persona_pass)
        for k in range(n_runs)
    ]
    a = np.asarray(drops, dtype=float)
    lo, hi = np.percentile(a, 10.0), np.percentile(a, 90.0)
    return {
        "drops": drops,
        "central": float(a.mean()),
        "interval_80": [float(lo), float(hi)],
        "n_runs": n_runs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/comparator_arm")
    ap.add_argument("--m4-gates", default="runs/m4_prep")
    ap.add_argument("--runs", type=int, default=ENSEMBLE)
    args = ap.parse_args(argv)

    from calibration.sr520_fit import _load_gates
    from calibration.sr520_target import sr520_rehearsal_schedule
    from grounding import seeding
    from grounding.adapters import psrc

    dataset = psrc.load_or_build()
    persona_index = seeding.persona_index(dataset)
    persona_pass, _ = _load_gates(args.m4_gates)
    cards = _pseudo_cards(persona_index)
    observed = _observed_days(dataset, persona_index)

    # A5.1: calibrated on the SAME rehearsal schedule as the method's A4.3 fit
    # (owner ruling 2026-07-17: the SR 520-derived masked schedule); the
    # PREDICTION runs under the masked M4 config schedule (schedule=None).
    # No say-do correction here — the comparator carries no stated-response
    # channel (A5.3(iii)).
    reh_schedule = sr520_rehearsal_schedule()
    scale = calibrate_vot_scale(cards, observed, persona_pass,
                                schedule=reh_schedule)
    pred = predict(cards, observed, persona_pass, scale)
    payload = {
        "amendment_draft": "A5 (docs/internal/AMENDMENT_A5_DRAFT.md)",
        "date": date.today().isoformat(),
        "model": "diary replay + frozen corridor route-choice logit; NO LLM",
        "calibration": {
            "anchor": "SR 520 plateau midpoint (calibration.sr520_target)",
            "vot_scale": scale,
            "rehearsal_schedule": {
                "source": "sr520",
                "rates_credits": dict(reh_schedule.rates),
                "nonpass_surcharge_credits": reh_schedule.nonpass_surcharge,
            },
            "note": "calibrated under the SR 520-derived masked rehearsal "
                    "schedule (owner ruling 2026-07-17, applied to this arm "
                    "identically per the A5 draft); prediction under the "
                    "masked M4 config schedule",
        },
        "prediction": pred,
        "drift_note": "deterministic replay: no machinery drift, no placebo "
                      "differencing needed",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(f"comparator: vot_scale={scale:.4f} central={pred['central']:.4f} "
          f"80% [{pred['interval_80'][0]:.4f}, {pred['interval_80'][1]:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
