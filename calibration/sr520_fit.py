"""A4.3 SR 520 joint calibration: habit-persistence threshold + VoT/elasticity,
and the A3.3(b)/A4.2 calibration rehearsal (E4 machinery + measured drift floor).

HARNESS-SIDE ONLY (calibration/ discipline): never imported by agent-facing
code. Everything scored here is the LABELED SR 520 calibration event pinned in
`calibration.sr520_target` — no blind quantity is touched, and this module
never imports the quarantined blind-truth package.

The scenario (A4.3): a permanent, masked corridor level shift — the A4.2(ii)
announced known price, on at the onset day, NEVER removed — run through the
two-brain loop on the corridor subpopulation of the deployed cards:

  days 0..warmup-1          warm-up (triggers disabled), free-crossing world
  warmup..onset-1           pre-onset window (free crossing; habit strengths
                            grow so every grid threshold is live at onset —
                            the real event's habits were years old)
  onset..n_days-1           permanent toll (era-3 state), announced onset

Corridor-subpopulation equivalence: the scored quantity is the tolled
facility's realized volume, and only corridor personas ever enter the
corridor equilibrium — excluding non-corridor personas changes that quantity
by ZERO and cuts cost ~8x. (BT1 itself fires on the FULL tier populations per
A4.1; this restriction is a fit/rehearsal cost decision only.)

Joint fit (A4.3): the plateau LEVEL is a price-elasticity property — fitted
with the VoT scale (config ``vot_median`` x scale; the toll term in the route
choice is toll/vot); the strong-habit THRESHOLD is fit to the transition
shape and drop-and-plateau persistence. The observed anchor is aggregate and
(monthly) coarse, so the shape criteria are deliberately ONE-SIDED — exactly
what A3.3 pins and nothing more:

  (level)        plateau drop inside DROP_BAND [0.36, 0.40];
  (transition)   plateau reached within TRANSITION_DAYS weekdays of onset
                 (the observed series shows the drop complete by its first
                 monthly observation);
  (persistence)  no drift back: the late-plateau drop may not fall more than
                 RECOVERY_TOL below the early-plateau drop (A3.3: no recovery
                 to baseline through the wall).

A threshold RANGE satisfying one-sided criteria is the expected honest
outcome (the anchor cannot discriminate inside it); the fitted point keeps
the provisional build constant 14 IF it lies inside the passing range (the
constant survives calibration), else the nearest passing threshold. The E6
sensitivity band (A4.3) is the range of thresholds whose trajectory stays
inside the CONFOUND-WIDENED envelope at the FROZEN fitted elasticity —
widened by ENVELOPE_WIDEN_PP for the A3.3 confounds (concurrent transit
boost, six rate steps, post-recession growth, parallel-crossing
construction, FY2018 counting break).

Rehearsal (A3.3(b) + A4.2, at the fitted point, ensemble N>=20):
  * toll arm vs CRN-paired placebo arm (yoked announced-onset trigger,
    nulled reason, untolled world) -> paired dQ = drop_toll - drop_placebo;
  * E4 machinery scored against the pinned forecast/actual pair (labeled,
    never blind): coverage of the realized band midpoint, central strictly
    closer than the 48% forecast;
  * the measured DRIFT FLOOR = placebo/toll magnitude ratio
    (`evaluation.blind_shock.measured_floor`), reported standalone in a
    dated note BEFORE BT1, with the two-leg self-verdict;
  * the T5 tail-off arm (post-onset time-surprise trigger OFF) -> the
    tail-off gap bounding the uncontrolled tail-drift channel.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.baseline_loop import AnnouncedOnset, run_baseline_loop
from agents.slow_brain import GatedSlowBrain, OnsetStubGenerator, StandardSurprisePolicy
from calibration.sr520_target import sr520_rehearsal_schedule, sr520_target
from evaluation import blind_shock
from world.bridge import population_from_cards
from world.config import cityk_corridor
from world.tolling import SAY_DO_PRICE_CORRECTION, announcement_of, placebo_announcement

# ---------------------------------------------------------------------------
# scenario + fit constants (implementation decisions, recorded in the manifest)
# ---------------------------------------------------------------------------

WARMUP_DAYS = 10
#: Pre-onset window: long enough that a near-daily rule's strength exceeds
#: every grid threshold at onset (the real event's habits were years old).
PRE_ONSET_DAYS = 25
POST_ONSET_DAYS = 40
ONSET_DAY = WARMUP_DAYS + PRE_ONSET_DAYS
N_DAYS = ONSET_DAY + POST_ONSET_DAYS

#: Era sample days for the network override (world/config timeline: era2 =
#: free crossing open, era3 = crossing tolled).
ERA_FREE_DAY = 200
ERA_TOLL_DAY = 300

#: One-sided shape criteria (see module docstring).
TRANSITION_DAYS = 20
RECOVERY_TOL = 0.03
PLATEAU_TAIL_DAYS = 10
EARLY_PLATEAU = (5, 15)  # days after onset for the early-plateau window

#: E6 envelope widening for the A3.3 confounds (reporting-robustness band,
#: not a sealed bar; the plateau level may sit this far outside DROP_BAND and
#: still count as inside the confound-widened envelope).
ENVELOPE_WIDEN_PP = 0.05

#: Threshold grid (build constant 14 included; upper values live because the
#: pre-onset window lets strengths reach ~PRE_ONSET_DAYS+WARMUP_DAYS).
THRESHOLD_GRID = (6, 10, 14, 18, 22, 26)

#: VoT-scale bisection bounds and tolerance (scale multiplies vot_median).
VOT_SCALE_LO = 0.05
VOT_SCALE_HI = 8.0
VOT_BISECT_ITERS = 10

REHEARSAL_ENSEMBLE = 20


# ---------------------------------------------------------------------------
# scenario plumbing
# ---------------------------------------------------------------------------

def corridor_subpopulation(cards: Sequence[dict], config, namespace: str = "sr520_sel",
                           persona_pass: Optional[Dict[str, bool]] = None) -> List[dict]:
    pop = population_from_cards(cards, config, namespace, persona_pass=persona_pass)
    return [c for c, on in zip(cards, pop.is_corridor) if bool(on)]


def _override(config, onset_day: int, tolled: bool, schedule=None):
    free = config.network_state_for_day(ERA_FREE_DAY)
    toll = config.network_state_for_day(ERA_TOLL_DAY, schedule=schedule)
    if not tolled:
        return lambda d: free
    return lambda d: (toll if d >= onset_day else free)


def run_arm(
    cards: Sequence[dict],
    *,
    config,
    namespace: str,
    arm: str,  # "toll" | "placebo" | "toll_tail_off"
    threshold: int,
    client,
    persona_pass: Optional[Dict[str, bool]] = None,
    car_access=None,
    schedule=None,
    say_do_correction: float = SAY_DO_PRICE_CORRECTION,
) -> dict:
    """One loop run of one arm; returns the per-day tolled-facility series and
    drop statistics. CRN pairing across arms = same namespace.

    ``schedule`` is the REHEARSAL toll schedule (owner ruling 2026-07-17:
    the SR 520-derived masked schedule, NOT the M4 config schedule) — it is
    both announced and charged; None falls back to ``config.toll_schedule``
    (the pre-ruling as-sealed reading, kept for the ablation/debug path).
    ``say_do_correction`` scales the ANNOUNCED charge only (A3.2 prior
    central, sealed application point) — the world always charges
    ``schedule`` itself; the placebo's nulled notice is untouched by it."""
    tolled = arm != "placebo"
    sched = schedule if schedule is not None else config.toll_schedule
    onset = AnnouncedOnset(
        day=ONSET_DAY,
        announcement=announcement_of(sched, say_do_price_correction=say_do_correction)
        if tolled else placebo_announcement(),
        tail_surprises=(arm != "toll_tail_off"),
    )
    res = run_baseline_loop(
        cards, config, {}, namespace=namespace, n_days=N_DAYS,
        warmup_days=WARMUP_DAYS, policy=StandardSurprisePolicy(),
        client=client,
        network_override=_override(config, ONSET_DAY, tolled, schedule=schedule),
        keep_full_window=False, onset=onset,
        strong_habit_threshold=threshold,
        persona_pass=persona_pass, car_access=car_access,
    )
    series = _facility_series(res.facility_loads, code="T")
    stats = _drop_stats(series)
    stats["arm"] = arm
    stats["namespace"] = namespace
    stats["n_rewrites_accepted"] = sum(1 for a in res.rewrite_audit if a.accepted)
    stats["n_rewrites_attempted"] = len(res.rewrite_audit)
    stats["surprise_total"] = int(sum(res.surprise_counts.values()))
    return stats


def _facility_series(facility_loads: Dict[int, dict], code: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for d, rec in facility_loads.items():
        codes = rec["codes"]
        if code in codes:
            out[int(d)] = float(rec["loads"][codes.index(code)])
    return out


def _drop_stats(series: Dict[int, float]) -> dict:
    """Baseline mean, per-day drop fractions, plateau/transition statistics."""
    base_days = [d for d in series if WARMUP_DAYS <= d < ONSET_DAY]
    post_days = sorted(d for d in series if d >= ONSET_DAY)
    base = float(np.mean([series[d] for d in base_days])) if base_days else float("nan")
    drop = {d: 1.0 - series[d] / base for d in post_days} if base and base > 0 else {}
    tail = post_days[-PLATEAU_TAIL_DAYS:]
    early = [d for d in post_days
             if ONSET_DAY + EARLY_PLATEAU[0] <= d < ONSET_DAY + EARLY_PLATEAU[1]]
    plateau = float(np.mean([drop[d] for d in tail])) if tail else float("nan")
    early_plateau = float(np.mean([drop[d] for d in early])) if early else float("nan")
    # settle day: first post-onset day whose 5-day forward rolling mean stays
    # within 3pp of the plateau for the rest of the window
    settle = None
    for i, d in enumerate(post_days):
        window_means = []
        for j in range(i, len(post_days) - 4):
            w = [drop[post_days[k]] for k in range(j, j + 5)]
            window_means.append(abs(float(np.mean(w)) - plateau))
        if window_means and max(window_means) <= 0.03:
            settle = d
            break
    return {
        "baseline_mean_load": base,
        "drop_by_day": {int(d): float(v) for d, v in drop.items()},
        "plateau_drop": plateau,
        "early_plateau_drop": early_plateau,
        "recovery": (early_plateau - plateau) if drop else float("nan"),
        "settle_days_after_onset": (settle - ONSET_DAY) if settle is not None else None,
    }


def shape_verdict(stats: dict) -> dict:
    """The one-sided A4.3 shape criteria on one arm's drop statistics."""
    t = sr520_target()
    lvl = stats["plateau_drop"]
    settle = stats["settle_days_after_onset"]
    recovery = stats["recovery"]
    level_ok = t.drop_band[0] <= lvl <= t.drop_band[1]
    envelope_ok = (t.drop_band[0] - ENVELOPE_WIDEN_PP) <= lvl <= (
        t.drop_band[1] + ENVELOPE_WIDEN_PP)
    transition_ok = settle is not None and settle <= TRANSITION_DAYS
    persistence_ok = bool(np.isfinite(recovery)) and recovery <= RECOVERY_TOL
    return {
        "level_ok": bool(level_ok),
        "envelope_ok": bool(envelope_ok),
        "transition_ok": bool(transition_ok),
        "persistence_ok": bool(persistence_ok),
        "shape_ok": bool(transition_ok and persistence_ok),
        "all_ok": bool(level_ok and transition_ok and persistence_ok),
    }


# ---------------------------------------------------------------------------
# the joint fit
# ---------------------------------------------------------------------------

def _config_at(vot_scale: float):
    base = cityk_corridor()
    return dc_replace(base, vot_median=base.vot_median * float(vot_scale))


def fit_vot_scale(
    cards: Sequence[dict],
    threshold: int,
    client,
    *,
    persona_pass=None,
    car_access=None,
    namespace: str = "sr520_fit",
    schedule=None,
    say_do_correction: float = SAY_DO_PRICE_CORRECTION,
    log=print,
) -> Tuple[float, dict]:
    """Bisect the VoT scale so the toll arm's plateau drop hits the pinned
    midpoint (drop is monotone DECREASING in VoT scale: cheaper time => the
    toll matters more). Returns (scale, that run's stats)."""
    target = sr520_target().drop_midpoint

    def drop_at(scale: float) -> Tuple[float, dict]:
        stats = run_arm(
            cards, config=_config_at(scale), namespace=namespace, arm="toll",
            threshold=threshold, client=client,
            persona_pass=persona_pass, car_access=car_access,
            schedule=schedule, say_do_correction=say_do_correction,
        )
        return stats["plateau_drop"], stats

    lo, hi = VOT_SCALE_LO, VOT_SCALE_HI
    d_lo, s_lo = drop_at(lo)
    d_hi, s_hi = drop_at(hi)
    log(f"  vot bisect th={threshold}: drop({lo})={d_lo:.3f} drop({hi})={d_hi:.3f} "
        f"target={target:.3f}")
    if d_lo < target:  # even the cheapest time cannot reach the level
        return lo, s_lo
    if d_hi > target:  # even the dearest time overshoots
        return hi, s_hi
    best = (lo, s_lo)
    for _ in range(VOT_BISECT_ITERS):
        mid = (lo + hi) / 2.0
        d_mid, s_mid = drop_at(mid)
        best = (mid, s_mid)
        if d_mid > target:
            lo = mid  # too much drop -> raise VoT
        else:
            hi = mid
        if abs(d_mid - target) < 0.005:
            break
    return best


def joint_fit(
    cards: Sequence[dict],
    client,
    *,
    persona_pass=None,
    car_access=None,
    thresholds: Sequence[int] = THRESHOLD_GRID,
    schedule=None,
    say_do_correction: float = SAY_DO_PRICE_CORRECTION,
    log=print,
) -> dict:
    """Per-threshold elasticity refit + shape evaluation; fitted point +
    passing range; E6 band at the frozen fitted elasticity."""
    per_threshold: Dict[int, dict] = {}
    for th in thresholds:
        scale, stats = fit_vot_scale(
            cards, th, client, persona_pass=persona_pass, car_access=car_access,
            namespace=f"sr520_fit_th{th}", schedule=schedule,
            say_do_correction=say_do_correction, log=log,
        )
        verdict = shape_verdict(stats)
        per_threshold[th] = {"vot_scale": scale, "stats": stats, "verdict": verdict}
        log(f"  th={th}: vot_scale={scale:.3f} plateau={stats['plateau_drop']:.3f} "
            f"settle={stats['settle_days_after_onset']} "
            f"recovery={stats['recovery']:.3f} all_ok={verdict['all_ok']}")

    passing = [th for th in thresholds if per_threshold[th]["verdict"]["all_ok"]]
    if passing:
        fitted_th = 14 if 14 in passing else min(passing, key=lambda t: abs(t - 14))
    else:
        # no threshold satisfies all criteria: report the closest-to-level one
        fitted_th = min(
            thresholds,
            key=lambda t: abs(per_threshold[t]["stats"]["plateau_drop"]
                              - sr520_target().drop_midpoint),
        )
    fitted_scale = per_threshold[fitted_th]["vot_scale"]

    # E6 band: thresholds whose trajectory stays inside the confound-widened
    # envelope AT THE FROZEN fitted elasticity (re-run, elasticity NOT refit —
    # refitting per threshold would make the band vacuous).
    band: List[int] = []
    band_runs: Dict[int, dict] = {}
    for th in thresholds:
        if th == fitted_th:
            stats = per_threshold[th]["stats"]
        else:
            stats = run_arm(
                cards, config=_config_at(fitted_scale),
                namespace=f"sr520_band_th{th}", arm="toll", threshold=th,
                client=client, persona_pass=persona_pass, car_access=car_access,
                schedule=schedule, say_do_correction=say_do_correction,
            )
        v = shape_verdict(stats)
        band_runs[th] = {"stats": stats, "verdict": v}
        if v["envelope_ok"] and v["persistence_ok"]:
            band.append(th)
        log(f"  band th={th} @scale={fitted_scale:.3f}: "
            f"plateau={stats['plateau_drop']:.3f} in_band={th in band}")

    return {
        "per_threshold": per_threshold,
        "passing_thresholds": passing,
        "fitted_threshold": fitted_th,
        "fitted_vot_scale": fitted_scale,
        "e6_band_thresholds": band,
        "band_runs": band_runs,
    }


# ---------------------------------------------------------------------------
# rehearsal ensemble (E4 machinery + drift floor + tail-off gap)
# ---------------------------------------------------------------------------

def rehearsal(
    cards: Sequence[dict],
    threshold: int,
    vot_scale: float,
    client,
    *,
    persona_pass=None,
    car_access=None,
    n_runs: int = REHEARSAL_ENSEMBLE,
    schedule=None,
    say_do_correction: float = SAY_DO_PRICE_CORRECTION,
    log=print,
) -> dict:
    config = _config_at(vot_scale)
    target = sr520_target()
    arms: Dict[str, List[dict]] = {"toll": [], "placebo": [], "toll_tail_off": []}
    for k in range(n_runs):
        ns = f"sr520_reh{k}"  # SAME namespace across arms -> CRN-paired
        for arm in arms:
            t0 = time.time()
            stats = run_arm(
                cards, config=config, namespace=ns, arm=arm, threshold=threshold,
                client=client, persona_pass=persona_pass, car_access=car_access,
                schedule=schedule, say_do_correction=say_do_correction,
            )
            arms[arm].append(stats)
            log(f"  reh run {k} {arm}: plateau={stats['plateau_drop']:.4f} "
                f"({time.time() - t0:.0f}s)")

    toll = np.array([s["plateau_drop"] for s in arms["toll"]])
    plac = np.array([s["plateau_drop"] for s in arms["placebo"]])
    tail_off = np.array([s["plateau_drop"] for s in arms["toll_tail_off"]])

    dq = blind_shock.paired_delta(toll, plac)
    dq_tail_off = blind_shock.paired_delta(tail_off, plac)
    gap = blind_shock.tail_off_gap(dq.delta, dq_tail_off.delta)

    observed = float(np.mean(target.rehearsal_realized_drop_band))
    coverage = blind_shock.interval_coverage(dq.delta, observed)
    closeness = blind_shock.closer_than_benchmark(
        dq.delta, observed, target.rehearsal_forecast_drop
    )
    floor = blind_shock.measured_floor(toll, plac)
    self_verdict = blind_shock.drift_verdict(toll, plac, floor)

    return {
        "n_runs": n_runs,
        "arms": arms,
        "delta_q": {"central": dq.central, "median": dq.median,
                    "lo": dq.lo, "hi": dq.hi, "n": dq.n,
                    "members": dq.delta.tolist()},
        "delta_q_tail_off": {"central": dq_tail_off.central,
                             "lo": dq_tail_off.lo, "hi": dq_tail_off.hi,
                             "members": dq_tail_off.delta.tolist()},
        "tail_off_gap": {"central": gap.central, "lo": gap.lo, "hi": gap.hi},
        "e4_rehearsal": {
            "observed_realized_mid": observed,
            "forecast_benchmark": target.rehearsal_forecast_drop,
            "coverage": vars(coverage),
            "closeness": vars(closeness),
        },
        "drift_floor": floor,
        "drift_floor_self_verdict": vars(self_verdict),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cards(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_gates(gates_dir: Optional[str]):
    """Lean loader for the frozen pre-M4 gate artifacts (pure JSON — no
    pandas; the cluster venv carries numpy/scipy/vllm only)."""
    if gates_dir is None:
        return None, None
    from agents.card_executor import BorrowedCarAccess

    gd = Path(gates_dir)
    persona_pass = json.loads((gd / "has_pass_household" / "persona_pass.json").read_text())
    bc = json.loads((gd / "borrowed_car" / "manifest.json").read_text())
    qualifying = frozenset(
        json.loads((gd / "borrowed_car" / "qualifying_personas.json").read_text())
    )
    car_access = BorrowedCarAccess(rate=float(bc["fitted_rate"]), qualifying=qualifying)
    return persona_pass, car_access


def export_context(cards_path: str, out_path: str, gates_dir: Optional[str]) -> int:
    """LOCAL prep step (needs pandas/PSRC data): compute the corridor
    subpopulation's validation context and write it as JSON, so the cluster
    run needs no PSRC data and no pandas."""
    from evaluation.run_m3 import _validation_context
    from grounding import seeding
    from grounding.adapters import psrc

    cards = _load_cards(cards_path)
    persona_pass, _ = _load_gates(gates_dir)
    sub = corridor_subpopulation(cards, cityk_corridor(), persona_pass=persona_pass)
    dataset = psrc.load_or_build()
    persona_index = seeding.persona_index(dataset)
    enriched = seeding.enriched_trips(dataset)
    vctx = _validation_context(
        persona_index, dataset, enriched, {c["persona_id"] for c in sub}
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(vctx, default=list))
    print(f"context: {len(vctx)} corridor personas -> {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cards", required=True)
    ap.add_argument("--out", default="runs/sr520_fit")
    ap.add_argument("--m4-gates", default=None, metavar="DIR")
    ap.add_argument("--generator", choices=("stub", "vllm"), default="stub",
                    help="stub = machinery shakeout ONLY (never reported); "
                         "vllm = the real model (the reportable fit)")
    ap.add_argument("--cache", default=None,
                    help="vLLM prompt cache JSONL (required for --generator vllm)")
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--rehearsal-runs", type=int, default=REHEARSAL_ENSEMBLE)
    ap.add_argument("--thresholds", type=int, nargs="*", default=list(THRESHOLD_GRID))
    ap.add_argument("--skip-rehearsal", action="store_true")
    ap.add_argument("--rehearsal-schedule", choices=("sr520", "config"),
                    default="sr520",
                    help="sr520 = the SR 520-derived masked schedule (owner "
                         "ruling 2026-07-17, the reportable fit); config = the "
                         "M4 config schedule (the pre-ruling as-sealed reading; "
                         "debug/comparison only)")
    ap.add_argument("--say-do-correction", type=float,
                    default=SAY_DO_PRICE_CORRECTION,
                    help="A3.2 stated->revealed factor applied to the ANNOUNCED "
                         "charge (sealed application point; 1.0 = the E3(iii) "
                         "uncorrected ablation)")
    ap.add_argument("--freeze", action="store_true",
                    help="write calibration/sr520_fit_manifest.json (the frozen "
                         "dated fit manifest; only meaningful with --generator vllm)")
    ap.add_argument("--context", default=None,
                    help="precomputed validation-context JSON (cluster mode: no "
                         "PSRC data / pandas needed)")
    ap.add_argument("--export-context", default=None, metavar="OUT",
                    help="LOCAL prep: write the corridor validation context to "
                         "OUT and exit")
    args = ap.parse_args(argv)

    if args.export_context:
        return export_context(args.cards, args.export_context, args.m4_gates)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "fit_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    cards = _load_cards(args.cards)
    persona_pass, car_access = _load_gates(args.m4_gates)
    config0 = cityk_corridor()
    sub = corridor_subpopulation(cards, config0, persona_pass=persona_pass)
    log(f"corridor subpopulation: {len(sub)} of {len(cards)} cards")

    # validation/render context for the gated slow brain (full observed stats)
    if args.context:
        vctx = json.loads(Path(args.context).read_text())
    else:
        from evaluation.run_m3 import _validation_context
        from grounding import seeding
        from grounding.adapters import psrc
        dataset = psrc.load_or_build()
        persona_index = seeding.persona_index(dataset)
        enriched = seeding.enriched_trips(dataset)
        vctx = _validation_context(
            persona_index, dataset, enriched, {c["persona_id"] for c in sub}
        )

    if args.generator == "vllm":
        if not args.cache:
            ap.error("--cache is required with --generator vllm")
        from serving.vllm_generator import CachedRewriteGenerator
        gen = CachedRewriteGenerator(
            cache_path=args.cache, tensor_parallel_size=args.tensor_parallel,
        )
    else:
        gen = OnsetStubGenerator()
    client = GatedSlowBrain(gen, vctx)

    reh_schedule = (
        sr520_rehearsal_schedule() if args.rehearsal_schedule == "sr520" else None
    )
    log(f"rehearsal schedule: {args.rehearsal_schedule} "
        f"rates={ {p: round(r, 4) for p, r in (reh_schedule or config0.toll_schedule).rates.items()} } "
        f"surcharge={(reh_schedule or config0.toll_schedule).nonpass_surcharge:.4f} "
        f"say_do_correction={args.say_do_correction}")

    t0 = time.time()
    fit = joint_fit(
        sub, client, persona_pass=persona_pass, car_access=car_access,
        thresholds=args.thresholds, schedule=reh_schedule,
        say_do_correction=args.say_do_correction, log=log,
    )
    log(f"fitted: threshold={fit['fitted_threshold']} "
        f"vot_scale={fit['fitted_vot_scale']:.4f} "
        f"passing={fit['passing_thresholds']} e6_band={fit['e6_band_thresholds']} "
        f"({time.time() - t0:.0f}s)")

    reh = None
    if not args.skip_rehearsal:
        reh = rehearsal(
            sub, fit["fitted_threshold"], fit["fitted_vot_scale"], client,
            persona_pass=persona_pass, car_access=car_access,
            n_runs=args.rehearsal_runs, schedule=reh_schedule,
            say_do_correction=args.say_do_correction, log=log,
        )
        log(f"rehearsal: dQ central={reh['delta_q']['central']:.4f} "
            f"[{reh['delta_q']['lo']:.4f},{reh['delta_q']['hi']:.4f}] "
            f"drift_floor={reh['drift_floor']:.4f} "
            f"tail_gap={reh['tail_off_gap']['central']:.4f}")

    sched_used = reh_schedule or config0.toll_schedule
    payload = {
        "date": __import__("datetime").date.today().isoformat(),
        "generator": args.generator,
        "generator_stats": gen.stats() if hasattr(gen, "stats") else None,
        "cards": args.cards,
        "n_corridor_cards": len(sub),
        "m4_gates": args.m4_gates,
        "rehearsal_schedule": {
            "source": args.rehearsal_schedule,
            "rates_credits": {p: sched_used.rates[p] for p in sched_used.rates},
            "nonpass_surcharge_credits": sched_used.nonpass_surcharge,
            "derivation": (
                "calibration.sr520_target.sr520_rehearsal_schedule: SR 520 "
                "opening weekday ladder (WAC 468-270-071 / WSR 11-04-007, "
                "corroborated by WSR 12-08-059's strikethrough baseline) "
                "hour-weighted onto the four masked periods, mapped to credits "
                "with the M4 masking's per-period credits-per-dollar factors "
                "(owner ruling 2026-07-17; docs/REHEARSAL_SCHEDULE_NOTE.md)"
                if args.rehearsal_schedule == "sr520"
                else "M4 config schedule (pre-ruling as-sealed reading)"
            ),
        },
        "say_do_price_correction": {
            "factor_applied": args.say_do_correction,
            "application_point": (
                "stimulus-side, announced charge only, INSIDE the pipeline "
                "BEFORE this elasticity fit (sealed 2026-07-17 in "
                "calibration/e3_fit_manifest.json -> price_prior."
                "application_point); world charges the un-scaled schedule; "
                "E3(iii) ablation runs factor 1.0 with elasticity unchanged"
            ),
        },
        "scenario": {
            "warmup_days": WARMUP_DAYS, "pre_onset_days": PRE_ONSET_DAYS,
            "post_onset_days": POST_ONSET_DAYS, "onset_day": ONSET_DAY,
            "era_free_day": ERA_FREE_DAY, "era_toll_day": ERA_TOLL_DAY,
            "corridor_subpopulation_note": (
                "only corridor personas enter the corridor equilibrium; the "
                "scored tolled-facility volume is identical to a full-population "
                "run — cost decision only, BT1 fires on full tier populations"
            ),
            "pre_onset_rationale": (
                "long enough that near-daily rules exceed every grid threshold "
                "at onset (real-event habits were years old)"
            ),
        },
        "criteria": {
            "drop_band": sr520_target().drop_band,
            "transition_days": TRANSITION_DAYS,
            "recovery_tol": RECOVERY_TOL,
            "plateau_tail_days": PLATEAU_TAIL_DAYS,
            "envelope_widen_pp": ENVELOPE_WIDEN_PP,
            "one_sided_note": (
                "the pinned anchor (aggregate drop band + drop-and-plateau on a "
                "monthly series) supports only one-sided transition/persistence "
                "criteria; a passing RANGE of thresholds is the expected honest "
                "outcome and feeds the A4.3 E6 sensitivity band"
            ),
        },
        "fit": fit,
        "rehearsal": reh,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))

    if args.freeze:
        manifest = {
            "amendment": "01_PREREGISTRATION.md section 7 A4.3",
            "date": payload["date"],
            "generator": args.generator,
            "stub_warning": (
                None if args.generator == "vllm" else
                "STUB GENERATOR — machinery shakeout only; NOT a reportable fit"
            ),
            "strong_habit_threshold": fit["fitted_threshold"],
            "vot_scale": fit["fitted_vot_scale"],
            "vot_median_effective": config0.vot_median * fit["fitted_vot_scale"],
            "e6_band_thresholds": fit["e6_band_thresholds"],
            "passing_thresholds": fit["passing_thresholds"],
            "rehearsal_schedule": payload["rehearsal_schedule"],
            "say_do_price_correction": payload["say_do_price_correction"],
            "criteria": payload["criteria"],
            "scenario": payload["scenario"],
            "drift_floor": (reh or {}).get("drift_floor"),
            "results": str(out / "results.json"),
        }
        Path("calibration/sr520_fit_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        log("froze calibration/sr520_fit_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
