"""BT2 assembly driver — the SINGLE transfer-arena blind firing (A8). HELD.

THIS MODULE FIRES THE SECOND AND FINAL IRREVERSIBLE ACTION OF THE PROJECT.
One firing runs the full P0->P1->P2->P3 phase timeline (A8.1) on the masked
cordon world for every E6 arm — arm (a) memory-on at each A4.3 band
threshold {6,10,14,18,22,26} and arm (b) memory-ablated — each with its own
yoked placebo (A8.2), scores the paired per-phase ΔQ estimand against
``evaluation.truth.transfer_p123``, applies the A8.3 two-leg drift rule
against the pre-measured P0 arena floor, emits the A8.4 channel
decomposition with the verdict, and seals. In THIS arena the E6 band arms
are the verdict proper (A8.2); E6's bands and pass conditions are A2.5's,
untouched.

Two independent locks, neither of which this module may weaken:
  1. the PreToolUse guard hook blocks the entrypoint (``run_bt2`` was
     entered into the matcher BEFORE this file existed, A8.5(iii)) unless
     the command carries the owner's inline ``AGORA_BT1_AUTHORIZED=1``;
  2. this module refuses to run without that variable, refuses a non-empty
     output directory (single firing), refuses to SEAL from a stub
     generator, and refuses to fire without the A8.3 floor rehearsal's
     results on disk (the floor must be REPORTED STANDALONE before the
     firing).
The truth package is imported only at the scoring step, after every
simulation arm has finished (AGORA_EVAL_CONTEXT is set immediately before
that import and never earlier).

Everything consumed here is FROZEN and verified at load:
  * ``calibration/sr520_fit_manifest.json`` — θ (the A4.3 fit), VoT scale
    (2.2238), say-do price correction (2.5), the E6 band; the frozen
    method's constants, applied unchanged (A1.2). The BT1 drift floor in
    that manifest is NOT used here — A8.3 pins the arena-2 floor to the P0
    placebo-only rehearsal.
  * the transfer population (``runs/transfer_pop``): 11,940 cards + the
    pop_context sidecar (the rewrite-gate reference), built by the frozen
    generation pipeline; its manifest pins the ADOPTED scoring weights
    (adults-only M1, owner ruling 2026-07-19) by sha256 — a swapped
    weights file refuses.
  * the masked cordon schedule (``runs/transfer_schedule/manifest.json``)
    — verified equal to ``world.tolling.CORDON_RATES`` at load, and the
    phase timeline verified equal to ``evaluation.transfer_protocol``.
  * the borrowed-car gate (CARRIES, A8.5(ii)); the household
    transponder-pass gate does NOT carry — this driver never loads one and
    the population build forces ``has_pass = False``.
  * the A8.3 floor (``runs/p0_floor/results.json``, N=20 placebo-only
    members, measured BEFORE the firing).

Scored quantity (A8.2): per phase p in {P1, P2, P3} and per CRN member,
Q̄ = the adopted-weights-weighted mean daily cordon-crossing car-travel
volume over the phase's scored window (evaluation.transfer_protocol);
drop_p = 1 − Q̄_p / Q̄_P0; ΔQ_p = drop_p(arm) − drop_p(placebo), paired.
P1 reduction and P3 level read against the truth series' stabilized
readings (coverage + closeness); P2 residual is E6's quantity, judged by
A2.5's band on arm (a) vs arm (b) with non-overlapping 80% intervals,
reported across the full A4.3 sensitivity band.

A8.4 output discipline: results.json carries, per phase × arm × member,
the exact two-term split ΔQ̄ = s̄_b·(N̄_b−N̄_p) + N̄_p·(s̄_b−s̄_p) (demand
term = cordon-crossing car-travel leaves, the card channel; share term =
crossing share of car-travel, the dial channel; residual identically zero
by construction — asserted). Pre-declared REPORTED reading: at P2 the
share term is expected arm-(a)-vs-(b) equal within CRN noise, the E6
separation carried by the demand term — whatever is found is reported.
Self-checks before sealing: per-agent crossing reconstruction (the
per-persona day lists must rebuild every recorded daily aggregate EXACTLY)
and the decomposition identity to <1e-9 per member.

P0 level-matching: NOT performed — every scored quantity is a relative
drop, so no demand/capacity constant needs matching; recorded here per the
TRANSFER_MASKING_NOTE's "reported, not worked around" discipline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from agents.baseline_loop import AnnouncedOnset, run_baseline_loop
from agents.card_executor import BorrowedCarAccess
from agents.slow_brain import GatedSlowBrain, OnsetStubGenerator, StandardSurprisePolicy
from evaluation import blind_shock
from evaluation import transfer_protocol as tp
from world.config import cityk_cordon
from world.tolling import CORDON_RATES, announcement_of, placebo_announcement

#: E6 arm (b): habit memory ablated (no rule ever strong; A2.5 arm (b)).
ABLATED_THRESHOLD = 10**9

#: A8.2/A2.5: the E6 pass band on arm (a)'s P2 residual (fractions of P0)
#: and the arm-(b) cap. Encoded as data so the driver cannot restate them.
E6_BAND_LO, E6_BAND_HI = 0.04, 0.12
E6_ARM_B_CAP = 0.04

DEFAULT_RUNS = 20
SCORED_PHASES = ("P1", "P2", "P3")


def _refuse(msg: str) -> "SystemExit":
    return SystemExit(f"BT2 REFUSED: {msg}")


def _require_authorization() -> None:
    if os.environ.get("AGORA_BT1_AUTHORIZED") != "1":
        raise _refuse(
            "AGORA_BT1_AUTHORIZED=1 is not set. BT2 fires ONCE, only on the "
            "project owner's explicit inline authorization (CLAUDE.md; A8). "
            "This refusal is expected for every caller that is not the owner "
            "firing deliberately."
        )


def _require_single_firing(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise _refuse(
            f"output directory {out} is non-empty. BT2 is scored once and "
            "sealed; a second run is post-hoc and cannot overwrite the "
            "verdict. If this is a stub shakeout, use a fresh scratch dir."
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# frozen-artifact loading (verify, never trust silently)
# ---------------------------------------------------------------------------

def load_frozen(fit_manifest: Path, *, allow_stub: bool) -> dict:
    fit = json.loads(fit_manifest.read_text())
    if fit.get("generator") != "vllm" and not allow_stub:
        raise _refuse("fit manifest is not a real-model fit (generator != vllm)")
    if fit.get("rehearsal_schedule", {}).get("source") != "sr520":
        raise _refuse("fit manifest was not calibrated under the adopted "
                      "SR 520-derived rehearsal schedule")
    return {
        "threshold": int(fit["strong_habit_threshold"]),
        "vot_scale": float(fit["vot_scale"]),
        "say_do": float(fit["say_do_price_correction"]["factor_applied"]),
        "e6_band": [int(t) for t in fit["e6_band_thresholds"]],
        "fit_manifest": fit,
    }


def load_transfer_pop(pop_dir: Path, weights_csv: Path) -> dict:
    build = json.loads((pop_dir / "build_manifest.json").read_text())
    pinned = build["scoring_weights"]["weights_csv_sha256"]
    actual = _sha256(weights_csv)
    if actual != pinned:
        raise _refuse(
            f"adopted weights {weights_csv} sha256 {actual} != the population "
            f"build's pin {pinned} — the weights the owner adopted are not "
            "the weights on disk"
        )
    cards_path = pop_dir / "cards_transfer.jsonl"
    with open(cards_path) as f:
        cards = [json.loads(line) for line in f if line.strip()]
    if len(cards) != 11940:
        raise _refuse(f"{cards_path}: {len(cards)} cards != 11940")
    weights: Dict[str, float] = {}
    with open(weights_csv) as f:
        header = f.readline().strip().split(",")
        pid_col, w_col = header.index("persona_id"), header.index("weight")
        for line in f:
            parts = line.strip().split(",")
            weights[parts[pid_col]] = float(parts[w_col])
    missing = [c["persona_id"] for c in cards if str(c["persona_id"]) not in weights]
    if missing:
        raise _refuse(f"{len(missing)} personas have no adopted weight "
                      f"(first: {missing[:3]})")
    return {
        "cards": cards,
        "context": json.loads((pop_dir / "pop_context.json").read_text()),
        "weights": weights,
        "cards_sha256": _sha256(cards_path),
        "build_manifest": build,
    }


def load_schedule_pin(schedule_manifest: Path) -> dict:
    sched = json.loads(schedule_manifest.read_text())
    if sched["masked_cordon_rates_credits"] != CORDON_RATES:
        raise _refuse("public schedule manifest rates != world.tolling."
                      "CORDON_RATES — the recorded schedule is not the "
                      "schedule the world charges")
    t = sched["phase_timeline"]
    pins = {
        "phase_bounds": {k: list(v) for k, v in tp.PHASE_BOUNDS.items()},
        "scored_windows": {k: list(v) for k, v in tp.SCORED_WINDOWS.items()},
        "transition_days": dict(tp.TRANSITION_DAYS),
        "phase_multiplier": dict(tp.PHASE_MULTIPLIER),
    }
    for key, want in pins.items():
        if t[key] != want:
            raise _refuse(f"schedule manifest {key} != evaluation."
                          f"transfer_protocol.{key.upper()} — the public "
                          "record and the implementation disagree")
    return sched


def load_floor(floor_results: Path) -> dict:
    """The A8.3 arena floor, measured by the P0 placebo-only rehearsal and
    REPORTED STANDALONE before this driver may fire."""
    if not floor_results.exists():
        raise _refuse(
            f"{floor_results} not found — A8.3 requires the P0 placebo-only "
            "floor rehearsal to be run and reported BEFORE the firing"
        )
    fl = json.loads(floor_results.read_text())
    if int(fl.get("n_members", 0)) < 20:
        raise _refuse("floor rehearsal has < 20 CRN members (A8.3 pins N=20)")
    return {"floor": float(fl["floor"]), "results": fl}


def load_borrowed_car(borrowed_dir: Path) -> dict:
    bc = json.loads((borrowed_dir / "manifest.json").read_text())
    if bc.get("degenerate"):
        raise _refuse(f"{borrowed_dir} carries the DEGENERATE borrowed-car fit")
    if "owner_ruling_2026_07_17" not in bc:
        raise _refuse(f"{borrowed_dir}/manifest.json carries no owner ruling block")
    qualifying = frozenset(
        json.loads((borrowed_dir / "qualifying_personas.json").read_text())
    )
    return {"car_access": BorrowedCarAccess(rate=float(bc["fitted_rate"]),
                                            qualifying=qualifying),
            "manifest": bc}


# ---------------------------------------------------------------------------
# one arm run (the full A8.1 phase timeline in ONE continuous loop)
# ---------------------------------------------------------------------------

def _onsets(charged: bool, config, say_do: float) -> List[AnnouncedOnset]:
    """The three announced transitions (A8.1). Charged arms announce the
    masked schedule at P1/P3 and the zero-multiplier schedule at P2 (the
    say-do factor multiplies an announced charge of zero — mechanically
    harmless, recorded); placebo arms fire the SAME trigger days with the
    nulled reconsideration cue (A8.2)."""
    out = []
    for phase in ("P1", "P2", "P3"):
        day = tp.TRANSITION_DAYS[phase]
        if charged:
            sched = config.toll_schedule.with_multiplier(tp.PHASE_MULTIPLIER[phase])
            ann = announcement_of(sched, say_do_price_correction=say_do)
        else:
            ann = placebo_announcement()
        out.append(AnnouncedOnset(day=day, announcement=ann, tail_surprises=True))
    return out


def _weighted_daily(day_lists: Dict[str, List[int]], weights: Dict[str, float],
                    n_days: int) -> np.ndarray:
    """Adopted-weights daily series from per-persona day lists."""
    q = np.zeros(n_days, dtype=float)
    for pid, days in day_lists.items():
        w = weights[pid]
        for d in days:
            q[d] += w
    return q


def _window_mean(series: np.ndarray, phase: str) -> float:
    lo, hi = tp.SCORED_WINDOWS[phase]  # 0-indexed inclusive bounds
    return float(series[lo:hi + 1].mean())


def _reconstruction_check(res) -> None:
    """A8.4 self-check: the per-persona day lists must rebuild every recorded
    daily aggregate EXACTLY (unweighted counts)."""
    n_car = {d: 0 for d in res.cordon_daily}
    n_cross = {d: 0 for d in res.cordon_daily}
    for pid, days in res.cordon_car_days.items():
        for d in days:
            n_car[d] += 1
    for pid, days in res.cordon_crossing_days.items():
        for d in days:
            n_cross[d] += 1
    for d, rec in res.cordon_daily.items():
        if n_car[d] != rec["n_car"] or n_cross[d] != rec["n_crossing"]:
            raise _refuse(
                f"per-agent crossing reconstruction failed at day {d}: "
                f"rebuilt ({n_car[d]}, {n_cross[d]}) != recorded "
                f"({rec['n_car']}, {rec['n_crossing']}) — sealing forbidden"
            )


def run_arm(
    cards: Sequence[dict],
    *,
    config,
    namespace: str,
    charged: bool,
    threshold: int,
    client,
    say_do: float,
    car_access,
    weights: Dict[str, float],
) -> dict:
    onsets = _onsets(charged, config, say_do)
    res = run_baseline_loop(
        cards, config, {}, namespace=namespace, n_days=tp.TOTAL_DAYS,
        warmup_days=tp.WARMUP_DAYS, policy=StandardSurprisePolicy(),
        client=client, keep_full_window=False,
        onset=onsets[0], extra_onsets=onsets[1:],
        strong_habit_threshold=threshold,
        car_access=car_access,
    )
    _reconstruction_check(res)
    q = _weighted_daily(res.cordon_crossing_days, weights, tp.TOTAL_DAYS)
    nn = _weighted_daily(res.cordon_car_days, weights, tp.TOTAL_DAYS)
    q0 = _window_mean(q, "P0")
    out = {
        "arm": "charged" if charged else "placebo",
        "namespace": namespace,
        "q_p0": q0,
        "drops": {}, "q_mean": {}, "n_mean": {},
        "n_rewrites_accepted": sum(1 for a in res.rewrite_audit if a.accepted),
        "n_rewrites_attempted": len(res.rewrite_audit),
        "surprise_total": int(sum(res.surprise_counts.values())),
        "strong_rule_stats": _habit_summary(res.cards, threshold),
        "daily_q": q.tolist(), "daily_n": nn.tolist(),
    }
    for phase in SCORED_PHASES:
        qp = _window_mean(q, phase)
        out["drops"][phase] = 1.0 - qp / q0 if q0 > 0 else float("nan")
        out["q_mean"][phase] = qp
        out["n_mean"][phase] = _window_mean(nn, phase)
    out["q_mean"]["P0"] = q0
    out["n_mean"]["P0"] = _window_mean(nn, "P0")
    return out


def _habit_summary(cards, threshold: int) -> dict:
    """A8.4(3): habit-trajectory summary at end of run — mean strength of
    rules, fraction at/above the arm's bar (final-state snapshot; the
    per-transition restoration counts live in the rewrite audit)."""
    strengths: List[int] = []
    for card in cards:
        counters = card.get("habit_counters", {})
        strengths.extend(int(c.get("strength", 0)) for c in counters.values())
    if not strengths:
        return {"mean_strength": 0.0, "frac_strong": 0.0, "n_rules": 0}
    arr = np.asarray(strengths, dtype=float)
    return {
        "mean_strength": float(arr.mean()),
        "frac_strong": float((arr >= threshold).mean()),
        "n_rules": int(arr.size),
    }


def _decompose(charged: dict, placebo: dict, phase: str) -> dict:
    """A8.4 two-term identity per member-pair: ΔQ̄ = s̄_b(N̄_b−N̄_p) +
    N̄_p(s̄_b−s̄_p), residual zero by construction (asserted <1e-9)."""
    qb, qp = charged["q_mean"][phase], placebo["q_mean"][phase]
    nb, np_ = charged["n_mean"][phase], placebo["n_mean"][phase]
    sb = qb / nb if nb > 0 else 0.0
    sp = qp / np_ if np_ > 0 else 0.0
    demand = sb * (nb - np_)
    share = np_ * (sb - sp)
    resid = (qb - qp) - (demand + share)
    if abs(resid) > 1e-9:
        raise _refuse(f"decomposition identity residual {resid} at {phase} — "
                      "sealing forbidden")
    return {"demand_term": demand, "share_term": share, "residual": resid,
            "q_charged": qb, "q_placebo": qp, "n_charged": nb, "n_placebo": np_}


def _ensemble(cards, context, *, config, threshold: int, gen, say_do: float,
              car_access, weights, n_runs: int, label: str, log) -> dict:
    client = GatedSlowBrain(gen, context)
    out = {"charged": [], "placebo": [], "decomposition": []}
    for k in range(n_runs):
        ns = f"bt2_r{k}"  # ONE namespace family: arms/thresholds all paired
        pair = {}
        for charged in (True, False):
            t0 = time.time()
            stats = run_arm(
                cards, config=config, namespace=ns, charged=charged,
                threshold=threshold, client=client, say_do=say_do,
                car_access=car_access, weights=weights,
            )
            pair["charged" if charged else "placebo"] = stats
            log(f"  {label} r{k} {'charged' if charged else 'placebo'}: "
                f"P1={stats['drops']['P1']:.4f} P2={stats['drops']['P2']:.4f} "
                f"P3={stats['drops']['P3']:.4f} ({time.time() - t0:.0f}s)")
        out["charged"].append(pair["charged"])
        out["placebo"].append(pair["placebo"])
        out["decomposition"].append(
            {ph: _decompose(pair["charged"], pair["placebo"], ph)
             for ph in SCORED_PHASES})
        # daily series are heavy; keep member 0 only (trajectory record)
        if k > 0:
            for arm in ("charged", "placebo"):
                pair[arm].pop("daily_q", None)
                pair[arm].pop("daily_n", None)
    return out


def _phase_delta(ens: dict, phase: str) -> dict:
    d = blind_shock.paired_delta(
        [s["drops"][phase] for s in ens["charged"]],
        [s["drops"][phase] for s in ens["placebo"]],
    )
    return {"central": d.central, "median": d.median, "lo": d.lo, "hi": d.hi,
            "n": d.n, "members": d.delta.tolist()}


def _drift_a83(ens: dict, phase: str, floor: float) -> dict:
    """A8.3 two-leg rule, arena-2 form: the firing's placebo phase-response
    MAGNITUDE (|central placebo drop|) trips on > 2x the measured floor
    (anomaly) or >= 0.5 (absolute)."""
    mag = abs(float(np.mean([s["drops"][phase] for s in ens["placebo"]])))
    anomaly = bool(mag > 2.0 * floor)
    absolute = bool(mag >= 0.5)
    return {"placebo_magnitude": mag, "floor": floor,
            "anomaly_leg": anomaly, "absolute_leg": absolute,
            "drift_dominated": bool(anomaly or absolute)}


# ---------------------------------------------------------------------------
# the firing
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pop-dir", default="runs/transfer_pop")
    ap.add_argument("--weights", default="runs/transfer_reweight_adultsM1/weights.csv")
    ap.add_argument("--fit-manifest", default="calibration/sr520_fit_manifest.json")
    ap.add_argument("--schedule-manifest", default="runs/transfer_schedule/manifest.json")
    ap.add_argument("--borrowed-car", default="runs/m4_prep/borrowed_car_t5")
    ap.add_argument("--floor", default="runs/p0_floor/results.json")
    ap.add_argument("--generator", choices=("stub", "vllm"), default="vllm")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--out", default="runs/bt2")
    args = ap.parse_args(argv)

    _require_authorization()
    out = Path(args.out)
    _require_single_firing(out)
    stub = args.generator == "stub"
    if stub and args.out == "runs/bt2":
        raise _refuse("stub shakeout may not write to the sealed output dir")

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "firing_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    # ---- load + verify every frozen input (fail before any run) ----------
    frozen = load_frozen(Path(args.fit_manifest), allow_stub=stub)
    pop = load_transfer_pop(Path(args.pop_dir), Path(args.weights))
    sched_pin = load_schedule_pin(Path(args.schedule_manifest))
    floor = load_floor(Path(args.floor))
    bc = load_borrowed_car(Path(args.borrowed_car))

    theta, vot_scale, say_do = (frozen["threshold"], frozen["vot_scale"],
                                frozen["say_do"])
    base_cfg = cityk_cordon()
    config = dc_replace(base_cfg, vot_median=base_cfg.vot_median * vot_scale)
    log(f"BT2 FIRING: theta={theta} vot_scale={vot_scale} say_do={say_do} "
        f"arena_floor={floor['floor']} runs={args.runs} "
        f"generator={args.generator}")

    if stub:
        gen = OnsetStubGenerator()
    else:
        if not args.cache:
            ap.error("--cache is required with --generator vllm")
        from serving.vllm_generator import CachedRewriteGenerator
        gen = CachedRewriteGenerator(
            cache_path=args.cache, tensor_parallel_size=args.tensor_parallel,
        )

    # ---- phase 1: all simulation arms (truth package NOT importable) ------
    # E6 arm (a) at every A4.3 band threshold + arm (b) ablated; one yoked
    # placebo per scored arm (A8.2). The fitted theta's arm is the (a)
    # headline; the band arms ARE the verdict proper in this arena.
    results: Dict[str, dict] = {"band": {}}
    for th in frozen["e6_band"]:
        ens = _ensemble(
            pop["cards"], pop["context"], config=config, threshold=th,
            gen=gen, say_do=say_do, car_access=bc["car_access"],
            weights=pop["weights"], n_runs=args.runs, label=f"a_th{th}", log=log,
        )
        results["band"][str(th)] = {
            "arms": ens,
            "delta_q": {ph: _phase_delta(ens, ph) for ph in SCORED_PHASES},
        }
    ens_b = _ensemble(
        pop["cards"], pop["context"], config=config, threshold=ABLATED_THRESHOLD,
        gen=gen, say_do=say_do, car_access=bc["car_access"],
        weights=pop["weights"], n_runs=args.runs, label="b_ablated", log=log,
    )
    results["memory_ablated"] = {
        "arms": ens_b,
        "delta_q": {ph: _phase_delta(ens_b, ph) for ph in SCORED_PHASES},
    }

    # ---- phase 2: scoring against the sealed answer key -------------------
    os.environ["AGORA_EVAL_CONTEXT"] = "1"
    from evaluation.truth import transfer_p123 as truth

    def kernels(members: List[float], observed: float) -> dict:
        d = np.asarray(members, dtype=float)
        return {"coverage": vars(blind_shock.interval_coverage(d, observed))}

    headline = results["band"][str(theta)]
    p2_a = headline["delta_q"]["P2"]
    p2_b = results["memory_ablated"]["delta_q"]["P2"]

    def e6_verdict(a_delta: dict, b_delta: dict) -> dict:
        """A2.5 on the paired P2 residuals: arm (a) central in the sealed
        band, arm (b) central below the cap, non-overlapping 80% intervals."""
        return {
            "arm_a_central": a_delta["central"],
            "arm_a_interval": [a_delta["lo"], a_delta["hi"]],
            "arm_a_in_band": bool(E6_BAND_LO <= a_delta["central"] <= E6_BAND_HI),
            "arm_b_central": b_delta["central"],
            "arm_b_interval": [b_delta["lo"], b_delta["hi"]],
            "arm_b_below_cap": bool(b_delta["central"] < E6_ARM_B_CAP),
            "intervals_disjoint": bool(a_delta["lo"] > b_delta["hi"]
                                       or b_delta["lo"] > a_delta["hi"]),
            "band": [E6_BAND_LO, E6_BAND_HI], "arm_b_cap": E6_ARM_B_CAP,
            "observed_anchor": list(truth.P2_RESIDUAL_RANGE),
            "anchor_caveat": truth.P2_CAVEAT,
        }

    verdict = {
        "e6_headline": e6_verdict(p2_a, p2_b),
        "e6_sensitivity_band": {
            th: e6_verdict(results["band"][th]["delta_q"]["P2"], p2_b)
            for th in results["band"]
        },
        "p1_introduction": {
            **kernels(headline["delta_q"]["P1"]["members"],
                      truth.P1_DROP_STABILIZED),
            "observed": truth.P1_DROP_STABILIZED,
            "alt_reading_e09": truth.P1_DROP_E09_STABILIZED,
            "forecast_context": {
                "planning_target": list(truth.P1_FORECAST_PLANNING_TARGET),
                "model_range": list(truth.P1_FORECAST_MODEL_RANGE)},
        },
        "p3_return": {
            **kernels(headline["delta_q"]["P3"]["members"],
                      truth.P3_DROP_FIRST_YEAR),
            "observed": truth.P3_DROP_FIRST_YEAR,
            "first_month_diagnostic": truth.P3_DROP_FIRST_MONTH,
        },
        "drift": {
            ph: _drift_a83(headline["arms"], ph, floor["floor"])
            for ph in SCORED_PHASES
        },
        "decomposition_reading": {
            "pre_declared": "at P2 the share term is expected arm-(a)-vs-(b) "
                            "equal within CRN noise; the E6 separation sits "
                            "in the demand term (A8.4) — reported as found",
            "p2_share_term_a_mean": float(np.mean(
                [m["P2"]["share_term"]
                 for m in headline["arms"]["decomposition"]])),
            "p2_share_term_b_mean": float(np.mean(
                [m["P2"]["share_term"]
                 for m in results["memory_ablated"]["arms"]["decomposition"]])),
            "p2_demand_term_a_mean": float(np.mean(
                [m["P2"]["demand_term"]
                 for m in headline["arms"]["decomposition"]])),
            "p2_demand_term_b_mean": float(np.mean(
                [m["P2"]["demand_term"]
                 for m in results["memory_ablated"]["arms"]["decomposition"]])),
        },
        "headline_discipline": (
            "the E6 band arms carry the verdict proper in this arena (A8.2); "
            "every number is METHOD-TRANSFER-labeled (A1.3); the approved "
            "marginals' child-share limitation is restated in the sealed "
            "verdict record verbatim from ADOPTION_RULING.md"
        ),
    }

    payload = {
        "eval": "BT2 — single transfer-arena blind firing (A8; A2.5 E6)",
        "label": "METHOD-TRANSFER",
        "date": __import__("datetime").date.today().isoformat(),
        "generator": args.generator,
        "NON_SEALABLE_STUB": stub or None,
        "frozen": {
            "threshold": theta, "vot_scale": vot_scale, "say_do": say_do,
            "e6_band": frozen["e6_band"], "arena_floor": floor["floor"],
        },
        "protocol": {
            "phase_bounds": {k: list(v) for k, v in tp.PHASE_BOUNDS.items()},
            "scored_windows": {k: list(v) for k, v in tp.SCORED_WINDOWS.items()},
            "transition_days": dict(tp.TRANSITION_DAYS),
            "warmup_days": tp.WARMUP_DAYS, "total_days": tp.TOTAL_DAYS,
            "masked_cordon_rates_credits": sched_pin["masked_cordon_rates_credits"],
            "masked_daily_cap_credits": sched_pin["masked_daily_cap_credits"],
            "p0_level_matching": "not performed — all scored quantities are "
                                 "relative drops (TRANSFER_MASKING_NOTE)",
        },
        "runs": args.runs,
        "results": results,
        "verdict": verdict,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    manifest = {
        "inputs": {
            "fit_manifest": {"path": args.fit_manifest,
                             "sha256": _sha256(Path(args.fit_manifest))},
            "schedule_manifest": {"path": args.schedule_manifest,
                                  "sha256": _sha256(Path(args.schedule_manifest))},
            "transfer_cards_sha256": pop["cards_sha256"],
            "adopted_weights": {"path": args.weights,
                                "sha256": _sha256(Path(args.weights))},
            "floor": {"path": args.floor, "sha256": _sha256(Path(args.floor))},
            "borrowed_car": {"path": args.borrowed_car,
                             "ruling": bc["manifest"]["owner_ruling_2026_07_17"]["ruling"]},
        },
        "results_sha256": _sha256(out / "results.json"),
        "seal": ("NON-SEALABLE STUB SHAKEOUT" if stub else
                 "scored once; the owner's commit of this directory is the seal"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"BT2 {'STUB SHAKEOUT' if stub else 'FIRING'} COMPLETE -> {out} "
        f"(E6 arm-a P2 residual central={p2_a['central']:.4f} "
        f"[{p2_a['lo']:.4f}, {p2_a['hi']:.4f}]; "
        f"arm-b central={p2_b['central']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
