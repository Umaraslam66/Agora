"""BT1 assembly driver — the SINGLE blind firing (M4). HELD by default.

THIS MODULE FIRES THE ONE IRREVERSIBLE ACTION OF THE PROJECT. In the
Seattle primary arena BT1 *is* the tolling onset (A1.1): one firing scores
E4 (dQ), all BT1 tier arms of E7, the T4-noclaims arm, the placebo, the T5
tail-off, the E3(iii) uncorrected ablation, the E6 band arms, and the A5
comparator TOGETHER, then seals the verdict against ``evaluation/truth``
permanently. There is no re-run.

Two independent locks, neither of which this module may weaken:
  1. the PreToolUse guard hook blocks the entrypoint unless the command
     carries the owner's inline ``AGORA_BT1_AUTHORIZED=1``;
  2. this module itself refuses to run unless that variable is set in its
     environment, refuses a non-empty output directory (single firing),
     and refuses to SEAL from a stub generator.
The driver sets ``AGORA_EVAL_CONTEXT=1`` only immediately before the
scoring step — the loop/simulation phase runs without the truth package
importable, so no simulation code path can touch the answer key.

Everything consumed here is FROZEN and verified at load:
  * ``calibration/sr520_fit_manifest.json`` — θ (22), VoT scale (2.2238),
    say-do price correction (2.5), drift floor (0.0254), rehearsal-schedule
    provenance (must be "sr520"), generator (must be "vllm");
  * ``calibration/e3_fit_manifest.json`` — the sealed application point
    (the manifest's central must equal the fit manifest's factor);
  * the two sealed gates — household pass inheritance
    (``persona_pass.json``) and the borrowed-car draw from the T5 re-fit
    (owner ruling 2026-07-17: ACCEPT AS-IS at rate 1.0; the r2b-era
    degenerate manifest is rejected if offered);
  * the seven frozen tier populations (A4.1). BT1 fires the six BT1 arms
    (T1..T5 + T4_noclaims); T4_nofidelity is ordinary-day ONLY and this
    driver refuses to fire it;
  * the frozen A5 comparator prediction (scored from its manifest with the
    same E4 kernels — never re-run);
  * the E5(i)/E5(ii) contamination-flag records, carried into the verdict.

Scenario (implementation constants, recorded in the output manifest; the
day-window mapping below is the A1.6(v) reduction of trajectory scoring to
period averages — sim days are weekdays by construction):
  warm-up 10 days; baseline window 29 days (the A1.1 formal baseline
  period 2019-09-23..2019-10-31 spans 29 weekdays); onset at day 39; post
  window 63 days (~the three-month period the observed −28% measures),
  with the first 10 post days reported as the two-week trajectory point
  (observed −26%, secondary, never a bar).
  Scored per run: drop = 1 − mean(post-window tolled-facility volume) /
  mean(baseline-window volume) — the WHOLE-WINDOW mean, matching A1.1's
  period-average definition (not the calibration driver's plateau-tail).

Arms and pairing: ONE CRN namespace family ``bt1_r{k}`` is shared by every
arm, tier, threshold and ablation — so tier contrasts are information-only
(A4.1), toll−placebo differences are paired (A4.2), and the E6/E3 variants
are paired against the same draws. The A4.2 estimand is dQ = Q_toll −
Q_placebo per tier; the placebo world stays untolled and its notice is the
nulled reconsideration cue.

Verdict discipline: the sealed E4 bars and drift legs are evaluated
mechanically from ``evaluation.truth.bt1`` + the frozen floor; every
quantity is written to ``results.json`` with input hashes in
``manifest.json``. The driver never edits sealed text — the OWNER commits
the output; that commit is the seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from agents.baseline_loop import AnnouncedOnset, run_baseline_loop
from agents.card_executor import BorrowedCarAccess
from agents.slow_brain import GatedSlowBrain, OnsetStubGenerator, StandardSurprisePolicy
from evaluation import blind_shock
from world.config import cityk_corridor
from world.tolling import announcement_of, placebo_announcement

# ---------------------------------------------------------------------------
# scenario constants (implementation decisions; recorded in the manifest)
# ---------------------------------------------------------------------------

WARMUP_DAYS = 10
BASELINE_DAYS = 29           # A1.1 formal baseline period = 29 weekdays
ONSET_DAY = WARMUP_DAYS + BASELINE_DAYS
POST_3MO_DAYS = 63           # ~3 months of weekdays (the headline window)
POST_2WK_DAYS = 10           # first-two-weeks trajectory point (reported)
N_DAYS = ONSET_DAY + POST_3MO_DAYS

#: Era sample days for the network override (world/config timeline: era2 =
#: crossing open free, era3 = crossing tolled) — same seam as calibration.
ERA_FREE_DAY = 200
ERA_TOLL_DAY = 300

#: The six BT1 tier arms (A4.1/A4.4). T4_nofidelity is ordinary-day ONLY.
BT1_TIERS = ("T1", "T2", "T3", "T4", "T4_noclaims", "T5")
FORBIDDEN_TIERS = ("T4_nofidelity",)

#: E6 arm (b): habit memory ablated — no rule ever reaches strong-habit
#: immutability, so nothing is restored and adaptation is unconstrained
#: (the A2.5 memory-ablation analog on BT1 machinery; diagnostic here —
#: the E6 verdict proper is carried by the transfer arena).
ABLATED_THRESHOLD = 10**9

#: Owner ruling 2026-07-18 (driver review, item (b)): every E6 arm fired on
#: BT1 machinery is diagnostic only — the label is written into results.json
#: itself so the sealed output carries the discipline, not just a comment.
E6_ARM_LABEL = "diagnostic, verdict carried by transfer arena"

DEFAULT_RUNS = 20


# ---------------------------------------------------------------------------
# refusals (defense in depth — the guard hook is the outer lock)
# ---------------------------------------------------------------------------

def _refuse(msg: str) -> "SystemExit":
    return SystemExit(f"BT1 REFUSED: {msg}")


def _require_authorization() -> None:
    if os.environ.get("AGORA_BT1_AUTHORIZED") != "1":
        raise _refuse(
            "AGORA_BT1_AUTHORIZED=1 is not set. BT1 fires ONCE, only on the "
            "project owner's explicit inline authorization (CLAUDE.md; "
            "pre-registration §2/§7). This refusal is expected for every "
            "caller that is not the owner firing deliberately."
        )


def _require_single_firing(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise _refuse(
            f"output directory {out} is non-empty. BT1 is scored once and "
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

def load_frozen(fit_manifest: Path, e3_manifest: Path, *, allow_stub: bool) -> dict:
    fit = json.loads(fit_manifest.read_text())
    if fit.get("generator") != "vllm" and not allow_stub:
        raise _refuse("fit manifest is not a real-model fit (generator != vllm)")
    if fit.get("rehearsal_schedule", {}).get("source") != "sr520":
        raise _refuse(
            "fit manifest was not calibrated under the adopted SR 520-derived "
            "rehearsal schedule (owner ruling 2026-07-17)"
        )
    say_do = float(fit["say_do_price_correction"]["factor_applied"])
    e3 = json.loads(e3_manifest.read_text())
    ap = e3["price_prior"]["application_point"]
    if not str(ap.get("status", "")).startswith("SEALED"):
        raise _refuse("e3 manifest application point is not sealed")
    if float(ap["central_applied"]) != say_do:
        raise _refuse(
            f"sealed price-prior central {ap['central_applied']} != fit "
            f"manifest factor {say_do} — frozen artifacts disagree"
        )
    if fit.get("drift_floor") is None:
        raise _refuse("fit manifest carries no measured drift floor")
    return {
        "threshold": int(fit["strong_habit_threshold"]),
        "vot_scale": float(fit["vot_scale"]),
        "say_do": say_do,
        "drift_floor": float(fit["drift_floor"]),
        "e6_band": [int(t) for t in fit["e6_band_thresholds"]],
        "fit_manifest": fit,
    }


def load_gates(pass_dir: Path, borrowed_dir: Path) -> dict:
    persona_pass = json.loads((pass_dir / "persona_pass.json").read_text())
    bc = json.loads((borrowed_dir / "manifest.json").read_text())
    if bc.get("degenerate"):
        raise _refuse(
            f"{borrowed_dir} carries the DEGENERATE borrowed-car fit (the "
            "r2b-era manifest); BT1 must load the T5 re-fit the owner "
            "accepted (runs/m4_prep/borrowed_car_t5)"
        )
    if "owner_ruling_2026_07_17" not in bc:
        raise _refuse(
            f"{borrowed_dir}/manifest.json carries no owner ruling block; "
            "only the ACCEPT AS-IS-ruled T5 fit may fire"
        )
    qualifying = frozenset(
        json.loads((borrowed_dir / "qualifying_personas.json").read_text())
    )
    car_access = BorrowedCarAccess(rate=float(bc["fitted_rate"]), qualifying=qualifying)
    return {"persona_pass": persona_pass, "car_access": car_access,
            "borrowed_manifest": bc}


def load_tiers(tiers_dir: Path) -> dict:
    """The frozen tier populations + their tier-visible validation contexts
    (the tier sidecar is the CORRECT gate context: a T1 rewrite may not be
    gated against a diary T1 never saw)."""
    out = {}
    for tier in BT1_TIERS:
        cards_path = tiers_dir / tier / f"cards_{tier}.jsonl"
        ctx_path = tiers_dir / tier / "tier_context.json"
        with open(cards_path) as f:
            cards = [json.loads(line) for line in f if line.strip()]
        if len(cards) != 11940:
            raise _refuse(f"{cards_path}: {len(cards)} cards != 11940")
        out[tier] = {
            "cards": cards,
            "context": json.loads(ctx_path.read_text()),
            "cards_sha256": _sha256(cards_path),
        }
    for tier in FORBIDDEN_TIERS:
        # ordinary-day-only arm: never fired; presence on disk is fine.
        pass
    return out


# ---------------------------------------------------------------------------
# one arm run (BT1 windows; mirrors the calibration runner's plumbing but
# scores WINDOW MEANS per A1.1, not the plateau tail)
# ---------------------------------------------------------------------------

def _override(config, tolled: bool):
    free = config.network_state_for_day(ERA_FREE_DAY)
    toll = config.network_state_for_day(ERA_TOLL_DAY)  # the masked M4 schedule
    if not tolled:
        return lambda d: free
    return lambda d: (toll if d >= ONSET_DAY else free)


def run_arm(
    cards: Sequence[dict],
    *,
    config,
    namespace: str,
    arm: str,  # "toll" | "placebo" | "toll_tail_off"
    threshold: int,
    client,
    say_do: float,
    persona_pass,
    car_access,
) -> dict:
    tolled = arm != "placebo"
    onset = AnnouncedOnset(
        day=ONSET_DAY,
        announcement=announcement_of(config.toll_schedule,
                                     say_do_price_correction=say_do)
        if tolled else placebo_announcement(),
        tail_surprises=(arm != "toll_tail_off"),
    )
    res = run_baseline_loop(
        cards, config, {}, namespace=namespace, n_days=N_DAYS,
        warmup_days=WARMUP_DAYS, policy=StandardSurprisePolicy(),
        client=client, network_override=_override(config, tolled),
        keep_full_window=False, onset=onset,
        strong_habit_threshold=threshold,
        persona_pass=persona_pass, car_access=car_access,
    )
    series: Dict[int, float] = {}
    for d, rec in res.facility_loads.items():
        codes = rec["codes"]
        if "T" in codes:
            series[int(d)] = float(rec["loads"][codes.index("T")])
    base_days = [d for d in series if WARMUP_DAYS <= d < ONSET_DAY]
    post_days = sorted(d for d in series if d >= ONSET_DAY)
    base = float(np.mean([series[d] for d in base_days]))
    post_3mo = [series[d] for d in post_days[:POST_3MO_DAYS]]
    post_2wk = [series[d] for d in post_days[:POST_2WK_DAYS]]
    return {
        "arm": arm,
        "namespace": namespace,
        "baseline_mean_load": base,
        "drop_3mo": 1.0 - float(np.mean(post_3mo)) / base if base > 0 else float("nan"),
        "drop_2wk": 1.0 - float(np.mean(post_2wk)) / base if base > 0 else float("nan"),
        "n_rewrites_accepted": sum(1 for a in res.rewrite_audit if a.accepted),
        "n_rewrites_attempted": len(res.rewrite_audit),
        "surprise_total": int(sum(res.surprise_counts.values())),
    }


def _ensemble(
    cards, context, *, config, arms: Sequence[str], threshold: int, gen,
    say_do: float, gates, n_runs: int, label: str, log,
) -> Dict[str, List[dict]]:
    client = GatedSlowBrain(gen, context)
    out: Dict[str, List[dict]] = {a: [] for a in arms}
    for k in range(n_runs):
        ns = f"bt1_r{k}"  # ONE namespace family for every arm/tier/variant
        for arm in arms:
            t0 = time.time()
            stats = run_arm(
                cards, config=config, namespace=ns, arm=arm,
                threshold=threshold, client=client, say_do=say_do,
                persona_pass=gates["persona_pass"],
                car_access=gates["car_access"],
            )
            out[arm].append(stats)
            log(f"  {label} r{k} {arm}: drop3mo={stats['drop_3mo']:.4f} "
                f"({time.time() - t0:.0f}s)")
    return out


def _delta(toll_stats, placebo_stats, key="drop_3mo"):
    t = np.array([s[key] for s in toll_stats])
    p = np.array([s[key] for s in placebo_stats])
    d = blind_shock.paired_delta(t, p)
    return {"central": d.central, "median": d.median, "lo": d.lo, "hi": d.hi,
            "n": d.n, "members": d.delta.tolist()}


# ---------------------------------------------------------------------------
# the firing
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiers-dir", default="runs/e7_tiers")
    ap.add_argument("--m4-gates-pass", default="runs/m4_prep/has_pass_household")
    ap.add_argument("--borrowed-car", default="runs/m4_prep/borrowed_car_t5")
    ap.add_argument("--fit-manifest", default="calibration/sr520_fit_manifest.json")
    ap.add_argument("--e3-manifest", default="calibration/e3_fit_manifest.json")
    ap.add_argument("--comparator", default="runs/comparator_arm/manifest.json")
    ap.add_argument("--e5i", default="runs/e5_m3/results.json")
    ap.add_argument("--e5ii", default="runs/e5_price_probe/results.json")
    ap.add_argument("--generator", choices=("stub", "vllm"), default="vllm",
                    help="stub = machinery shakeout in a SCRATCH dir only; "
                         "its output is marked non-sealable")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--e6-runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--out", default="runs/bt1")
    args = ap.parse_args(argv)

    _require_authorization()
    out = Path(args.out)
    _require_single_firing(out)
    stub = args.generator == "stub"
    if stub and args.out == "runs/bt1":
        raise _refuse("stub shakeout may not write to the sealed output dir")

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "firing_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    # ---- load + verify every frozen input (fail before any run) ----------
    frozen = load_frozen(Path(args.fit_manifest), Path(args.e3_manifest),
                         allow_stub=stub)
    gates = load_gates(Path(args.m4_gates_pass), Path(args.borrowed_car))
    tiers = load_tiers(Path(args.tiers_dir))
    comparator = json.loads(Path(args.comparator).read_text())
    e5_flags = {}
    for name, p in (("e5i", args.e5i), ("e5ii", args.e5ii)):
        e5_flags[name] = (json.loads(Path(p).read_text())
                          if Path(p).exists() else {"missing": p})
    theta, vot_scale, say_do = (frozen["threshold"], frozen["vot_scale"],
                                frozen["say_do"])
    from dataclasses import replace as dc_replace
    base_cfg = cityk_corridor()
    config = dc_replace(base_cfg, vot_median=base_cfg.vot_median * vot_scale)
    log(f"BT1 FIRING: theta={theta} vot_scale={vot_scale} say_do={say_do} "
        f"floor={frozen['drift_floor']} runs={args.runs} generator={args.generator}")

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
    results: Dict[str, dict] = {"tiers": {}}
    for tier in BT1_TIERS:
        arms = ("toll", "placebo", "toll_tail_off") if tier == "T5" else ("toll", "placebo")
        ens = _ensemble(
            tiers[tier]["cards"], tiers[tier]["context"], config=config,
            arms=arms, threshold=theta, gen=gen, say_do=say_do, gates=gates,
            n_runs=args.runs, label=tier, log=log,
        )
        entry = {
            "arms": ens,
            "delta_q": _delta(ens["toll"], ens["placebo"]),
            "delta_q_2wk": _delta(ens["toll"], ens["placebo"], key="drop_2wk"),
        }
        if tier == "T5":
            entry["delta_q_tail_off"] = _delta(ens["toll_tail_off"], ens["placebo"])
        results["tiers"][tier] = entry

    # E3(iii) uncorrected ablation: T5 toll arm at factor 1.0, same seeds,
    # same placebo (the nulled notice carries no price for the factor to touch).
    ens_ab = _ensemble(
        tiers["T5"]["cards"], tiers["T5"]["context"], config=config,
        arms=("toll",), threshold=theta, gen=gen, say_do=1.0, gates=gates,
        n_runs=args.runs, label="T5_uncorrected", log=log,
    )
    results["e3_ablation"] = {
        "say_do": 1.0,
        "arms": ens_ab,
        "delta_q": _delta(ens_ab["toll"], results["tiers"]["T5"]["arms"]["placebo"]),
    }

    # E6 band arms on BT1 machinery (diagnostic; verdict proper = transfer
    # arena): toll+placebo per band threshold, plus the memory-ablated arm.
    results["e6_band"] = {}
    for th in frozen["e6_band"]:
        if th == theta:
            results["e6_band"][str(th)] = {
                "delta_q": results["tiers"]["T5"]["delta_q"], "reused": "T5",
                "discipline": E6_ARM_LABEL}
            continue
        ens_th = _ensemble(
            tiers["T5"]["cards"], tiers["T5"]["context"], config=config,
            arms=("toll", "placebo"), threshold=th, gen=gen, say_do=say_do,
            gates=gates, n_runs=args.e6_runs, label=f"e6_th{th}", log=log,
        )
        results["e6_band"][str(th)] = {
            "delta_q": _delta(ens_th["toll"], ens_th["placebo"]),
            "discipline": E6_ARM_LABEL}
    ens_abl = _ensemble(
        tiers["T5"]["cards"], tiers["T5"]["context"], config=config,
        arms=("toll", "placebo"), threshold=ABLATED_THRESHOLD, gen=gen,
        say_do=say_do, gates=gates, n_runs=args.e6_runs, label="e6_ablated", log=log,
    )
    results["e6_memory_ablated"] = {
        "delta_q": _delta(ens_abl["toll"], ens_abl["placebo"]),
        "discipline": E6_ARM_LABEL}

    # ---- phase 2: scoring against the sealed answer key -------------------
    os.environ["AGORA_EVAL_CONTEXT"] = "1"
    from evaluation.truth import bt1 as truth

    def kernels(delta_members: List[float]) -> dict:
        d = np.asarray(delta_members, dtype=float)
        cov = blind_shock.interval_coverage(d, truth.OBSERVED_DROP_3MO)
        clo = blind_shock.closer_than_benchmark(
            d, truth.OBSERVED_DROP_3MO, truth.FORECAST_BENCHMARK_DROP)
        return {"coverage": vars(cov), "closeness": vars(clo)}

    t5 = results["tiers"]["T5"]
    verdict = {
        "e4_headline_T5": kernels(t5["delta_q"]["members"]),
        "e4_trajectory": {
            "observed_2wk": truth.OBSERVED_DROP_2WK,
            "predicted_2wk": t5["delta_q_2wk"],
            "note": "A1.6(v): trajectory scoring reduced to period averages; "
                    "reported, not a bar",
        },
        "e7_bt1_curve": {
            tier: kernels(results["tiers"][tier]["delta_q"]["members"])
            for tier in BT1_TIERS
        },
        "e3_iii_transfer": {
            "corrected": kernels(t5["delta_q"]["members"]),
            "uncorrected": kernels(results["e3_ablation"]["delta_q"]["members"]),
            "clause": "the frozen correction must improve blind E4 vs the "
                      "uncorrected ablation, else the transfer claim FAILS "
                      "and is reported as such (A3.2)",
        },
        "tail_off_bound": blind_shock.tail_off_gap(
            np.asarray(t5["delta_q"]["members"]),
            np.asarray(t5["delta_q_tail_off"]["members"]),
        ).__dict__,
        "drift": {},
        "comparator": {
            "frozen_prediction": comparator["prediction"],
            **kernels(comparator["prediction"]["drops"]),
            "discipline": "REPORTED, not a pass bar (A5.2); comparative "
                          "reading = headline-adjacent reported quantity",
        },
        "e5_contamination_flags": e5_flags,
        "headline_discipline": "T5 is the SOLE headline (A4.1); T1-T4 and "
                               "all diagnostics may never be promoted",
    }
    floor = frozen["drift_floor"]
    for tier in BT1_TIERS:
        arms = results["tiers"][tier]["arms"]
        dv = blind_shock.drift_verdict(
            np.array([s["drop_3mo"] for s in arms["toll"]]),
            np.array([s["drop_3mo"] for s in arms["placebo"]]),
            floor,
        )
        verdict["drift"][tier] = vars(dv)

    # ---- seal step ---------------------------------------------------------
    payload = {
        "eval": "BT1 — single blind firing (A1.1; A4; A5)",
        "date": __import__("datetime").date.today().isoformat(),
        "generator": args.generator,
        "NON_SEALABLE_STUB": stub or None,
        "frozen": {
            "threshold": theta, "vot_scale": vot_scale, "say_do": say_do,
            "drift_floor": floor, "e6_band": frozen["e6_band"],
        },
        "scenario": {
            "warmup_days": WARMUP_DAYS, "baseline_days": BASELINE_DAYS,
            "onset_day": ONSET_DAY, "post_3mo_days": POST_3MO_DAYS,
            "post_2wk_days": POST_2WK_DAYS,
        },
        "runs": {"headline": args.runs, "e6": args.e6_runs},
        "results": results,
        "verdict": verdict,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    manifest = {
        "inputs": {
            "fit_manifest": {"path": args.fit_manifest, "sha256": _sha256(Path(args.fit_manifest))},
            "e3_manifest": {"path": args.e3_manifest, "sha256": _sha256(Path(args.e3_manifest))},
            "comparator": {"path": args.comparator, "sha256": _sha256(Path(args.comparator))},
            "borrowed_car": {"path": args.borrowed_car,
                             "ruling": gates["borrowed_manifest"]["owner_ruling_2026_07_17"]["ruling"]},
            "tier_cards_sha256": {t: tiers[t]["cards_sha256"] for t in BT1_TIERS},
        },
        "results_sha256": _sha256(out / "results.json"),
        "seal": ("NON-SEALABLE STUB SHAKEOUT" if stub else
                 "scored once; the owner's commit of this directory is the seal"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"BT1 {'STUB SHAKEOUT' if stub else 'FIRING'} COMPLETE -> {out} "
        f"(headline dQ central={t5['delta_q']['central']:.4f} "
        f"[{t5['delta_q']['lo']:.4f}, {t5['delta_q']['hi']:.4f}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
