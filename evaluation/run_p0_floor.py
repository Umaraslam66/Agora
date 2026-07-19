"""A8.3 P0 placebo-only arena-floor rehearsal (transfer arena). BLIND-SAFE.

Measures the transfer arena's drift floor BEFORE the BT2 firing, exactly as
A8.3 pins it: the cordon world with the charge NEVER on (this module is
structurally incapable of charging — it builds only placebo arms; no
announcement in this file ever carries a price), the nulled-notice trigger
fired at the SAME day offsets the real transitions will use
(evaluation.transfer_protocol.TRANSITION_DAYS), N = 20 CRN members. No
P1-P3 charged quantity exists or is touched; the truth package is never
imported.

Floor = the mean absolute placebo phase response: per member and per phase
in {P1, P2, P3} windows, drop = 1 − Q̄_phase / Q̄_P0 on the adopted-weights
cordon-crossing volume (all windows uncharged); the floor pools |drop| over
members and phases. Reported standalone with both A8.3 legs armed:
(i) anomaly leg — a firing placebo phase magnitude > 2× this floor;
(ii) absolute leg — it reaches 0.5.

CRN namespaces: an INDEPENDENT family ``p0floor_r{k}`` — the rehearsal is
its own measurement and does not consume the firing's ``bt2_r{k}`` draws
(recorded here; the BT1 floor was likewise measured on its own rehearsal
ensemble).

Compute: one pre-firing full-node run (A8.7(iv)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from agents.baseline_loop import AnnouncedOnset, run_baseline_loop
from agents.slow_brain import GatedSlowBrain, OnsetStubGenerator, StandardSurprisePolicy
from evaluation import transfer_protocol as tp
from evaluation.run_bt2 import (
    SCORED_PHASES,
    _reconstruction_check,
    _weighted_daily,
    _window_mean,
    load_borrowed_car,
    load_frozen,
    load_transfer_pop,
)
from world.config import cityk_cordon
from world.tolling import placebo_announcement

N_MEMBERS = 20  # A8.3 pin


def _placebo_onsets():
    """The three nulled-notice firings at the real transition offsets. The
    ONLY announcement this module can construct is the nulled cue."""
    return [
        AnnouncedOnset(day=tp.TRANSITION_DAYS[ph],
                       announcement=placebo_announcement(),
                       tail_surprises=True)
        for ph in ("P1", "P2", "P3")
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pop-dir", default="runs/transfer_pop")
    ap.add_argument("--weights", default="runs/transfer_reweight_adultsM1/weights.csv")
    ap.add_argument("--fit-manifest", default="calibration/sr520_fit_manifest.json")
    ap.add_argument("--borrowed-car", default="runs/m4_prep/borrowed_car_t5")
    ap.add_argument("--generator", choices=("stub", "vllm"), default="vllm")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--members", type=int, default=N_MEMBERS)
    ap.add_argument("--out", default="runs/p0_floor")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"P0 FLOOR REFUSED: {out} is non-empty — one "
                         "rehearsal, one record; use a fresh dir")
    out.mkdir(parents=True, exist_ok=True)
    stub = args.generator == "stub"

    frozen = load_frozen(Path(args.fit_manifest), allow_stub=stub)
    pop = load_transfer_pop(Path(args.pop_dir), Path(args.weights))
    bc = load_borrowed_car(Path(args.borrowed_car))
    theta, vot_scale = frozen["threshold"], frozen["vot_scale"]

    base_cfg = cityk_cordon()
    config = dc_replace(base_cfg, vot_median=base_cfg.vot_median * vot_scale)

    if stub:
        gen = OnsetStubGenerator()
    else:
        if not args.cache:
            ap.error("--cache is required with --generator vllm")
        from serving.vllm_generator import CachedRewriteGenerator
        gen = CachedRewriteGenerator(
            cache_path=args.cache, tensor_parallel_size=args.tensor_parallel,
        )
    client = GatedSlowBrain(gen, pop["context"])
    onsets = _placebo_onsets()

    members = []
    for k in range(args.members):
        ns = f"p0floor_r{k}"
        t0 = time.time()
        res = run_baseline_loop(
            pop["cards"], config, {}, namespace=ns, n_days=tp.TOTAL_DAYS,
            warmup_days=tp.WARMUP_DAYS, policy=StandardSurprisePolicy(),
            client=client, keep_full_window=False,
            onset=onsets[0], extra_onsets=onsets[1:],
            strong_habit_threshold=theta,
            car_access=bc["car_access"],
        )
        _reconstruction_check(res)
        q = _weighted_daily(res.cordon_crossing_days, pop["weights"], tp.TOTAL_DAYS)
        q0 = _window_mean(q, "P0")
        drops = {ph: (1.0 - _window_mean(q, ph) / q0 if q0 > 0 else float("nan"))
                 for ph in SCORED_PHASES}
        members.append({
            "namespace": ns, "q_p0": q0, "drops": drops,
            "n_rewrites_accepted": sum(1 for a in res.rewrite_audit if a.accepted),
            "surprise_total": int(sum(res.surprise_counts.values())),
        })
        print(f"p0floor r{k}: " +
              " ".join(f"{ph}={drops[ph]:+.5f}" for ph in SCORED_PHASES) +
              f" ({time.time() - t0:.0f}s)", flush=True)

    mags = np.array([[abs(m["drops"][ph]) for ph in SCORED_PHASES]
                     for m in members], dtype=float)
    floor = float(mags.mean())
    per_phase = {ph: float(mags[:, i].mean())
                 for i, ph in enumerate(SCORED_PHASES)}

    results = {
        "eval": "A8.3 P0 placebo-only arena drift floor (transfer arena)",
        "label": "METHOD-TRANSFER",
        "date": __import__("datetime").date.today().isoformat(),
        "generator": args.generator,
        "n_members": len(members),
        "threshold": theta,
        "floor": floor,
        "per_phase_mean_abs": per_phase,
        "legs_armed": {
            "anomaly": {"rule": "firing placebo phase magnitude > 2x floor",
                        "trip_at": 2.0 * floor},
            "absolute": {"rule": "firing placebo phase magnitude >= 0.5",
                         "trip_at": 0.5},
        },
        "namespace_family": "p0floor_r{k} (independent of the firing's bt2_r{k})",
        "transition_days": dict(tp.TRANSITION_DAYS),
        "members": members,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    manifest = {
        "inputs": {
            "transfer_cards_sha256": pop["cards_sha256"],
            "adopted_weights_sha256": hashlib.sha256(
                Path(args.weights).read_bytes()).hexdigest(),
            "fit_manifest": args.fit_manifest,
        },
        "results_sha256": hashlib.sha256(
            (out / "results.json").read_bytes()).hexdigest(),
        "note": "reported standalone BEFORE the BT2 firing (A8.3)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"P0 FLOOR COMPLETE -> {out} floor={floor:.6f} "
          f"(legs: anomaly>{2*floor:.6f}, absolute>=0.5)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
