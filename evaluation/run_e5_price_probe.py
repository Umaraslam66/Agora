"""E5(ii) fictional-price contamination probe (sealed A2.3(ii); §3 E5).

Sweeps toll prices never used in history — including implausible ones —
through the DEPLOYED M4 announced-onset pipeline (the two-brain loop, the
frozen habit threshold + VoT scale, the sealed say-do price correction), and
checks the two sealed conditions:

  (monotone)  the aggregate tolled-facility drop must be MONOTONE in price
              across the CRN-paired ensemble means (same route-draw
              namespaces at every multiplier, so the comparison is paired);
  (flag)      the response must NOT reproduce the famous historical
              aggregate at non-historical prices: a probe price >= 25% away
              from the perturbed historical schedule whose aggregate change
              lands within +/-2pp of -28% raises the A2.3(ii) contamination
              FLAG (the -28% figure and both window constants are quoted
              from the SEALED A2.3(ii) text itself — nothing here reads the
              quarantined truth package).

E5 sets FLAGS, not pass bars: a tripped flag marks every headline claim
contaminated until resolved (§3 E5); it never blocks the firing.

HARNESS-SIDE ONLY. The probe reuses `calibration.sr520_fit.run_arm` — ONE
code path for every announced-onset loop run (fit, rehearsal, probe) — but
sweeps the masked M4 CONFIG schedule (the BT1 instrument), never the SR 520
rehearsal schedule. The frozen fitted point (threshold, VoT scale) is read
from `calibration/sr520_fit_manifest.json`; run AFTER the fit freezes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from world.config import cityk_corridor
from world.tolling import SAY_DO_PRICE_CORRECTION

#: Sealed A2.3(ii) flag-window constants, primary arena (quoted from the
#: sealed text, NOT from any truth file): flag if the aggregate change is
#: within +/-2pp of -28% at any probe price >= 25% away from the perturbed
#: historical schedule.
HISTORICAL_AGGREGATE_DROP = 0.28
FLAG_WINDOW_PP = 0.02
MIN_PRICE_DISTANCE = 0.25

#: Probe multipliers of the masked M4 config schedule. 1.0 is the perturbed
#: historical schedule itself (monotonicity reference; never flag-eligible);
#: 0.0 is the free facility (the natural zero); 5.0 is the implausible
#: extreme the sealed text asks for.
MULTIPLIERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)

ENSEMBLE = 5


def probe(cards, client, *, threshold: int, vot_scale: float,
          persona_pass=None, car_access=None,
          multipliers=MULTIPLIERS, n_runs: int = ENSEMBLE,
          say_do_correction: float = SAY_DO_PRICE_CORRECTION,
          log=print) -> dict:
    from calibration.sr520_fit import _config_at, run_arm

    config = _config_at(vot_scale)
    base_schedule = cityk_corridor().toll_schedule

    drops: Dict[float, List[float]] = {float(m): [] for m in multipliers}
    for k in range(n_runs):
        ns = f"e5p_reh{k}"  # SAME namespace across multipliers -> CRN-paired
        for m in multipliers:
            t0 = time.time()
            stats = run_arm(
                cards, config=config, namespace=ns, arm="toll",
                threshold=threshold, client=client,
                persona_pass=persona_pass, car_access=car_access,
                schedule=base_schedule.with_multiplier(m),
                say_do_correction=say_do_correction,
            )
            drops[float(m)].append(float(stats["plateau_drop"]))
            log(f"  probe run {k} m={m}: drop={stats['plateau_drop']:.4f} "
                f"({time.time() - t0:.0f}s)")

    means = {m: float(np.mean(v)) for m, v in drops.items()}
    ms = sorted(means)
    monotone_ok = all(means[a] <= means[b] for a, b in zip(ms, ms[1:]))

    flags = []
    for m in ms:
        if abs(m - 1.0) < MIN_PRICE_DISTANCE:
            continue  # historical-adjacent price: not flag-eligible
        if abs(means[m] - HISTORICAL_AGGREGATE_DROP) <= FLAG_WINDOW_PP:
            flags.append({
                "multiplier": m,
                "mean_drop": means[m],
                "window": f"within +/-{FLAG_WINDOW_PP:.0%}pp of "
                          f"-{HISTORICAL_AGGREGATE_DROP:.0%}",
            })

    return {
        "multipliers": list(ms),
        "drops_by_multiplier": {str(m): drops[m] for m in ms},
        "mean_drop_by_multiplier": {str(m): means[m] for m in ms},
        "monotone_ok": bool(monotone_ok),
        "flag_windows_tripped": flags,
        "contamination_flag": bool(flags) or not monotone_ok,
        "sealed_constants": {
            "historical_aggregate_drop": HISTORICAL_AGGREGATE_DROP,
            "flag_window_pp": FLAG_WINDOW_PP,
            "min_price_distance": MIN_PRICE_DISTANCE,
            "source": "sealed A2.3(ii) text (01_PREREGISTRATION.md); no "
                      "truth import",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cards", required=True)
    ap.add_argument("--out", default="runs/e5_price_probe")
    ap.add_argument("--m4-gates", default=None, metavar="DIR")
    ap.add_argument("--generator", choices=("stub", "vllm"), default="stub",
                    help="stub = machinery shakeout ONLY; vllm = the real "
                         "model (the reportable probe)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--runs", type=int, default=ENSEMBLE)
    ap.add_argument("--context", default=None)
    ap.add_argument("--manifest", default="calibration/sr520_fit_manifest.json",
                    help="frozen fit manifest supplying threshold + VoT scale")
    args = ap.parse_args(argv)

    from agents.slow_brain import GatedSlowBrain, OnsetStubGenerator
    from calibration.sr520_fit import _load_cards, _load_gates, corridor_subpopulation

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("generator") != "vllm" and args.generator == "vllm":
        raise SystemExit("frozen manifest is not a real-model fit; refusing")
    threshold = int(manifest["strong_habit_threshold"])
    vot_scale = float(manifest["vot_scale"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "probe_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    cards = _load_cards(args.cards)
    persona_pass, car_access = _load_gates(args.m4_gates)
    sub = corridor_subpopulation(cards, cityk_corridor(), persona_pass=persona_pass)
    log(f"corridor subpopulation: {len(sub)} of {len(cards)} cards; "
        f"frozen point: threshold={threshold} vot_scale={vot_scale:.4f}")

    if args.context:
        vctx = json.loads(Path(args.context).read_text())
    else:
        from evaluation.run_m3 import _validation_context
        from grounding import seeding
        from grounding.adapters import psrc
        dataset = psrc.load_or_build()
        vctx = _validation_context(
            seeding.persona_index(dataset), dataset,
            seeding.enriched_trips(dataset), {c["persona_id"] for c in sub},
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

    result = probe(
        sub, client, threshold=threshold, vot_scale=vot_scale,
        persona_pass=persona_pass, car_access=car_access,
        n_runs=args.runs, log=log,
    )
    payload = {
        "eval": "E5(ii) fictional-price probe (sealed A2.3(ii))",
        "date": __import__("datetime").date.today().isoformat(),
        "generator": args.generator,
        "stub_warning": (
            None if args.generator == "vllm" else
            "STUB GENERATOR — machinery shakeout only; NOT a reportable probe"
        ),
        "cards": args.cards,
        "frozen_manifest": args.manifest,
        "threshold": threshold,
        "vot_scale": vot_scale,
        "say_do_price_correction": SAY_DO_PRICE_CORRECTION,
        "n_runs": args.runs,
        "result": result,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    log(f"probe: monotone_ok={result['monotone_ok']} "
        f"flags={len(result['flag_windows_tripped'])} "
        f"contamination_flag={result['contamination_flag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
