#!/usr/bin/env python3
"""M3 baseline-loop driver: rescore E1/E2 end-to-end through the ordinary-day
two-brain loop (design D6).

WHY THIS FILE EXISTS — at M2 the sealed scorers consumed ``execute_days`` output
directly. M3 runs the world-coupled baseline loop (agents.baseline_loop) as an
ensemble of N runs (namespaces ``m3_run{k}``), then feeds each run's scoring
window into the UNCHANGED sealed scoring paths through their producer-injection
seam (``e1.simulate_arm(..., producer=)`` and ``e2.score_e2(..., producer=)``).
The MNL falsification arm and the E1/E2 statistics are otherwise byte-for-byte
the M2 machinery.

Quarantine (design D6): this driver NEVER imports ``evaluation.truth``. E5 wiring
lives elsewhere.

Usage:
  run_m3.py --cards CARDS.jsonl --out runs/m3_baseline --runs 20 --seed 20260717 \
            --warmup 10 --scoring-days 7 [--slow-brain none|stub|batch] [--limit N] [--label L]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from agents.baseline_loop import run_baseline_loop
from evaluation import e1
from evaluation import mnl_arm as mnl
from grounding import seeding
from grounding.adapters import psrc
from world.config import cityk_corridor


# ---------------------------------------------------------------------------
# card IO (same tolerance as run_e1 / run_e2)
# ---------------------------------------------------------------------------

def load_cards(path: str) -> List[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _date_stamp() -> str:
    try:
        return subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _safe_map_meta() -> dict:
    try:
        from grounding import zone_map
        return zone_map.map_metadata()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# slow-brain wiring (design D5/D6)
# ---------------------------------------------------------------------------

class _NoTriggerPolicy:
    """Fallback surprise policy used ONLY when agents.slow_brain is not yet
    present (parallel build). It records surprises to the card log but never
    triggers a rewrite — the correct behaviour for a scored baseline where
    organic triggers are expected to be ~0, and it keeps the driver runnable
    for timing smokes before the canonical StandardSurprisePolicy ships."""

    def log_surprise(self, card: dict, event) -> None:
        log = card.setdefault("surprise_log", [])
        log.append({
            "day_index": event.day_index,
            "context_key": event.context_key,
            "expected": event.expected_minutes,
            "realized": event.realized_minutes,
            "z": event.z,
            "status": "open",
        })
        # cap at agents.habit_memory.SURPRISE_LOG_CAP (=5), oldest dropped
        while len(log) > 5:
            log.pop(0)

    def should_rewrite(self, card, day_index: int) -> bool:
        return False


def _resolve_policy():
    """Prefer the canonical StandardSurprisePolicy (agents.slow_brain, D3);
    fall back to the no-trigger policy with a clear warning if it is not built
    yet."""
    try:
        from agents.slow_brain import StandardSurprisePolicy  # type: ignore
        return StandardSurprisePolicy(), "agents.slow_brain.StandardSurprisePolicy"
    except Exception as exc:  # noqa: BLE001 — parallel build; degrade loudly
        print(f"WARNING: agents.slow_brain.StandardSurprisePolicy unavailable ({exc}); "
              "using no-trigger fallback policy (baseline collects zero rewrites).",
              file=sys.stderr)
        return _NoTriggerPolicy(), "run_m3._NoTriggerPolicy(fallback)"


def _native(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def _validation_context(persona_index, dataset, enriched, card_personas):
    """Assemble the per-persona validation context the gated slow brain needs
    (persona_id -> {skeleton, observed, observed_day_sequences}), built from the
    SAME seeding frames generation used — mirroring run_e1.build_fallback_cards's
    per-person grouping. Only the personas whose cards are being run are built."""
    from grounding import seeding
    from grounding.card_validation import day_signatures

    et = enriched.copy()
    et["person_id"] = et["person_id"].astype(str)
    trips_by_person = {pid: grp for pid, grp in et.groupby("person_id")}
    pdays = dataset.person_days.copy()
    pdays["person_id"] = pdays["person_id"].astype(str)
    days_by_person = {pid: grp for pid, grp in pdays.groupby("person_id")}
    empty_days, empty_trips = dataset.person_days.iloc[0:0], et.iloc[0:0]

    ctx = {}
    for r in persona_index.itertuples(index=False):
        persona_id = str(r.persona_id)
        if persona_id not in card_personas:
            continue
        person_id = str(r.person_id)
        pdd = days_by_person.get(person_id, empty_days)
        ptr = trips_by_person.get(person_id, empty_trips)
        ctx[persona_id] = {
            "skeleton": {f: _native(getattr(r, f)) for f in seeding.SKELETON_FIELDS},
            "observed": seeding.observed_stats_of(pdd, ptr),
            "observed_day_sequences": day_signatures(pdd, ptr),
        }
    return ctx


def _resolve_stub_client(persona_index, dataset, enriched, card_personas):
    """Lazily construct the deterministic stub-backed gated slow brain
    (agents.slow_brain.GatedSlowBrain over a StubGenerator) for a REHEARSAL that
    actually rewrites. Degrades with a clear error if the slow-brain module is
    not present yet."""
    try:
        from agents.slow_brain import GatedSlowBrain, StubGenerator  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "ERROR: --slow-brain stub requires agents.slow_brain "
            f"(GatedSlowBrain + StubGenerator), not importable yet ({exc}). Use "
            "--slow-brain none for the collect-only baseline, or wait for the "
            "slow-brain module to land."
        )
    vc = _validation_context(persona_index, dataset, enriched, card_personas)
    return GatedSlowBrain(StubGenerator(), vc)


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------

def run(cards_path: str, out_dir: str, n_runs: int, base_seed: int, warmup: int,
        scoring_days: int, slow_brain: str = "none", limit: Optional[int] = None,
        mnl_iters: int = 250, label: str = "") -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- data -------------------------------------------------------------
    dataset = psrc.load_or_build()
    persona_index = seeding.persona_index(dataset)
    enriched = seeding.enriched_trips(dataset)

    cell_of_household, drops = e1.repinned_cell_of_household(dataset)
    persona_of_person = e1.persona_of_person_map(persona_index)
    persona_cell = e1.persona_cell_map(persona_index, cell_of_household)
    day_slots = e1.day_slots_by_persona(dataset, persona_of_person)

    truth = e1.truth_distributions(dataset, cell_of_household)
    matrices = e1.household_family_matrices(dataset)

    cards = load_cards(cards_path)
    if limit is not None:
        cards = cards[:limit]
    card_personas = {c["persona_id"] for c in cards}

    config = cityk_corridor()
    n_days = warmup + scoring_days
    policy, policy_name = _resolve_policy()
    # none/batch collect (client=None); stub actually rewrites via the gated brain
    client = (_resolve_stub_client(persona_index, dataset, enriched, card_personas)
              if slow_brain == "stub" else None)

    # -- loop ensemble (memoized per loop namespace; E1 and E2 share runs) --
    loop_scoring: Dict[str, Dict] = {}
    audit = {"rewrite_audit": [], "pending_rewrites": [], "surprise_counts": {}}

    def loop_for(namespace: str) -> Dict:
        if namespace not in loop_scoring:
            res = run_baseline_loop(
                cards, config, day_slots, namespace=namespace, n_days=n_days,
                warmup_days=warmup, policy=policy, client=client, keep_full_window=False,
            )
            loop_scoring[namespace] = res.scoring_days
            audit["rewrite_audit"].extend(r.to_dict() for r in res.rewrite_audit)
            audit["pending_rewrites"].extend(res.pending_rewrites)
            for d, c in res.surprise_counts.items():
                audit["surprise_counts"][d] = audit["surprise_counts"].get(d, 0) + c
        return loop_scoring[namespace]

    def e1_producer(namespace: str) -> Dict:
        return loop_for(namespace)  # namespace == "m3_run{k}"

    def e2_producer(namespace: str) -> Dict:
        # e2 namespaces are "run{k}"; map to the shared m3 loop runs
        k = namespace[len("run"):] if namespace.startswith("run") else namespace
        return loop_for(f"m3_run{k}")

    # -- E1 (method arm through the loop; MNL arm unchanged) --------------
    method = e1.simulate_arm(cards, dataset, persona_cell, day_slots,
                             n_runs=n_runs, namespace_prefix="m3_", producer=e1_producer)
    t_method = time.time()

    arm = mnl.build_mnl_arm(dataset, persona_index, enriched, cell_of_household,
                            day_slots, iters=mnl_iters)
    mnl_dists = e1.ensemble_arm(arm.producer, persona_cell, n_runs=n_runs,
                                namespace_prefix="mnl_")
    t_mnl = time.time()

    method_pooled = e1.pooled_tvd(method.pooled, truth.pooled)
    mnl_pooled = e1.pooled_tvd(mnl_dists.pooled, truth.pooled)
    method_cells = e1.cell_tvds(method.cells, truth.cells)
    mnl_cells = e1.cell_tvds(mnl_dists.cells, truth.cells)
    boot = e1.paired_bootstrap(method.pooled, mnl_dists.pooled, matrices=matrices,
                               B=500, base_seed=base_seed)
    worst_method_cell = max(method_cells, key=method_cells.get)
    pooled_ok = method_pooled <= e1.POOLED_BAR
    cells_ok = all(v <= e1.CELL_BAR for v in method_cells.values())

    # -- E2 (through the same shared loop runs) --------------------------
    from evaluation import e2  # local import: e2 pulls pandas, keep graph light
    e2_results = e2.score_e2(cards, dataset, n_runs=n_runs, seed=0, producer=e2_producer)
    t_e2 = time.time()

    # -- collected rewrites (batch mode / collect-only) ------------------
    pending_path = _write_pending_rewrites(out, audit["pending_rewrites"], slow_brain)

    results = {
        "label": label,
        "bars": {"pooled": e1.POOLED_BAR, "cell": e1.CELL_BAR, "epsilon": e1.EPSILON},
        "n_runs": n_runs,
        "loop": {
            "warmup_days": warmup, "scoring_days": scoring_days, "n_days": n_days,
            "namespaces": [f"m3_run{k}" for k in range(n_runs)],
            "policy": policy_name, "slow_brain": slow_brain,
            "surprise_counts_by_day": {str(k): v for k, v in sorted(audit["surprise_counts"].items())},
            "n_rewrite_requests_collected": len(audit["pending_rewrites"]),
            "n_rewrites_accepted": sum(1 for r in audit["rewrite_audit"] if r["accepted"]),
            "n_rewrites_attempted": len(audit["rewrite_audit"]),
        },
        "method_arm": {
            "pooled_tvd": method_pooled,
            "per_family_tvd": e1.per_family_tvd(method.pooled, truth.pooled),
            "cell_tvds": method_cells,
            "worst_cell": worst_method_cell,
            "worst_cell_tvd": method_cells[worst_method_cell],
            "pooled_bar_met": bool(pooled_ok),
            "cells_bar_met": bool(cells_ok),
        },
        "mnl_arm": {
            "pooled_tvd": mnl_pooled,
            "per_family_tvd": e1.per_family_tvd(mnl_dists.pooled, truth.pooled),
            "cell_tvds": mnl_cells,
        },
        "falsification": boot,
        "e1_pass": {
            "pooled_bar": bool(pooled_ok),
            "cell_bars": bool(cells_ok),
            "beats_or_matches_mnl": bool(boot["pass"]),
            "overall": bool(pooled_ok and cells_ok and boot["pass"]),
        },
        "e2": e2_results,
        "coverage": {
            "n_personas": len(persona_index),
            "n_cards": len(cards),
            "n_personas_with_card": len(card_personas & set(day_slots)),
            "limited": limit is not None,
        },
        "drops": drops,
        "timing_seconds": {
            "e1_method_arm": round(t_method - t0, 1),
            "mnl_arm": round(t_mnl - t_method, 1),
            "e2": round(t_e2 - t_mnl, 1),
            "total": round(time.time() - t0, 1),
        },
    }

    manifest = {
        "eval": "m3",
        "adapter_version": psrc.ADAPTER_VERSION,
        "map_version": _safe_map_meta(),
        "taxonomy": "m0-1.0",
        "cards_path": str(cards_path),
        "cards_sha256": _sha256_file(cards_path),
        "n_runs": n_runs,
        "bootstrap_B": 500,
        "base_seed": base_seed,
        "seed_derivation": boot["seed_derivation"],
        "mnl_fit_iters": mnl_iters,
        "mnl_diagnostics": arm.diagnostics,
        "loop_constants": {
            "warmup_days": warmup, "scoring_days": scoring_days, "n_days": n_days,
            "strong_habit_threshold": __import__("agents.baseline_loop",
                                                 fromlist=["STRONG_HABIT_THRESHOLD"]).STRONG_HABIT_THRESHOLD,
            "world_config": config.name,
            "slow_brain_mode": slow_brain,
            "policy": policy_name,
        },
        "rewrite_audit_summary": {
            "n_requests_collected": len(audit["pending_rewrites"]),
            "n_accepted": sum(1 for r in audit["rewrite_audit"] if r["accepted"]),
            "n_attempted": len(audit["rewrite_audit"]),
        },
        "pending_rewrites_path": pending_path,
        "namespaces": [f"m3_run{k}" for k in range(n_runs)],
        "timestamp": _date_stamp(),
    }

    (out / "results.json").write_text(json.dumps(_to_native(results), indent=2))
    (out / "manifest.json").write_text(json.dumps(_to_native(manifest), indent=2))
    return results


def _write_pending_rewrites(out: Path, pending: List[dict], slow_brain: str) -> Optional[str]:
    """Write collected rewrite requests for the offline cluster driver.

    In ``batch`` mode requests are grouped per day into
    ``rewrites/day{d}_requests.jsonl`` (the M2 batch_gen pattern); in the
    collect-only ``none`` mode a single ``pending_rewrites.jsonl`` is written.
    Returns the directory/file path written, or None when there is nothing to
    collect (the ordinary baseline outcome)."""
    if not pending:
        return None
    if slow_brain == "batch":
        rdir = out / "rewrites"
        rdir.mkdir(parents=True, exist_ok=True)
        by_day: Dict[int, List[dict]] = {}
        for req in pending:
            by_day.setdefault(int(req["day_index"]), []).append(req)
        for d, reqs in sorted(by_day.items()):
            with (rdir / f"day{d}_requests.jsonl").open("w", encoding="utf-8") as fh:
                for req in reqs:
                    fh.write(json.dumps(_to_native(req), sort_keys=True))
                    fh.write("\n")
        print(f"HALT (--slow-brain batch): wrote {len(pending)} organic rewrite "
              f"request(s) under {rdir}/ for the offline cluster round-trip. Apply "
              "the gated outcomes and resume from the day checkpoint before reading "
              "M3 numbers.", file=sys.stderr)
        return str(rdir)
    path = out / "pending_rewrites.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for req in pending:
            fh.write(json.dumps(_to_native(req), sort_keys=True))
            fh.write("\n")
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cards", required=True, help="persona cards JSONL / JSON array")
    ap.add_argument("--out", required=True, help="output run directory")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=e1.DEFAULT_BASE_SEED)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--scoring-days", type=int, default=7)
    ap.add_argument("--slow-brain", choices=("none", "stub", "batch"), default="none")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N cards")
    ap.add_argument("--mnl-iters", type=int, default=250)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    results = run(args.cards, args.out, args.runs, args.seed, args.warmup,
                  args.scoring_days, slow_brain=args.slow_brain, limit=args.limit,
                  mnl_iters=args.mnl_iters, label=args.label)
    print(json.dumps(_to_native(results["e1_pass"]), indent=2))
    print("E1 method pooled TVD %.5f  MNL pooled TVD %.5f  Delta CI [%.5f, %.5f] eps %.5f" % (
        results["method_arm"]["pooled_tvd"], results["mnl_arm"]["pooled_tvd"],
        results["falsification"]["ci_lo"], results["falsification"]["ci_hi"], e1.EPSILON))
    print("E2 pass=%s spread_pass=%s max_rho=%.4f" % (
        results["e2"]["e2_pass"], results["e2"]["spread_ratios"]["pass"],
        results["e2"]["error_correlation"]["max_rho"]))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
