#!/usr/bin/env python3
"""E1 grounding-fidelity harness runner (pre-registration §3 E1, A2.1; M2 D5).

Scores the METHOD arm (persona cards executed by ``agents.card_executor``)
against the sealed full-sample truth, scores the MNL falsification arm
(``evaluation.mnl_arm``, strictly out-of-fold) through the SAME
``evaluation.e1`` path, and runs the paired household-atomic falsification
bootstrap. Writes ``results.json`` (pooled + per-cell TVDs both arms, Delta CI,
pass/fail vs the sealed bars, fallback-card share, drop counts) and
``manifest.json`` (adapter version, map version, card sha256, seeds, timestamps).

Runs are tracked but never git-committed (per the task); this writes under the
``--out`` directory only.

Usage:
  # build deterministic fallback cards for every persona (no GPU) — DEV shakeout
  run_e1.py --build-fallback-cards runs/e1_dev_fallback/cards.jsonl

  # score an arm's cards
  run_e1.py --cards CARDS.jsonl --out runs/e1/<name>/ --runs 20 --bootstrap 500 --seed 20260717
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from evaluation import e1
from evaluation import mnl_arm as mnl
from grounding import seeding
from grounding.adapters import psrc


# ---------------------------------------------------------------------------
# card IO
# ---------------------------------------------------------------------------

def load_cards(path: str) -> List[dict]:
    """Load persona cards from a JSONL (one card per line) or a JSON array."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _native(v):
    """Coerce numpy scalars to JSON-native types for card serialization."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def build_fallback_cards(dataset, persona_index, enriched_trips) -> List[dict]:
    """Deterministic fallback cards for EVERY persona (no GPU) via
    ``grounding.card_validation.fallback_card`` — the DEV fidelity fixture."""
    from grounding.card_validation import fallback_card

    et = enriched_trips.copy()
    et["person_id"] = et["person_id"].astype(str)
    trips_by_person = {pid: grp for pid, grp in et.groupby("person_id")}
    pdays = dataset.person_days.copy()
    pdays["person_id"] = pdays["person_id"].astype(str)
    days_by_person = {pid: grp for pid, grp in pdays.groupby("person_id")}
    empty_days = dataset.person_days.iloc[0:0]
    empty_trips = et.iloc[0:0]

    cards: List[dict] = []
    for r in persona_index.itertuples(index=False):
        persona_id = str(r.persona_id)
        person_id = str(r.person_id)
        skeleton = {f: _native(getattr(r, f)) for f in seeding.SKELETON_FIELDS}
        pdd = days_by_person.get(person_id, empty_days)
        ptr = trips_by_person.get(person_id, empty_trips)
        cards.append(fallback_card(persona_id, skeleton, pdd, ptr))
    return cards


def write_cards(cards: List[dict], path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for card in cards:
            fh.write(json.dumps(card, sort_keys=True))
            fh.write("\n")
    return str(out)


def _fallback_share(cards: List[dict]) -> dict:
    n = len(cards)
    n_fb = sum(1 for c in cards if c.get("provenance", {}).get("card_source") == "fallback")
    n_llm = sum(1 for c in cards if c.get("provenance", {}).get("card_source") == "llm")
    return {"n_cards": n, "n_fallback": n_fb, "n_llm": n_llm,
            "fallback_share": (n_fb / n if n else 0.0)}


# ---------------------------------------------------------------------------
# manifest / IO
# ---------------------------------------------------------------------------

def _date_stamp() -> str:
    try:
        return subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


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


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------

def run(cards_path: str, out_dir: str, n_runs: int, B: int, base_seed: int,
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

    # -- truth (fixed) ----------------------------------------------------
    truth = e1.truth_distributions(dataset, cell_of_household)
    matrices = e1.household_family_matrices(dataset)

    # -- method arm -------------------------------------------------------
    cards = load_cards(cards_path)
    card_personas = {c["persona_id"] for c in cards}
    method = e1.simulate_arm(cards, dataset, persona_cell, day_slots,
                             n_runs=n_runs, namespace_prefix="method_")
    t_method = time.time()

    # -- MNL arm (strictly out-of-fold) ----------------------------------
    arm = mnl.build_mnl_arm(dataset, persona_index, enriched, cell_of_household,
                            day_slots, iters=mnl_iters)
    mnl_dists = e1.ensemble_arm(arm.producer, persona_cell, n_runs=n_runs,
                                namespace_prefix="mnl_")
    t_mnl = time.time()

    # -- scoring ----------------------------------------------------------
    method_pooled = e1.pooled_tvd(method.pooled, truth.pooled)
    mnl_pooled = e1.pooled_tvd(mnl_dists.pooled, truth.pooled)
    method_cells = e1.cell_tvds(method.cells, truth.cells)
    mnl_cells = e1.cell_tvds(mnl_dists.cells, truth.cells)
    boot = e1.paired_bootstrap(method.pooled, mnl_dists.pooled,
                               matrices=matrices, B=B, base_seed=base_seed)

    worst_method_cell = max(method_cells, key=method_cells.get)
    worst_mnl_cell = max(mnl_cells, key=mnl_cells.get)

    pooled_bar_ok = method_pooled <= e1.POOLED_BAR
    cells_bar_ok = all(v <= e1.CELL_BAR for v in method_cells.values())

    results = {
        "label": label,
        "bars": {"pooled": e1.POOLED_BAR, "cell": e1.CELL_BAR, "epsilon": e1.EPSILON},
        "n_runs": n_runs,
        "method_arm": {
            "pooled_tvd": method_pooled,
            "per_family_tvd": e1.per_family_tvd(method.pooled, truth.pooled),
            "cell_tvds": method_cells,
            "worst_cell": worst_method_cell,
            "worst_cell_tvd": method_cells[worst_method_cell],
            "pooled_bar_met": bool(pooled_bar_ok),
            "cells_bar_met": bool(cells_bar_ok),
        },
        "mnl_arm": {
            "pooled_tvd": mnl_pooled,
            "per_family_tvd": e1.per_family_tvd(mnl_dists.pooled, truth.pooled),
            "cell_tvds": mnl_cells,
            "worst_cell": worst_mnl_cell,
            "worst_cell_tvd": mnl_cells[worst_mnl_cell],
        },
        "falsification": boot,
        "pass": {
            "pooled_bar": bool(pooled_bar_ok),
            "cell_bars": bool(cells_bar_ok),
            "beats_or_matches_mnl": bool(boot["pass"]),
            "overall": bool(pooled_bar_ok and cells_bar_ok and boot["pass"]),
        },
        "cards": _fallback_share(cards),
        "coverage": {
            "n_personas": len(persona_index),
            "n_cards": len(cards),
            "n_personas_with_card": len(card_personas & set(day_slots)),
        },
        "drops": drops,
        "timing_seconds": {
            "method_arm": round(t_method - t0, 1),
            "mnl_arm": round(t_mnl - t_method, 1),
            "total": round(time.time() - t0, 1),
        },
    }

    manifest = {
        "adapter_version": psrc.ADAPTER_VERSION,
        "map_version": _safe_map_meta(),
        "taxonomy": "m0-1.0",
        "cards_path": str(cards_path),
        "cards_sha256": _sha256_file(cards_path),
        "n_runs": n_runs,
        "bootstrap_B": B,
        "base_seed": base_seed,
        "seed_derivation": boot["seed_derivation"],
        "mnl_fit_iters": mnl_iters,
        "mnl_diagnostics": arm.diagnostics,
        "timestamp": _date_stamp(),
    }

    (out / "results.json").write_text(json.dumps(_to_native(results), indent=2))
    (out / "manifest.json").write_text(json.dumps(_to_native(manifest), indent=2))
    return results


def _safe_map_meta() -> dict:
    try:
        from grounding import zone_map
        return zone_map.map_metadata()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-fallback-cards", metavar="OUT_JSONL",
                    help="build deterministic fallback cards for every persona and exit")
    ap.add_argument("--cards", metavar="CARDS", help="persona cards JSONL / JSON array")
    ap.add_argument("--out", metavar="DIR", help="output run directory")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=e1.DEFAULT_BASE_SEED)
    ap.add_argument("--mnl-iters", type=int, default=250)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    if args.build_fallback_cards:
        dataset = psrc.load_or_build()
        persona_index = seeding.persona_index(dataset)
        enriched = seeding.enriched_trips(dataset)
        cards = build_fallback_cards(dataset, persona_index, enriched)
        path = write_cards(cards, args.build_fallback_cards)
        print(f"wrote {len(cards)} fallback cards -> {path}")
        return 0

    if not args.cards or not args.out:
        ap.error("--cards and --out are required (unless --build-fallback-cards)")
    results = run(args.cards, args.out, args.runs, args.bootstrap, args.seed,
                  mnl_iters=args.mnl_iters, label=args.label)
    print(json.dumps(_to_native(results["pass"]), indent=2))
    print("method pooled TVD %.5f  MNL pooled TVD %.5f  Delta CI [%.5f, %.5f] eps %.5f" % (
        results["method_arm"]["pooled_tvd"], results["mnl_arm"]["pooled_tvd"],
        results["falsification"]["ci_lo"], results["falsification"]["ci_hi"], e1.EPSILON))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
