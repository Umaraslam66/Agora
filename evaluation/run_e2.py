"""CLI runner for E2 (variance preservation, sealed A2.2 / spec D8).

Usage (from the repo root):

    .venv/bin/python -m evaluation.run_e2 \
        --cards runs/gen/cards.json --out runs/e2/m2-first --runs 20 --seed 0

Loads the persona cards (a JSON list of card dicts), loads the seeding
dataset through the promoted adapter (grounding.adapters.psrc — the single
versioned source), scores E2 via evaluation.e2.score_e2, and writes
``results.json`` + ``manifest.json`` into the --out directory.

Manifest style mirrors the E1 runner (evaluation/run_e1.py): flat dict with
adapter/taxonomy versions, card path + sha256, n_runs/seed, timestamp; cards
load from JSONL (one card per line) or a JSON array, like run_e1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from evaluation import e2
from grounding.adapters import psrc


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cards(path) -> List[dict]:
    """Load persona cards from a JSONL (one card per line) or a JSON array —
    the same tolerance as the E1 runner, so every eval reads the same files."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_outputs(out_dir: Path, results: dict, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def build_manifest(args, cards_path: Path, n_cards: int, results: dict) -> dict:
    # mirrors run_e1's manifest style (flat; adapter/taxonomy pins, card
    # sha256, run/seed record, timestamp)
    return {
        "eval": "e2",
        "adapter_version": psrc.ADAPTER_VERSION,
        "taxonomy": "m0-1.0",
        "cards_path": str(cards_path),
        "cards_sha256": _sha256_of_file(cards_path),
        "n_cards": n_cards,
        "n_runs": args.runs,
        "seed": args.seed,
        "namespaces": results["namespaces"],
        "spread_band": list(e2.SPREAD_BAND),
        "correlation_bar": e2.CORRELATION_BAR,
        "dimensions": list(e2.DIMENSIONS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", required=True, help="JSON list of persona cards")
    parser.add_argument("--out", required=True, help="output directory (runs/e2/<name>/)")
    parser.add_argument("--runs", type=int, default=e2.DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=0,
                        help="starting run index for the run{k} CRN namespaces")
    parser.add_argument("--data-dir", default=None, help="raw survey CSV directory")
    parser.add_argument("--cache-dir", default=None, help="adapter cache directory")
    args = parser.parse_args(argv)

    cards_path = Path(args.cards)
    cards = load_cards(cards_path)

    load_kwargs = {}
    if args.data_dir is not None:
        load_kwargs["data_dir"] = args.data_dir
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = args.cache_dir
    dataset = psrc.load_or_build(**load_kwargs)

    results = e2.score_e2(cards, dataset, n_runs=args.runs, seed=args.seed)
    manifest = build_manifest(args, cards_path, len(cards), results)
    out_dir = Path(args.out)
    write_outputs(out_dir, results, manifest)

    verdict = "PASS" if results["e2_pass"] else "FAIL"
    print(f"E2 {verdict}: spread pass={results['spread_ratios']['pass']} "
          f"max_rho={results['error_correlation']['max_rho']:.4f} "
          f"-> {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
