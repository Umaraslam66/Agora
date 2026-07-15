"""CLI runner for the E5(i) masked-vs-unmasked contamination probe
(sealed A2.3(i) / spec D9).

Usage (from the repo root):

    .venv/bin/python -m evaluation.run_e5 \
        --masked-cards runs/gen/cards.json \
        --unmasked-cards runs/gen_unmasked/cards.json \
        --out runs/e5/m2-first --runs 20 --seed 0

Takes the two card files (masked arm, unmasked arm — the unmasked cards are
regenerated from the unmasked prompt file on the GPU cluster, later), executes
both populations under the SAME ``run{k}`` CRN namespaces (the paired seeds
the 25% threshold was sealed with), scores both once against the same fixed
full-sample reference, and writes ``results.json`` + ``manifest.json`` with
the contamination-flag verdict into the --out directory.

Manifest style mirrors the E1 runner (evaluation/run_e1.py): flat dict with
adapter/taxonomy versions, card paths + sha256s, n_runs/seed, timestamp;
cards load from JSONL (one card per line) or a JSON array, like run_e1.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation import contamination
from evaluation.run_e2 import load_cards, write_outputs
from grounding.adapters import psrc


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(args, masked_path: Path, unmasked_path: Path, results: dict) -> dict:
    # mirrors run_e1's manifest style (flat; adapter/taxonomy pins, input
    # sha256s, run/seed record, timestamp)
    return {
        "eval": "e5i",
        "adapter_version": psrc.ADAPTER_VERSION,
        "taxonomy": "m0-1.0",
        "masked_cards_path": str(masked_path),
        "masked_cards_sha256": _sha256_of_file(masked_path),
        "unmasked_cards_path": str(unmasked_path),
        "unmasked_cards_sha256": _sha256_of_file(unmasked_path),
        "n_personas": results["n_personas"],
        "n_runs": args.runs,
        "seed": args.seed,
        "namespaces": results["namespaces"],
        "threshold": contamination.E5_FLAG_THRESHOLD,
        "families": list(contamination.FAMILIES),
        "pairing": "identical populations; SAME CRN namespaces both arms; "
                   "one fixed full-sample reference, no resampling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-cards", required=True, help="masked-arm card JSON list")
    parser.add_argument("--unmasked-cards", required=True, help="unmasked-arm card JSON list")
    parser.add_argument("--out", required=True, help="output directory (runs/e5/<name>/)")
    parser.add_argument("--runs", type=int, default=contamination.DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=0,
                        help="starting run index for the run{k} CRN namespaces")
    parser.add_argument("--data-dir", default=None, help="raw survey CSV directory")
    parser.add_argument("--cache-dir", default=None, help="adapter cache directory")
    args = parser.parse_args(argv)

    masked_path = Path(args.masked_cards)
    unmasked_path = Path(args.unmasked_cards)
    masked_cards = load_cards(masked_path)
    unmasked_cards = load_cards(unmasked_path)

    load_kwargs = {}
    if args.data_dir is not None:
        load_kwargs["data_dir"] = args.data_dir
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = args.cache_dir
    dataset = psrc.load_or_build(**load_kwargs)

    results = contamination.score_e5(
        masked_cards, unmasked_cards, dataset, n_runs=args.runs, seed=args.seed
    )
    manifest = build_manifest(args, masked_path, unmasked_path, results)
    out_dir = Path(args.out)
    write_outputs(out_dir, results, manifest)

    probe = results["probe"]
    verdict = "CONTAMINATION FLAG" if results["contamination_flag"] else "no flag"
    print(f"E5(i) {verdict}: relative improvement "
          f"{probe['relative_improvement']:.4f} (threshold {probe['threshold']}) "
          f"-> {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
