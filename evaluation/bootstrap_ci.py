#!/usr/bin/env python3
"""Percentile bootstrap confidence intervals for per-segment (per-group) means.

Generic statistical utility: given a 1-D sample of per-record values,
bootstrap a percentile CI on the sample mean. Also provides a per-segment
sweep helper for the common case of several segments/groups scored under one
shared, reproducible RNG stream.

RESAMPLING SCHEME (preserved exactly):
    For a sample of n values, each of `n_boot` bootstrap draws resamples n
    values with replacement — one draw at a time via `rng.randrange(n)` —
    and takes their mean. This is deliberately element-wise rather than a
    single vectorized `rng.choices(xs, k=n)` call: it reproduces the exact
    RNG draw sequence of the original tool for a given seed, so seeded runs
    are bit-for-bit reproducible across re-implementations.

CI CONSTRUCTION (preserved exactly):
    The `n_boot` bootstrap means are sorted, and a percentile p is read off
    by NEAREST-RANK index `round(p / 100 * (n_boot - 1))` into the sorted
    array (not linear interpolation, e.g. not `numpy.percentile`'s default).
    The default interval is the 95% CI via the [2.5, 97.5] percentiles; the
    50th-percentile (bootstrap median) is also returned alongside the point
    estimate as a symmetry/robustness check.

SEED HANDLING (preserved exactly):
    A single `random.Random(seed)` instance is the unit of reproducibility.
    When bootstrapping several segments under one configuration (a "sweep"),
    construct ONE `random.Random` and pass the SAME object across all
    segments in that sweep, in a fixed (e.g. sorted) order — this reproduces
    the original tool's convention of sharing one continuing RNG stream
    across segments so the whole sweep is deterministic end to end, not just
    seed-identical per segment in isolation. Start a fresh `random.Random`
    (new seed or new instance) only for an independent sweep (e.g. a new
    evaluation configuration).

I/O: functions here take plain arrays / mappings of arrays, not any
particular file format — `bootstrap_ci_per_segment` accepts any mapping from
segment label to a 1-D array-like of values (e.g. the output of
`df.groupby("segment")["value"].apply(list).to_dict()` for a pandas
DataFrame). A small JSONL-based CLI is included for convenience.

Usage (CLI):
    python3 bootstrap_ci.py DATA.jsonl --n-boot 10000 --seed 1234

    where DATA.jsonl has one record per line: {"segment": <label>, "value": <float>}
"""
import random
from typing import Dict, Mapping, Sequence, Tuple

DEFAULT_LO = 2.5
DEFAULT_HI = 97.5


def bootstrap_ci(
    xs: Sequence[float],
    n_boot: int,
    rng: random.Random,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of the mean of `xs`.

    Returns (lo_pct, median, hi_pct) of the bootstrap distribution of the
    mean, using nearest-rank percentiles over `n_boot` element-wise
    resamples-with-replacement of `xs`.

    `rng` is a `random.Random` instance. Callers computing several CIs in one
    sweep (e.g. one per segment, at a fixed configuration) should share a
    single `rng` across those calls — see module docstring, "SEED HANDLING".
    """
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        means.append(s / n)
    means.sort()

    def pct(p: float) -> float:
        idx = min(n_boot - 1, max(0, int(round(p / 100.0 * (n_boot - 1)))))
        return means[idx]

    return (pct(lo), pct(50), pct(hi))


def bootstrap_ci_per_segment(
    values_by_segment: Mapping[str, Sequence[float]],
    n_boot: int,
    seed: int,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
) -> Dict[str, dict]:
    """Bootstrap CI for the mean of each segment's values.

    One `random.Random(seed)` is created and shared across ALL segments,
    visited in sorted-name order, reproducing the original tool's "same
    resample indices across segments" convention: the whole sweep is
    reproducible given (`values_by_segment` iteration content, `seed`,
    `n_boot`), not merely each segment in isolation.
    """
    rng = random.Random(seed)
    out: Dict[str, dict] = {}
    for name in sorted(values_by_segment):
        xs = list(values_by_segment[name])
        point = sum(xs) / len(xs) if xs else 0.0
        lo_v, med, hi_v = bootstrap_ci(xs, n_boot, rng, lo, hi)
        out[name] = {
            "n": len(xs),
            "mean": point,
            "ci_lo": lo_v,
            "ci_median": med,
            "ci_hi": hi_v,
            "ci_width": hi_v - lo_v,
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_records(path: str):
    import json

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main(argv=None) -> int:
    import argparse
    import json
    from collections import defaultdict

    ap = argparse.ArgumentParser(
        description="Percentile bootstrap CIs on the per-segment mean of a "
        "generic value column, read from a JSONL file of "
        "{'segment': <label>, 'value': <float>} records."
    )
    ap.add_argument(
        "data", help="JSONL file, one record per line: {'segment': ..., 'value': ...}"
    )
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lo", type=float, default=DEFAULT_LO)
    ap.add_argument("--hi", type=float, default=DEFAULT_HI)
    args = ap.parse_args(argv)

    records = _load_records(args.data)
    print(f"[bootstrap_ci] {len(records)} records from {args.data}")

    by_segment = defaultdict(list)
    for r in records:
        by_segment[r["segment"]].append(float(r["value"]))

    result = bootstrap_ci_per_segment(by_segment, args.n_boot, args.seed, args.lo, args.hi)

    print(f"{'segment':16} {'n':>6} {'mean':>10}  {'CI':>22}  {'width':>9}")
    for name, s in result.items():
        print(
            f"{name:16} {s['n']:>6} {s['mean']:>10.4f}  "
            f"[{s['ci_lo']:>9.4f},{s['ci_hi']:>9.4f}]  {s['ci_width']:>9.4f}"
        )
    print()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
