#!/usr/bin/env python3
"""A4.1 E7 ordinary-day ESS scoring driver (harness side; NON-BLIND).

Scores every E7 arm at the INDIVIDUAL level against the persona's own real
diary days, in equivalent-sample-size units, per the adapter pinned in
``runs/e7_tiers/e7_manifest.json -> ess_loss_adapter`` (pinned 2026-07-20,
before any ESS quantity was computed). The estimator is
``evaluation.ess`` (Gao, Han & Liang 2026, exactly as published); this
driver supplies the pinned per-persona loss and the pinned flexible
baseline (the E1 MNL arm's day-structure + mode-choice components, refit
per training block) and owns nothing else.

Loss (pinned): per persona, unweighted mean over the persona's DEFINED
families (observed trip mass > 0; zero-trip-only personas carry the trips
family alone) of TVD(predicted, observed) over the three frozen E1
families. Predicted = normalize(mean over M ensemble members of the
persona's unnormalized masses) — ratio-of-means; a defined family whose
predicted mass is zero scores TVD = 1.0 (the predictor says "never
travels", the person travels). Both arms flow through the SAME
RealizedDay Monte-Carlo path with matched M, so MC-TVD inflation is
symmetric and first-order cancels in the difference loss.

Targets (pinned): target-ALL (primary; every persona, all E1 day slots;
identical across arms) and target-HELDOUT (secondary; the multi-day
personas, slots excluding the manifest-pinned T4 ``selected_daynum`` —
a true generalization read for T4/T4-noclaims, in-evidence for T5 and the
deployed/template arms, flagged as such).

NON-BLIND: ordinary-day scoring only. Imports nothing from
``evaluation.truth``; touches nothing under ``runs/bt1`` / ``runs/bt2``.

Usage:
  .venv/bin/python -m evaluation.run_e7_ess --out runs/e7_ess [--runs 20]
      [--smoke N]   # dev shakeout on a persona subsample; never a record
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from agents import logit_chooser
from agents.card_executor import execute_days
from evaluation import e1, ess
from evaluation import golden_pairs as gp
from evaluation.e1 import (
    MODES,
    TIME_BANDS,
    _trips_bin_index,
    normalize,
    tvd,
)
from evaluation.mnl_arm import _softmax_probs
from evaluation.run_e1 import _date_stamp, _sha256_file, _to_native, load_cards
from grounding import seeding
from grounding.adapters import psrc
from world.crn import pick_weighted

# ---------------------------------------------------------------------------
# Pinned protocol constants (mirror runs/e7_tiers/e7_manifest.json)
# ---------------------------------------------------------------------------

ESS_GRID: Tuple[int, ...] = (10, 20, 50, 100, 250, 500, 1000, 2500, 5000)
ALPHA = 0.05
DEFAULT_RUNS = 20
BLOCK_NAMESPACE = "e7ess_v1"
LOGIT_ITERS = 250
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260720

TIER_ARMS = ("T1", "T2", "T3", "T4", "T4_noclaims", "T4_nofidelity", "T5")
ARM_CARDS = {
    "T1": "runs/e7_tiers/T1/cards_T1.jsonl",
    "T2": "runs/e7_tiers/T2/cards_T2.jsonl",
    "T3": "runs/e7_tiers/T3/cards_T3.jsonl",
    "T4": "runs/e7_tiers/T4/cards_T4.jsonl",
    "T4_noclaims": "runs/e7_tiers/T4_noclaims/cards_T4_noclaims.jsonl",
    "T4_nofidelity": "runs/e7_tiers/T4_nofidelity/cards_T4_nofidelity.jsonl",
    "T5": "runs/e7_tiers/T5/cards_T5.jsonl",
    "m2_deployed": "data/cards/cards_m2_masked_r2b.jsonl",
    "template": "runs/e1_dev_fallback/cards.jsonl",
}
T4_CONTEXT = "runs/e7_tiers/T4/tier_context.json"

_MODE_IDX = {m: j for j, m in enumerate(MODES)}
_BAND_IDX = {b: j for j, b in enumerate(TIME_BANDS)}
_POOLED = "__pooled__"


# ---------------------------------------------------------------------------
# Per-persona static state (shared, read-only)
# ---------------------------------------------------------------------------

class PersonaStatic:
    """Everything per-persona the driver needs, precomputed once.

    ``day_sigs``   [(daynum, w_day, purpose-sequence)] — baseline day fit
    ``trip_pbw``   [(purpose, band, w_trip)] — baseline band fit
    ``obs_all/hold`` observed family masses per target (hold None unless
    multi-day)
    """

    __slots__ = (
        "persona_id", "person_id", "household_id", "cell", "slots",
        "day_sigs", "trip_pbw", "golden_records", "parsed", "feasible",
        "obs_all", "obs_hold", "heldout_days",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _observed_masses(
    day_rows: Sequence[Tuple[int, float, int]],
    trip_rows: Sequence[Tuple[int, str, str, float]],
    exclude_day: Optional[int],
) -> Optional[List[np.ndarray]]:
    """Observed family MASSES for one persona.

    ``day_rows``: (daynum, w_day, n_collapsed) per observed weekday;
    ``trip_rows``: (daynum, mode, band, w_trip) per observed weekday trip.
    ``exclude_day`` drops that daynum (the held-out target); returns None
    when no day survives the exclusion.
    """
    m = [np.zeros(9), np.zeros(len(MODES)), np.zeros(len(TIME_BANDS))]
    any_day = False
    for daynum, w_day, n_coll in day_rows:
        if exclude_day is not None and daynum == exclude_day:
            continue
        any_day = True
        m[0][_trips_bin_index(int(n_coll))] += float(w_day)
    for daynum, mode, band, w_trip in trip_rows:
        if exclude_day is not None and daynum == exclude_day:
            continue
        m[1][_MODE_IDX[mode]] += float(w_trip)
        m[2][_BAND_IDX[band]] += float(w_trip)
    return m if any_day else None


def build_static(
    dataset, persona_index, enriched, limit: Optional[int] = None
) -> Tuple[List[str], Dict[str, PersonaStatic]]:
    """Precompute the shared per-persona state for every arm and the baseline."""
    cell_of_household, _drops = e1.repinned_cell_of_household(dataset)
    t4_ctx = json.loads(Path(T4_CONTEXT).read_text())

    pi = persona_index.copy()
    for c in ("persona_id", "person_id", "household_id"):
        pi[c] = pi[c].astype(str)

    et = enriched.copy()
    et["person_id"] = et["person_id"].astype(str)
    trips_by_person: Dict[str, list] = defaultdict(list)
    ordered = et.sort_values(["person_id", "daynum", "tripnum"])
    for pid, dn, mode, band, w, purpose in zip(
        ordered["person_id"], ordered["daynum"], ordered["mode"],
        ordered["band"], ordered["w_trip"], ordered["purpose"],
    ):
        trips_by_person[str(pid)].append(
            (int(dn), str(mode), str(band), float(w), str(purpose))
        )

    pdt = dataset.person_days
    days_by_person: Dict[str, list] = defaultdict(list)
    for pid, dn, w, nc in zip(
        pdt.person_id.astype(str), pdt.daynum, pdt.w_day, pdt.n_collapsed
    ):
        days_by_person[str(pid)].append((int(dn), float(w), int(nc)))

    skel_fields = ["age", "income_class", "employed", "can_drive", "household_cars"]
    ids: List[str] = []
    static: Dict[str, PersonaStatic] = {}
    for r in pi.itertuples(index=False):
        persona_id = str(r.persona_id)
        person_id = str(r.person_id)
        day_rows = sorted(days_by_person.get(person_id, []))
        if not day_rows:
            continue  # no observed weekday slot -> not an observation
        raw = trips_by_person.get(person_id, [])
        trip_rows = [(dn, mode, band, w) for dn, mode, band, w, _pu in raw]

        sig_by_day: Dict[int, list] = defaultdict(list)
        for dn, _mode, _band, _w, purpose in raw:
            sig_by_day[dn].append(purpose)
        day_sigs = [
            (dn, w_day, tuple(sig_by_day.get(dn, ())))
            for dn, w_day, _nc in day_rows
        ]
        trip_pbw = [(pu, band, w) for _dn, _mode, band, w, pu in raw]

        skel = {f: getattr(r, f) for f in skel_fields}
        modes = [t[1] for t in raw]
        purposes = [t[4] for t in raw]
        ev = gp.person_evidence(persona_id, skel, modes, purposes)
        records = []
        for _dn, mode, _band, _w, purpose in raw:
            rec = gp.training_record(ev, purpose, mode, split="train")
            if rec is not None:
                records.append(rec)
        parsed = logit_chooser.parse_prompt(gp.build_prompt(ev))

        sel = t4_ctx.get(persona_id, {}).get("selected_daynum")
        multi = len(day_rows) > 1
        heldout_days: Optional[Set[int]] = (
            {dn for dn, _w, _nc in day_rows if dn != sel}
            if (multi and sel is not None)
            else None
        )
        if heldout_days is not None and not heldout_days:
            heldout_days = None  # selected day was the only day (defensive)

        st = PersonaStatic(
            persona_id=persona_id,
            person_id=person_id,
            household_id=str(r.household_id),
            cell=cell_of_household.get(str(r.household_id)),
            slots=[(dn, w) for dn, w, _nc in day_rows],
            day_sigs=day_sigs,
            trip_pbw=trip_pbw,
            golden_records=records,
            parsed=parsed,
            feasible=ev.feasible,
            obs_all=_observed_masses(day_rows, trip_rows, None),
            obs_hold=(
                _observed_masses(day_rows, trip_rows, sel)
                if heldout_days is not None
                else None
            ),
            heldout_days=heldout_days,
        )
        ids.append(persona_id)
        static[persona_id] = st
    ids.sort()
    if limit:
        ids = ids[:limit]
        static = {p: static[p] for p in ids}
    return ids, static


# ---------------------------------------------------------------------------
# The pinned per-persona loss
# ---------------------------------------------------------------------------

def persona_loss(
    pred_mass: Sequence[np.ndarray], obs_mass: Sequence[np.ndarray]
) -> float:
    """Mean over DEFINED families of TVD(normalize(pred), normalize(obs)).

    trips/day is always defined (>=1 observed day); mode/band are defined
    iff the observed trip mass is positive; a defined family with zero
    predicted mass scores TVD = 1.0 (pinned corner case)."""
    vals: List[float] = []
    for j in range(3):
        obs = obs_mass[j]
        if obs.sum() <= 0:
            continue
        pred = pred_mass[j]
        if pred.sum() <= 0:
            vals.append(1.0)
        else:
            vals.append(tvd(normalize(pred), normalize(obs)))
    return float(np.mean(vals))


def _zero_masses(ids: Sequence[str]) -> Dict[str, List[np.ndarray]]:
    return {
        pid: [np.zeros(9), np.zeros(len(MODES)), np.zeros(len(TIME_BANDS))]
        for pid in ids
    }


def accumulate_masses(
    days_by_persona: Mapping[str, Sequence],
    static: Mapping[str, PersonaStatic],
    acc_all: Dict[str, List[np.ndarray]],
    acc_hold: Optional[Dict[str, List[np.ndarray]]] = None,
) -> None:
    """Add one ensemble member's unnormalized masses (both targets) in place."""
    for pid, days in days_by_persona.items():
        st = static[pid]
        a = acc_all[pid]
        h = (
            acc_hold.get(pid)
            if (acc_hold is not None and st.heldout_days is not None)
            else None
        )
        for day in days:
            w = day.day_weight
            b = _trips_bin_index(len(day.trips))
            in_hold = h is not None and day.day_index in st.heldout_days
            a[0][b] += w
            if in_hold:
                h[0][b] += w
            for t in day.trips:
                mj = _MODE_IDX[t.mode]
                cj = _BAND_IDX[t.depart_band]
                a[1][mj] += w
                a[2][cj] += w
                if in_hold:
                    h[1][mj] += w
                    h[2][cj] += w


def score_producer(
    producer,
    ids: Sequence[str],
    static: Mapping[str, PersonaStatic],
    n_runs: int,
    ns_prefix: str,
) -> Dict[str, Dict[str, float]]:
    """Per-persona losses for one arm: {'all': {pid: loss}, 'heldout': {...}}.

    ``producer(namespace)`` -> persona_id -> [RealizedDay]. Ensemble masses
    are summed across members then normalized once (ratio-of-means, pinned).
    """
    hold_ids = [p for p in ids if static[p].heldout_days is not None]
    acc_all = _zero_masses(ids)
    acc_hold = _zero_masses(hold_ids)
    for k in range(n_runs):
        days = producer(f"{ns_prefix}run{k}")
        accumulate_masses(days, static, acc_all, acc_hold)
    out_all = {pid: persona_loss(acc_all[pid], static[pid].obs_all) for pid in ids}
    out_hold = {
        pid: persona_loss(acc_hold[pid], static[pid].obs_hold) for pid in hold_ids
    }
    return {"all": out_all, "heldout": out_hold}


# ---------------------------------------------------------------------------
# The pinned flexible baseline: E1 MNL components refit on a training block
# ---------------------------------------------------------------------------

class BlockModel:
    """The E1 MNL arm's two components fitted on ONE training block.

    Draw semantics and CRN key structure are exactly the E1 arm's
    (``evaluation.mnl_arm.MNLArm.make_days``); only the training sample
    differs (the block instead of four folds)."""

    __slots__ = ("sig_dist", "band_dist", "band_global", "coef")

    def __init__(self, sig_dist, band_dist, band_global, coef):
        self.sig_dist = sig_dist      # {cell/__pooled__: (sigs, weights)}
        self.band_dist = band_dist    # {(cell|__pooled__, purpose): (bands, weights)}
        self.band_global = band_global  # (bands, weights); uniform if block empty
        self.coef = coef              # logit coefficients (possibly {})


def _to_lists(counter: Counter) -> Tuple[list, list]:
    items = sorted(counter.items(), key=lambda kv: (repr(kv[0]),))
    return [k for k, _ in items], [float(v) for _, v in items]


def fit_block_model(
    train_ids: Sequence[str],
    static: Mapping[str, PersonaStatic],
    workdir: Path,
    tag: str,
) -> BlockModel:
    """Refit the day-signature, depart-band, and logit components on the block.

    Mirrors ``evaluation.mnl_arm`` fitting semantics with the block as the
    entire training sample (within-block cell -> pooled -> global fallback;
    a block with zero trip mass falls back to uniform bands and an empty
    coefficient set = uniform-over-feasible softmax, per the pinned adapter).
    """
    sig_cell: Dict[str, Counter] = defaultdict(Counter)
    sig_pool: Counter = Counter()
    band_cp: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    band_p: Dict[str, Counter] = defaultdict(Counter)
    band_glob: Counter = Counter()
    records: List[dict] = []
    for pid in train_ids:
        st = static[pid]
        for _dn, w_day, sig in st.day_sigs:
            sig_pool[sig] += w_day
            if st.cell is not None:
                sig_cell[st.cell][sig] += w_day
        for purpose, band, w in st.trip_pbw:
            band_cp[((st.cell if st.cell is not None else _POOLED), purpose)][band] += w
            band_p[purpose][band] += w
            band_glob[band] += w
        records.extend(st.golden_records)

    sig_dist = {cell: _to_lists(c) for cell, c in sig_cell.items()}
    sig_dist[_POOLED] = _to_lists(sig_pool)

    band_dist: Dict[Tuple[str, str], Tuple[list, list]] = {}
    for (cell, purpose), c in band_cp.items():
        band_dist[(cell, purpose)] = _to_lists(c)
    for purpose, c in band_p.items():
        band_dist[(_POOLED, purpose)] = _to_lists(c)
    band_global = (
        _to_lists(band_glob)
        if band_glob
        else (list(TIME_BANDS), [1.0] * len(TIME_BANDS))
    )

    coef: dict = {}
    if records:
        import contextlib
        import io

        pairs_path = workdir / f"pairs_{tag}.jsonl"
        coef_path = workdir / f"coef_{tag}.json"
        with pairs_path.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            logit_chooser.fit(
                str(pairs_path), str(coef_path), iters=LOGIT_ITERS, lr=0.5, l2=1e-3
            )
        coef = logit_chooser.load_coef(str(coef_path))
        pairs_path.unlink()
        coef_path.unlink()
    return BlockModel(sig_dist, band_dist, band_global, coef)


def block_make_days(
    model: BlockModel,
    st: PersonaStatic,
    namespace: str,
):
    """Simulate one persona's slots under the block model — the E1 arm's
    draw semantics and CRN key structure verbatim."""
    from agents.card_executor import RealizedDay, RealizedTrip

    sigs, weights = model.sig_dist.get(st.cell) or ([], [])
    if not sigs:
        sigs, weights = model.sig_dist[_POOLED]
    order, probs = _softmax_probs(st.parsed, st.feasible, model.coef)
    days = []
    pid = st.persona_id
    for day_index, w in st.slots:
        sig = pick_weighted(f"{namespace}:{pid}:{day_index}:skeleton", sigs, weights)
        trips = []
        for i, purpose in enumerate(sig):
            key = f"{namespace}:{pid}:{day_index}:trip{i}:band"
            band = None
            for k in ((st.cell, purpose), (_POOLED, purpose)):
                entry = model.band_dist.get(k)
                if entry is not None and entry[0]:
                    band = pick_weighted(key, entry[0], entry[1])
                    break
            if band is None:
                bands, bweights = model.band_global
                band = pick_weighted(key, bands, bweights)
            mode = pick_weighted(
                f"{namespace}:{pid}:{day_index}:trip{i}:mode", order, probs
            )
            trips.append(RealizedTrip(purpose, mode, band, None))
        days.append(RealizedDay(int(day_index), float(w), trips))
    return days


# ---------------------------------------------------------------------------
# Block-out loss rows (parallel over blocks)
# ---------------------------------------------------------------------------

_G: dict = {}  # fork-shared read-only worker state


def _init_worker_state(ids, static, n_runs, tag, workdir):
    _G["ids"] = list(ids)
    _G["static"] = static
    _G["n_runs"] = int(n_runs)
    _G["tag"] = str(tag)
    _G["workdir"] = str(workdir)
    _G["idx_of"] = {pid: i for i, pid in enumerate(ids)}


def _block_worker(job) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Fit one block, simulate its complement, return both targets' loss rows."""
    size, bi, train_idx = job
    ids: List[str] = _G["ids"]
    static: Mapping[str, PersonaStatic] = _G["static"]
    n_runs: int = _G["n_runs"]
    tag: str = _G["tag"]
    n = len(ids)

    train_set = set(int(i) for i in train_idx)
    train_ids = [ids[i] for i in sorted(train_set)]
    test_ids = [ids[i] for i in range(n) if i not in train_set]

    model = fit_block_model(
        train_ids, static, Path(_G["workdir"]), f"{tag}_N{size}_b{bi}"
    )

    hold_ids = [p for p in test_ids if static[p].heldout_days is not None]
    acc_all = _zero_masses(test_ids)
    acc_hold = _zero_masses(hold_ids)
    for k in range(n_runs):
        ns = f"e7ess_base_{tag}_N{size}_b{bi}_run{k}"
        days = {pid: block_make_days(model, static[pid], ns) for pid in test_ids}
        accumulate_masses(days, static, acc_all, acc_hold)

    row_all = np.full(n, np.nan)
    row_hold = np.full(n, np.nan)
    idx_of = _G["idx_of"]
    for pid in test_ids:
        row_all[idx_of[pid]] = persona_loss(acc_all[pid], static[pid].obs_all)
    for pid in hold_ids:
        row_hold[idx_of[pid]] = persona_loss(acc_hold[pid], static[pid].obs_hold)
    return size, bi, row_all, row_hold


class BaselineCache:
    """Lazily computes and caches, per candidate size, the baseline loss
    matrices (both targets) and the block index sets — shared across arms."""

    def __init__(self, ids, static, n_runs, tag, workdir, processes=None):
        self.ids = list(ids)
        self.static = static
        self.n_runs = n_runs
        self.tag = tag
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.processes = processes or max(1, (os.cpu_count() or 2) - 1)
        self.hh_ids = [static[p].household_id for p in self.ids]
        self._cache: Dict[int, dict] = {}

    def matrices(self, size: int) -> dict:
        if size in self._cache:
            return self._cache[size]
        blocks = ess.household_blocks(
            self.hh_ids, size, namespace=f"{BLOCK_NAMESPACE}_{self.tag}"
        )
        jobs = [(size, bi, blocks[bi]) for bi in range(len(blocks))]
        t0 = time.time()
        _init_worker_state(self.ids, self.static, self.n_runs, self.tag, self.workdir)
        n = len(self.ids)
        loss_all = np.full((len(blocks), n), np.nan)
        loss_hold = np.full((len(blocks), n), np.nan)
        if self.processes > 1 and len(jobs) > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(self.processes) as pool:
                for sz, bi, row_all, row_hold in pool.imap_unordered(
                    _block_worker, jobs, chunksize=max(1, len(jobs) // (self.processes * 8))
                ):
                    loss_all[bi] = row_all
                    loss_hold[bi] = row_hold
        else:
            for job in jobs:
                _sz, bi, row_all, row_hold = _block_worker(job)
                loss_all[bi] = row_all
                loss_hold[bi] = row_hold
        # NaN-out the training block entries explicitly (they already are,
        # by construction: workers only fill test columns).
        entry = {
            "blocks": blocks,
            "all": loss_all,
            "heldout": loss_hold,
            "seconds": round(time.time() - t0, 1),
            "n_blocks": len(blocks),
        }
        self._cache[size] = entry
        print(
            f"[base:{self.tag}] size {size}: {len(blocks)} blocks in "
            f"{entry['seconds']}s", flush=True,
        )
        return entry


# ---------------------------------------------------------------------------
# ESS per arm (sequential estimator on the difference loss)
# ---------------------------------------------------------------------------

def ess_for_arm(
    arm_loss: Mapping[str, float],
    target: str,
    cache: BaselineCache,
    sizes: Sequence[int],
    alpha: float = ALPHA,
) -> dict:
    """Run the pinned sequential ESS estimator for one arm on one target.

    Personas without a loss on this target (e.g. single-day personas on
    target-HELDOUT) stay NaN in the arm vector; they are excluded from the
    difference-loss matrix columns by masking to the target's persona set —
    the observation set of the HELDOUT analysis is rebuilt by the caller,
    so here every id in ``cache.ids`` must carry a loss."""
    arm_vec = np.array([arm_loss[pid] for pid in cache.ids], dtype=float)

    def diff_loss_of(size: int):
        entry = cache.matrices(size)
        return entry[target] - arm_vec[None, :], entry["blocks"]

    res = ess.ess_sequential(sizes, diff_loss_of, alpha=alpha)
    out = {
        "plugin": res.plugin,
        "lower_bound": res.lower_bound,
        "exceeds_grid": res.exceeds_grid,
        "alpha": res.alpha,
        "per_size": res.per_size,
    }
    return out


# ---------------------------------------------------------------------------
# Aggregate diagnostic (sealed A2.1 pooled TVD per arm)
# ---------------------------------------------------------------------------

def aggregate_pooled_tvd(cards, dataset, persona_cell, day_slots, n_runs, prefix):
    arm = e1.simulate_arm(
        cards, dataset, persona_cell, day_slots, n_runs=n_runs,
        namespace_prefix=prefix,
    )
    truth = e1.truth_distributions(dataset)
    return {
        "pooled_tvd": e1.pooled_tvd(arm.pooled, truth.pooled),
        "per_family_tvd": e1.per_family_tvd(arm.pooled, truth.pooled),
    }


# ---------------------------------------------------------------------------
# Template-vs-LLM head-to-head (paired, household-atomic bootstrap)
# ---------------------------------------------------------------------------

def headtohead(
    llm_loss: Mapping[str, float],
    template_loss: Mapping[str, float],
    deployed_source: Mapping[str, str],
    static: Mapping[str, PersonaStatic],
    target_ids: Sequence[str],
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired per-persona contrast on personas whose DEPLOYED card is LLM:
    delta_i = loss(llm card) - loss(template card); household-atomic
    bootstrap CI on the mean delta. Group means reported alongside."""
    paired = [
        p for p in target_ids
        if deployed_source.get(p) == "llm" and p in llm_loss and p in template_loss
    ]
    fallback_ids = [
        p for p in target_ids
        if deployed_source.get(p) == "fallback" and p in template_loss
    ]
    if not paired:
        return {"n_paired": 0, "note": "no paired personas on this target"}
    delta = np.array([llm_loss[p] - template_loss[p] for p in paired])
    hh = [static[p].household_id for p in paired]
    hh_index: Dict[str, List[int]] = defaultdict(list)
    for i, h in enumerate(hh):
        hh_index[h].append(i)
    households = sorted(hh_index)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for b in range(B):
        draw = rng.integers(0, len(households), size=len(households))
        idx = np.concatenate([hh_index[households[j]] for j in draw])
        means[b] = delta[idx].mean()
    return {
        "n_paired": len(paired),
        "n_fallback_personas": len(fallback_ids),
        "mean_delta_llm_minus_template": float(delta.mean()),
        "delta_ci95": [float(np.percentile(means, 2.5)),
                       float(np.percentile(means, 97.5))],
        "mean_loss_llm_cards_on_llm_personas": float(
            np.mean([llm_loss[p] for p in paired])
        ),
        "mean_loss_template_cards_on_llm_personas": float(
            np.mean([template_loss[p] for p in paired])
        ),
        "mean_loss_template_cards_on_fallback_personas": float(
            np.mean([template_loss[p] for p in fallback_ids])
        ) if fallback_ids else None,
        "bootstrap": {"B": B, "seed": seed, "unit": "household"},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _summary(loss: Mapping[str, float]) -> dict:
    v = np.array(list(loss.values()), dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


def run(out_dir: str, n_runs: int = DEFAULT_RUNS, smoke: Optional[int] = None,
        processes: Optional[int] = None, grid: Sequence[int] = ESS_GRID) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="e7ess_fit_"))

    dataset = psrc.load_or_build()
    persona_index = seeding.persona_index(dataset)
    enriched = seeding.enriched_trips(dataset)
    ids, static = build_static(dataset, persona_index, enriched, limit=smoke)
    print(f"[static] {len(ids)} observations "
          f"({sum(1 for p in ids if static[p].heldout_days is not None)} multi-day) "
          f"in {time.time()-t0:.1f}s", flush=True)

    persona_of_person = e1.persona_of_person_map(persona_index)
    cell_of_household, _ = e1.repinned_cell_of_household(dataset)
    persona_cell = e1.persona_cell_map(persona_index, cell_of_household)
    day_slots_full = e1.day_slots_by_persona(dataset, persona_of_person)
    day_slots = {p: day_slots_full[p] for p in ids}

    # -- card arms ---------------------------------------------------------
    arm_losses: Dict[str, Dict[str, Dict[str, float]]] = {}
    arm_meta: Dict[str, dict] = {}
    aggregate: Dict[str, dict] = {}
    for arm, path in ARM_CARDS.items():
        t1 = time.time()
        cards = [c for c in load_cards(path) if c["persona_id"] in static]
        src = Counter(
            c.get("provenance", {}).get("card_source", "unknown") for c in cards
        )

        def producer(namespace, _cards=cards):
            return execute_days(_cards, day_slots, namespace, update_habits=False)

        arm_losses[arm] = score_producer(
            producer, ids, static, n_runs, f"e7ess_{arm}_"
        )
        aggregate[arm] = aggregate_pooled_tvd(
            cards, dataset, persona_cell, day_slots, n_runs, f"e7ess_agg_{arm}_"
        )
        arm_meta[arm] = {
            "cards_path": path,
            "cards_sha256": _sha256_file(path),
            "n_cards": len(cards),
            "card_source_counts": dict(src),
            "seconds": round(time.time() - t1, 1),
        }
        print(f"[arm {arm}] mean loss ALL "
              f"{_summary(arm_losses[arm]['all'])['mean']:.4f} "
              f"({arm_meta[arm]['seconds']}s)", flush=True)

    # -- baseline + ESS: target-ALL (primary, full population) -------------
    cache_all = BaselineCache(ids, static, n_runs, "main", workdir,
                              processes=processes)
    n = len(ids)
    sizes_all = [s for s in grid if n // s >= 2]
    ess_results: Dict[str, dict] = {}
    for arm in ARM_CARDS:
        ess_results[arm] = ess_for_arm(
            arm_losses[arm]["all"], "all", cache_all, sizes_all
        )
        print(f"[ESS all] {arm}: plugin {ess_results[arm]['plugin']} "
              f"lower {ess_results[arm]['lower_bound']} "
              f"exceeds {ess_results[arm]['exceeds_grid']}", flush=True)

    # -- baseline + ESS: target-HELDOUT (multi-day subpopulation) ----------
    hold_ids = [p for p in ids if static[p].heldout_days is not None]
    static_hold = {p: static[p] for p in hold_ids}
    cache_hold = BaselineCache(hold_ids, static_hold, n_runs, "hold", workdir,
                               processes=processes)
    n_h = len(hold_ids)
    sizes_hold = [s for s in grid if n_h // s >= 2]
    ess_heldout: Dict[str, dict] = {}
    for arm in ARM_CARDS:
        ess_heldout[arm] = ess_for_arm(
            arm_losses[arm]["heldout"], "heldout", cache_hold, sizes_hold
        )
        print(f"[ESS heldout] {arm}: plugin {ess_heldout[arm]['plugin']} "
              f"lower {ess_heldout[arm]['lower_bound']} "
              f"exceeds {ess_heldout[arm]['exceeds_grid']}", flush=True)

    # baseline mean-loss learning curves (for the report/figure)
    base_curve_all = {
        str(s): float(np.nanmean(cache_all.matrices(s)["all"]))
        for s in sizes_all if s in cache_all._cache
    }
    base_curve_hold = {
        str(s): float(np.nanmean(cache_hold.matrices(s)["heldout"]))
        for s in sizes_hold if s in cache_hold._cache
    }

    # -- template-vs-LLM head-to-head --------------------------------------
    deployed_cards = load_cards(ARM_CARDS["m2_deployed"])
    deployed_source = {
        c["persona_id"]: c.get("provenance", {}).get("card_source", "unknown")
        for c in deployed_cards
    }
    h2h = {
        "target_all": headtohead(
            arm_losses["m2_deployed"]["all"], arm_losses["template"]["all"],
            deployed_source, static, ids,
        ),
        "target_heldout": headtohead(
            arm_losses["m2_deployed"]["heldout"],
            arm_losses["template"]["heldout"],
            deployed_source, static, hold_ids,
        ),
    }

    # -- results -----------------------------------------------------------
    results = {
        "protocol": {
            "loss_adapter": "runs/e7_tiers/e7_manifest.json -> ess_loss_adapter "
                            "(pinned 2026-07-20)",
            "estimator": "evaluation.ess (Gao, Han & Liang 2026, "
                         "arXiv 2601.12343, exact difference-loss test)",
            "grid": list(grid),
            "alpha": ALPHA,
            "ensemble_members": n_runs,
            "n_observations": n,
            "n_multiday": n_h,
            "smoke": smoke,
        },
        "per_arm": {
            arm: {
                "loss_all": _summary(arm_losses[arm]["all"]),
                "loss_heldout": _summary(arm_losses[arm]["heldout"]),
                "aggregate_pooled_tvd": aggregate[arm],
                "ess_all": ess_results[arm],
                "ess_heldout": ess_heldout[arm],
                "in_evidence_flag": {
                    "target_all": arm in ("T4", "T4_noclaims", "T4_nofidelity",
                                          "T5", "m2_deployed", "template"),
                    "target_heldout": arm in ("T5", "m2_deployed", "template"),
                },
            }
            for arm in ARM_CARDS
        },
        "baseline_mean_loss_curve": {
            "target_all": base_curve_all,
            "target_heldout": base_curve_hold,
        },
        "template_vs_llm": h2h,
        "timing_seconds": {"total": round(time.time() - t0, 1)},
    }

    manifest = {
        "driver": "evaluation.run_e7_ess",
        "adapter_version": psrc.ADAPTER_VERSION,
        "taxonomy": "m0-1.0",
        "arms": {a: arm_meta[a] for a in ARM_CARDS},
        "t4_context_sha256": _sha256_file(T4_CONTEXT),
        "e7_manifest_sha256": _sha256_file("runs/e7_tiers/e7_manifest.json"),
        "grid": list(grid),
        "alpha": ALPHA,
        "ensemble_members": n_runs,
        "block_namespace": BLOCK_NAMESPACE,
        "logit_iters": LOGIT_ITERS,
        "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
        "smoke": smoke,
        "timestamp": _date_stamp(),
    }

    (out / "results.json").write_text(json.dumps(_to_native(results), indent=2))
    (out / "manifest.json").write_text(json.dumps(_to_native(manifest), indent=2))
    # per-persona losses (full vectors, for figures and audits)
    per_persona = {
        arm: {t: arm_losses[arm][t] for t in ("all", "heldout")}
        for arm in ARM_CARDS
    }
    (out / "per_persona_losses.json").write_text(
        json.dumps(_to_native(per_persona))
    )
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--smoke", type=int, default=None,
                    help="DEV shakeout on the first N personas; never a record")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--grid", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)
    results = run(args.out, n_runs=args.runs, smoke=args.smoke,
                  processes=args.processes,
                  grid=tuple(args.grid) if args.grid else ESS_GRID)
    print(json.dumps(_to_native({
        arm: {
            "mean_loss_all": results["per_arm"][arm]["loss_all"]["mean"],
            "ess_all": {
                k: results["per_arm"][arm]["ess_all"][k]
                for k in ("plugin", "lower_bound", "exceeds_grid")
            },
        }
        for arm in results["per_arm"]
    }), indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
