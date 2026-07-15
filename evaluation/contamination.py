"""E5(i) contamination-probe machinery, per sealed A2.3 + A1.4 and spec D9.

Two jobs live here, both eval-harness-only:

1. ``build_unmasked_prompts`` — the UNMASKED arm's prompt file. Same evidence
   pipeline, same single render path (grounding.render.render_seed_prompt) as
   the masked seeding step; the ONLY difference is data-level substitution
   before rendering (spec D9): skeleton zone codes and masked evidence-line
   phrases are replaced through the quarantined mapping in
   evaluation/truth/unmasked_vocabulary.py. There is no second prompt
   constructor and no render fork — an unmasked prompt differs from its
   masked twin exactly by the substituted data.

   MASK-LINT EXEMPTION — EXPLICIT AND QUARANTINED HERE. The unmasked prompt
   file is DESIGNED to carry forbidden tokens (that is the probe). The
   exemption is implemented by construction, in this module only:

   * this builder mirrors grounding.seeding.build_prompts's loop but never
     calls seeding's mask-lint gate (the gate function is bypassed, not
     parameterized — grounding/seeding.py has no lint=False flag and must
     never grow one);
   * the output file must be written OUTSIDE the repo tree or under data/
     (gitignored; data/synthetic/ is tracked and therefore refused), enforced
     by ``_assert_out_path_allowed`` — so the file can never land on a
     surface the repo mask-lint gate scans, or in git history;
   * grounding/, agents/, serving/, world/ can never import the vocabulary
     this builder substitutes (tests/test_truth_import_boundary.py).

2. ``paired_probe`` / ``score_e5`` — A2.3(i) scoring: identical agent
   populations, CRN-paired seeds (the SAME ``run{k}`` namespaces in both
   arms — the pairing that seals the 25% threshold), both arms scored ONCE
   against the same fixed reference sample (the full-sample diary
   distributions from the promoted adapter; no resampling anywhere in
   scoring). Relative improvement = (TVD_masked - TVD_unmasked) / TVD_masked
   on the ensemble-mean pooled TVD (pooled = max across the three frozen
   families, the E1 pooling); flag if > 0.25.

Quarantine note: the unmasked vocabulary import happens lazily inside
``build_unmasked_prompts`` — the evaluation harness (or test) must run with
AGORA_EVAL_CONTEXT=1; the scoring half of this module needs no truth import
at all (E5(i) is a calibration-period task).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from agents.card_executor import RealizedDay, execute_days
from evaluation.e2 import day_slots_of, namespaces_for
from grounding import seeding
from grounding.adapters import psrc
from grounding.masking.mask_lint import default_token_path, lint_text, load_forbidden_tokens
from grounding.render import render_seed_prompt
from grounding.taxonomy import MODES

_REPO_ROOT = Path(__file__).resolve().parents[1]

E5_FLAG_THRESHOLD = 0.25  # sealed A2.3(i): paired protocol only
FAMILIES = ("trips_per_day", "mode_shares", "time_bands")
DEFAULT_RUNS = 20


# ---------------------------------------------------------------------------
# the unmasked arm's prompt file
# ---------------------------------------------------------------------------

def _unmasked_vocabulary():
    """Deferred quarantined import (requires AGORA_EVAL_CONTEXT=1)."""
    from evaluation.truth import unmasked_vocabulary

    return unmasked_vocabulary


def _assert_out_path_allowed(out_path: Path) -> None:
    """The unmasked prompt file may live OUTSIDE the repo tree, or under
    data/ (gitignored) — never anywhere else in the repo, and never under
    data/synthetic/ (which is tracked). Part of the quarantined lint
    exemption (module docstring)."""
    resolved = Path(out_path).resolve()
    try:
        rel = resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return  # outside the repo tree: fine
    parts = rel.parts
    if parts[:1] == ("data",) and parts[:2] != ("data", "synthetic"):
        return  # data/** is gitignored (data/synthetic/ excepted -> refused)
    raise ValueError(
        "E5 unmasked prompt files are designed to carry forbidden tokens and "
        "must be written outside the repo tree or under data/ (gitignored; "
        f"not data/synthetic/). Refusing: {resolved}"
    )


def build_unmasked_prompts(
    dataset,
    out_path,
    data_dir=psrc.DEFAULT_DATA_DIR,
    zone_of_tract=None,
    mode: str = "serve",
) -> dict:
    """Render every persona's E5(i) UNMASKED seed prompt to a JSONL file.

    Mirrors grounding.seeding.build_prompts line by line — same persona
    index, same evidence lines, same render_seed_prompt — with exactly two
    data-level substitutions before rendering (spec D9): the skeleton's zone
    codes become real place names, and the documented string mapping
    (unmasked_vocabulary.EVIDENCE_LINE_MAP) is applied to each masked
    evidence line. The mask-lint gate is deliberately NOT applied (see the
    module docstring for how that exemption is quarantined); the file
    location is restricted instead. Each line is
    ``{"persona_id", "prompt", "attempt": 1, "arm": "unmasked"}`` —
    a superset of the masked prompt-file schema, so the same batch driver
    consumes both.

    Returns a summary including ``n_prompts_with_unmasked_tokens``, the
    count of prompts carrying at least one forbidden token (an INVERSE
    sanity metric: for a real build it should equal n_prompts)."""
    uv = _unmasked_vocabulary()
    out_path = Path(out_path)
    _assert_out_path_allowed(out_path)

    pidx = seeding.persona_index(dataset, data_dir=data_dir, zone_of_tract=zone_of_tract)
    trips = seeding.enriched_trips(dataset, data_dir=data_dir)
    trips_by_person = {pid: grp for pid, grp in trips.groupby("person_id")}

    person_days = dataset.person_days.copy()
    person_days["person_id"] = person_days["person_id"].astype(str)
    days_by_person = {pid: grp for pid, grp in person_days.groupby("person_id")}

    person_attrs: Dict[str, dict] = {}
    if data_dir is not None and (Path(data_dir) / psrc._PERSONS_CSV).exists():
        pa = seeding.enriched_person_attrs(data_dir)
        pa["person_id"] = pa["person_id"].astype(str)
        person_attrs = {r.person_id: r._asdict() for r in pa.itertuples(index=False)}

    # forbidden tokens loaded for the INVERSE count only — never as a gate
    tokens = load_forbidden_tokens(default_token_path())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_with_tokens = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for r in pidx.itertuples(index=False):
            row = r._asdict()
            persona_id = row["persona_id"]
            person_id = row["person_id"]
            skeleton = seeding._skeleton_from_index_row(row)
            pdays = days_by_person.get(person_id, dataset.person_days.iloc[0:0])
            ptrips = trips_by_person.get(person_id, trips.iloc[0:0])
            prow = person_attrs.get(person_id, {})
            evidence = seeding.evidence_lines_of(pdays, ptrips, prow)

            # spec D9: data-level substitution, same render path
            unmasked_skeleton = uv.unmask_skeleton(skeleton)
            unmasked_evidence = [uv.unmask_text(line) for line in evidence]

            render_skeleton = {
                k: ("none" if v is None else v) for k, v in unmasked_skeleton.items()
            }
            prompt = render_seed_prompt(render_skeleton, unmasked_evidence, len(pdays), mode)
            if lint_text(prompt, tokens):
                n_with_tokens += 1
            fh.write(
                json.dumps(
                    {"persona_id": persona_id, "prompt": prompt, "attempt": 1,
                     "arm": "unmasked"},
                    sort_keys=True,
                )
            )
            fh.write("\n")
            n_written += 1

    return {
        "n_personas": int(len(pidx)),
        "n_prompts": n_written,
        "n_prompts_with_unmasked_tokens": n_with_tokens,
        "out_path": str(out_path),
        "lint_exempt": True,
        "vocabulary_version": uv.VOCABULARY_VERSION,
    }


# ---------------------------------------------------------------------------
# A2.3(i) paired scoring
# ---------------------------------------------------------------------------

def tvd(p, q) -> float:
    """Total variation distance between two distributions (0.5 * L1)."""
    return 0.5 * float(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)).sum())


def realized_distributions(realized: Mapping[str, Sequence[RealizedDay]]) -> Dict[str, np.ndarray]:
    """The three frozen E1 distribution families from one run's realized days.

    Day-weighted trips/day bins (zero-trip days included), trip-weighted mode
    and departure-band shares — each realized trip carrying its day's slot
    weight (spec D5), exactly mirroring the adapter's real-side families."""
    trips_mass = np.zeros(len(psrc.TRIPS_PER_DAY_BINS))
    mode_mass = np.zeros(len(MODES))
    band_mass = np.zeros(len(psrc.TIME_BANDS))
    bin_index = {b: i for i, b in enumerate(psrc.TRIPS_PER_DAY_BINS)}
    mode_index = {m: i for i, m in enumerate(MODES)}
    band_index = {b: i for i, b in enumerate(psrc.TIME_BANDS)}
    for days in realized.values():
        for day in days:
            trips_mass[bin_index[psrc.trips_bin(len(day.trips))]] += day.day_weight
            for t in day.trips:
                mode_mass[mode_index[t.mode]] += day.day_weight
                band_mass[band_index[t.depart_band]] += day.day_weight
    return {
        "trips_per_day": psrc.normalize(trips_mass),
        "mode_shares": psrc.normalize(mode_mass),
        "time_bands": psrc.normalize(band_mass),
    }


def ensemble_mean_distributions(
    cards: Sequence[dict],
    day_slots: Mapping[str, Sequence],
    namespaces: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Ensemble-mean family distributions: execute the cards once per CRN
    namespace and average the per-run distributions per family."""
    per_run = []
    for ns in namespaces:
        realized = execute_days(cards, day_slots, ns, update_habits=False)
        per_run.append(realized_distributions(realized))
    return {
        fam: np.mean([d[fam] for d in per_run], axis=0) for fam in FAMILIES
    }


def reference_distributions(dataset) -> Dict[str, np.ndarray]:
    """The fixed reference sample: the FULL-SAMPLE weighted three-family
    diary distributions from the promoted adapter (no resampling, ever —
    sealed A2.3(i))."""
    return {
        "trips_per_day": psrc.trips_per_day_distribution(dataset.person_days),
        "mode_shares": psrc.mode_share_distribution(dataset.weekday_trips),
        "time_bands": psrc.departure_band_distribution(dataset.weekday_trips),
    }


def paired_probe(
    masked_results: Mapping[str, np.ndarray],
    unmasked_results: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    threshold: float = E5_FLAG_THRESHOLD,
) -> dict:
    """A2.3(i) verdict from two arms' ensemble-mean family distributions.

    Both arms are scored ONCE against the same fixed ``reference``; pooled
    TVD = max across the three families (the E1 pooling); relative
    improvement = (TVD_masked - TVD_unmasked) / TVD_masked; flag if it
    exceeds ``threshold``. A zero masked TVD (nothing left to improve on)
    defines the relative improvement as 0.0 — never a division error."""
    per_family = {}
    for fam in FAMILIES:
        per_family[fam] = {
            "tvd_masked": tvd(masked_results[fam], reference[fam]),
            "tvd_unmasked": tvd(unmasked_results[fam], reference[fam]),
        }
    pooled_masked = max(v["tvd_masked"] for v in per_family.values())
    pooled_unmasked = max(v["tvd_unmasked"] for v in per_family.values())
    if pooled_masked > 0:
        relative_improvement = (pooled_masked - pooled_unmasked) / pooled_masked
    else:
        relative_improvement = 0.0
    return {
        "per_family": per_family,
        "pooled_tvd_masked": float(pooled_masked),
        "pooled_tvd_unmasked": float(pooled_unmasked),
        "relative_improvement": float(relative_improvement),
        "threshold": float(threshold),
        "flag": bool(relative_improvement > threshold),
        "scoring": "ensemble-mean pooled TVD (max across the three families), "
                   "one fixed full-sample reference, no resampling; paired "
                   "protocol (sealed A2.3(i))",
    }


def score_e5(
    masked_cards: Sequence[dict],
    unmasked_cards: Sequence[dict],
    dataset,
    n_runs: int = DEFAULT_RUNS,
    seed: int = 0,
) -> dict:
    """End-to-end E5(i): execute both card populations under the SAME CRN
    namespaces (paired seeds), score both once against the same fixed
    reference, return the probe verdict.

    Enforces the sealed protocol's "identical agent populations": the two
    card files must cover exactly the same persona ids."""
    masked_ids = {c["persona_id"] for c in masked_cards}
    unmasked_ids = {c["persona_id"] for c in unmasked_cards}
    if masked_ids != unmasked_ids:
        diff = sorted(masked_ids.symmetric_difference(unmasked_ids))
        raise ValueError(
            "E5 paired protocol (sealed A2.3(i)) requires IDENTICAL agent "
            f"populations in both arms; {len(diff)} persona id(s) differ: "
            f"{diff[:10]}"
        )

    id_map = seeding._persona_id_map(dataset.persons["person_id"].astype(str))
    slots = day_slots_of(dataset.person_days, id_map)
    namespaces = namespaces_for(n_runs, seed)  # SAME namespaces, both arms

    masked_mean = ensemble_mean_distributions(masked_cards, slots, namespaces)
    unmasked_mean = ensemble_mean_distributions(unmasked_cards, slots, namespaces)
    reference = reference_distributions(dataset)
    probe = paired_probe(masked_mean, unmasked_mean, reference)

    return {
        "eval": "e5i",
        "n_personas": len(masked_ids),
        "n_runs": int(n_runs),
        "seed": int(seed),
        "namespaces": namespaces,
        "ensemble_mean_masked": {f: [float(x) for x in masked_mean[f]] for f in FAMILIES},
        "ensemble_mean_unmasked": {f: [float(x) for x in unmasked_mean[f]] for f in FAMILIES},
        "reference": {f: [float(x) for x in reference[f]] for f in FAMILIES},
        "probe": probe,
        "contamination_flag": probe["flag"],
    }
