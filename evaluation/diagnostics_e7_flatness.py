"""E7 flatness audit — diagnostics on the SEALED BT1 outputs (WS1, 2026-07-19).

Read-only over ``runs/bt1/`` (the sealed verdict is never touched); everything
else is deterministic CPU replay of already-fired arms:

  * replays tier arms with ``CachedRewriteGenerator(offline=True)`` — every
    generation is served from the pulled BT1 cache; a single cache miss aborts
    (it would mean the replay diverged from the firing);
  * FIDELITY GATE: each replayed arm's drop_3mo must equal the sealed value in
    ``runs/bt1/results.json`` to 1e-9, else the replay (and every number built
    on it) is rejected;
  * per-agent facility assignment is reconstructed by re-walking the loop's
    corridor block (same equilibrium call, same CRN route keys) —
    RECONSTRUCTION GATE: the per-day bincount must reproduce the loop's
    recorded ``facility_loads`` exactly, else abort;
  * decomposition (per tier, toll arm r0, placebo for T3): the baseline->post
    tunnel-volume drop is split exactly into
      demand term  sb*(Nb-Np)   (corridor car-travel leaves: mode change /
                                 suppression — card/slow-brain channel)
      route term   Np*(sb-sp)   (T-vs-R share shift among remaining car
                                 travelers — fast-brain/VoT-dial channel)
    where N = corridor car travelers/day, s = tunnel share, b/p = window means;
  * per-agent outcome classes (kept_tunnel / diverted / mode_change /
    suppressed) from realized days + reconstructed facilities;
  * card-channel classification per corridor agent: original tier card vs
    LoopResult final card — did any work-pattern trip leave mode 'car' or a
    pattern weight change ("behavioral") or only rules/habit text ("advisory");
  * cross-tier rewrite similarity: same-persona added-rule Jaccard vs T5;
  * tier-population pre-onset differences (card bytes, rule/pattern counts,
    same-persona identity rate vs T5);
  * optional inspector dump (WS2): per-agent JSON for T5 toll+placebo.

No truth import, no GPU, no blind quantity: every replayed number was sealed
in the firing; this module only re-derives and decomposes it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from agents.baseline_loop import AnnouncedOnset, run_baseline_loop
from agents.slow_brain import GatedSlowBrain, StandardSurprisePolicy
from evaluation import run_bt1 as bt1
from serving.vllm_generator import CachedRewriteGenerator
from world import bridge, crn
from world.config import cityk_corridor
from world.network import realized_facilities, solve_corridor_equilibrium
from world.tolling import announcement_of, placebo_announcement

SEALED = Path("runs/bt1/results.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# replay (mirrors run_bt1.run_arm, but keep_full_window=True)
# ---------------------------------------------------------------------------

def replay_arm(cards, context, *, config, k: int, arm: str, threshold: int,
               say_do: float, gates, cache_path: str):
    gen = CachedRewriteGenerator(cache_path=cache_path, offline=True)
    client = GatedSlowBrain(gen, context)
    tolled = arm != "placebo"
    onset = AnnouncedOnset(
        day=bt1.ONSET_DAY,
        announcement=announcement_of(config.toll_schedule,
                                     say_do_price_correction=say_do)
        if tolled else placebo_announcement(),
        tail_surprises=(arm != "toll_tail_off"),
    )
    ns = f"bt1_r{k}"
    res = run_baseline_loop(
        cards, config, {}, namespace=ns, n_days=bt1.N_DAYS,
        warmup_days=bt1.WARMUP_DAYS, policy=StandardSurprisePolicy(),
        client=client, network_override=bt1._override(config, tolled),
        keep_full_window=True, onset=onset,
        strong_habit_threshold=threshold,
        persona_pass=gates["persona_pass"], car_access=gates["car_access"],
    )
    return res, ns


def drop_of(res) -> float:
    series = {}
    for d, rec in res.facility_loads.items():
        codes = rec["codes"]
        if "T" in codes:
            series[int(d)] = float(rec["loads"][codes.index("T")])
    base = [series[d] for d in series if bt1.WARMUP_DAYS <= d < bt1.ONSET_DAY]
    post = [series[d] for d in sorted(series) if d >= bt1.ONSET_DAY][:bt1.POST_3MO_DAYS]
    return 1.0 - float(np.mean(post)) / float(np.mean(base))


# ---------------------------------------------------------------------------
# per-agent facility reconstruction (verified against recorded aggregates)
# ---------------------------------------------------------------------------

def reconstruct_assignments(res, cards_final, config, namespace: str,
                            persona_pass, tolled: bool) -> Dict[int, Dict[str, str]]:
    """day -> {persona_id -> facility code}, verified day-by-day against the
    loop's recorded facility_loads (exact bincount match or ValueError)."""
    population = bridge.population_from_cards(
        cards_final, config, namespace, persona_pass=persona_pass)
    row_index = bridge.persona_row_index(cards_final)
    by_day: Dict[int, Dict[str, str]] = {}
    for d, rec in sorted(res.facility_loads.items()):
        day_out = {}
        for pid, days in res.realized_days_full.items():
            todays = [rd for rd in days if rd.day_index == d]
            if todays:
                day_out[pid] = todays
        state_net = bt1._override(config, tolled)(d)
        if list(state_net.facility_codes) != list(rec["codes"]):
            raise ValueError(f"day {d}: facility codes mismatch vs recorded")
        table = bridge.corridor_travelers_of_day(
            day_out, cards_final, config, population=population,
            row_index=row_index)
        if len(table) != int(rec.get("n_travelers", len(table))):
            raise ValueError(
                f"day {d}: traveler count {len(table)} != recorded "
                f"{rec.get('n_travelers')}")
        facilities = [config.facility(c) for c in state_net.facility_codes]
        eq = solve_corridor_equilibrium(
            facilities, access=table.access, vot=table.vot,
            period_codes=table.period_codes, has_pass=table.has_pass,
            state=state_net, theta=config.logit_theta)
        keys = ["%s:%s:%d:route" % (namespace, pid, d) for pid in table.persona_ids]
        choice = realized_facilities(crn.draws(keys), eq.choice_probs)
        loads = np.bincount(choice, minlength=len(facilities)).astype(float)
        if not np.array_equal(loads, np.asarray(rec["loads"], dtype=float)):
            raise ValueError(
                f"day {d}: reconstructed loads {loads.tolist()} != recorded "
                f"{rec['loads']} — reconstruction NOT trusted")
        by_day[d] = {pid: rec["codes"][int(c)]
                     for pid, c in zip(table.persona_ids, choice)}
    return by_day


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

def volume_decomposition(res, assign: Dict[int, Dict[str, str]]) -> dict:
    """Exact two-term split of the baseline->post tunnel drop."""
    days = sorted(res.facility_loads)
    base_days = [d for d in days if bt1.WARMUP_DAYS <= d < bt1.ONSET_DAY]
    post_days = [d for d in days if d >= bt1.ONSET_DAY][:bt1.POST_3MO_DAYS]

    def stats(dd):
        n = [len(assign[d]) for d in dd]
        t = [sum(1 for f in assign[d].values() if f == "T") for d in dd]
        return float(np.mean(n)), float(np.mean(t))

    nb, tb = stats(base_days)
    np_, tp = stats(post_days)
    sb, sp = tb / nb, tp / np_
    dv = tb - tp
    demand = sb * (nb - np_)
    route = np_ * (sb - sp)
    return {
        "baseline": {"travelers_per_day": nb, "tunnel_per_day": tb, "share": sb},
        "post": {"travelers_per_day": np_, "tunnel_per_day": tp, "share": sp},
        "volume_drop_abs": dv,
        "drop_relative": dv / tb,
        "demand_term_abs": demand,
        "route_term_abs": route,
        "demand_share_of_drop": demand / dv if dv else None,
        "route_share_of_drop": route / dv if dv else None,
        "identity_residual": dv - demand - route,  # exact: 0 by construction
    }


def agent_outcomes(res, assign, corridor_pids) -> Dict[str, dict]:
    """Per corridor agent: baseline/post traveler-day counts, majority
    facility, and outcome class."""
    base_rng = range(bt1.WARMUP_DAYS, bt1.ONSET_DAY)
    post_rng = range(bt1.ONSET_DAY, bt1.N_DAYS)
    out: Dict[str, dict] = {}
    trav_days: Dict[str, Dict[str, List[int]]] = {}
    for d, m in assign.items():
        for pid, fac in m.items():
            trav_days.setdefault(pid, {}).setdefault(fac, []).append(d)
    for pid in corridor_pids:
        days = res.realized_days_full.get(pid, [])
        def work_modes(rng):
            c = Counter()
            for rd in days:
                if rd.day_index in rng:
                    for t in rd.trips:
                        if t.purpose == "work":
                            c[t.mode] += 1
            return c
        wb, wp = work_modes(base_rng), work_modes(post_rng)
        facs = trav_days.get(pid, {})
        base_T = sum(1 for d in facs.get("T", []) if d in base_rng)
        base_R = sum(1 for d in facs.get("R", []) if d in base_rng)
        post_T = sum(1 for d in facs.get("T", []) if d in post_rng)
        post_R = sum(1 for d in facs.get("R", []) if d in post_rng)
        post_car_days = post_T + post_R
        if post_car_days > 0:
            outcome = "kept_tunnel" if post_T >= post_R else "diverted"
        elif sum(wp.values()) > 0:
            outcome = "mode_change"
        else:
            outcome = "suppressed"
        out[pid] = {
            "baseline": {"T": base_T, "R": base_R, "work_modes": dict(wb)},
            "post": {"T": post_T, "R": post_R, "work_modes": dict(wp)},
            "outcome": outcome,
        }
    return out


def _work_pattern_sig(card) -> List[tuple]:
    sig = []
    for p in card.get("patterns", []):
        for t in p.get("trips", []):
            if t.get("purpose") == "work":
                sig.append((p.get("id"), round(float(p.get("weight", 0)), 6),
                            t.get("mode"), t.get("depart_band")))
    return sorted(sig)


def _rule_texts(card) -> set:
    return {json.dumps({k: r[k] for k in sorted(r) if k != "id"}, sort_keys=True)
            for r in card.get("rules", [])}


def card_channel(card_before, card_after) -> dict:
    """behavioral = any work-trip mode/weight/pattern change; advisory = only
    rules/other text changed; none = identical."""
    if card_after is None:
        return {"channel": "none", "rules_added": 0, "rules_removed": 0}
    b_sig, a_sig = _work_pattern_sig(card_before), _work_pattern_sig(card_after)
    b_rules, a_rules = _rule_texts(card_before), _rule_texts(card_after)
    behavioral = b_sig != a_sig
    changed = behavioral or (b_rules != a_rules)
    return {
        "channel": ("behavioral" if behavioral else
                    "advisory" if changed else "none"),
        "rules_added": len(a_rules - b_rules),
        "rules_removed": len(b_rules - a_rules),
        "added_rules": sorted(a_rules - b_rules),
    }


def tier_population_diffs(tiers: dict, ref: str = "T5") -> dict:
    out = {}
    ref_cards = {c["persona_id"]: c for c in tiers[ref]["cards"]}
    for tier, tv in tiers.items():
        n_rules, n_pat, nbytes, identical, jac = [], [], [], 0, []
        for c in tv["cards"]:
            n_rules.append(len(c.get("rules", [])))
            n_pat.append(len(c.get("patterns", [])))
            nbytes.append(len(json.dumps(c, sort_keys=True)))
            rc = ref_cards[c["persona_id"]]
            if tier != ref:
                if json.dumps(c, sort_keys=True) == json.dumps(rc, sort_keys=True):
                    identical += 1
                a, b = _rule_texts(c), _rule_texts(rc)
                jac.append(len(a & b) / len(a | b) if (a | b) else 1.0)
        out[tier] = {
            "mean_rules": float(np.mean(n_rules)),
            "mean_patterns": float(np.mean(n_pat)),
            "mean_card_bytes": float(np.mean(nbytes)),
            "identical_to_ref_rate": identical / len(tv["cards"]) if tier != ref else 1.0,
            "mean_rule_jaccard_vs_ref": float(np.mean(jac)) if jac else 1.0,
        }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiers-dir", default="runs/e7_tiers")
    ap.add_argument("--cache", default="ignore/bt1_cache/gen_cache.jsonl")
    ap.add_argument("--m4-gates-pass", default="runs/m4_prep/has_pass_household")
    ap.add_argument("--borrowed-car", default="runs/m4_prep/borrowed_car_t5")
    ap.add_argument("--fit-manifest", default="calibration/sr520_fit_manifest.json")
    ap.add_argument("--e3-manifest", default="calibration/e3_fit_manifest.json")
    ap.add_argument("--run", type=int, default=0, help="CRN member k to replay")
    ap.add_argument("--tiers", default="T1,T2,T3,T4,T4_noclaims,T5")
    ap.add_argument("--out", default="runs/e7_flatness_audit")
    ap.add_argument("--dump-inspector", default=None,
                    help="dir for the WS2 per-agent dump (T5 both arms)")
    args = ap.parse_args(argv)

    sealed = json.loads(SEALED.read_text())
    frozen = bt1.load_frozen(Path(args.fit_manifest), Path(args.e3_manifest),
                             allow_stub=False)
    gates = bt1.load_gates(Path(args.m4_gates_pass), Path(args.borrowed_car))
    tiers = bt1.load_tiers(Path(args.tiers_dir))
    from dataclasses import replace as dc_replace
    base_cfg = cityk_corridor()
    config = dc_replace(base_cfg, vot_median=base_cfg.vot_median * frozen["vot_scale"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = args.run
    tier_list = [t.strip() for t in args.tiers.split(",") if t.strip()]

    fidelity, decomposition, outcomes_summary, channels = {}, {}, {}, {}
    added_rules_by_tier: Dict[str, Dict[str, set]] = {}
    inspector_payload = {}

    for tier in tier_list:
        arms = ("toll", "placebo") if tier in ("T3", "T5") else ("toll",)
        for arm in arms:
            label = f"{tier}/{arm}"
            print(f"[replay] {label} r{k} ...", flush=True)
            cards0 = copy.deepcopy(tiers[tier]["cards"])
            res, ns = replay_arm(
                cards0, tiers[tier]["context"], config=config, k=k, arm=arm,
                threshold=frozen["threshold"], say_do=frozen["say_do"],
                gates=gates, cache_path=args.cache)
            got = drop_of(res)
            want = sealed["results"]["tiers"][tier]["arms"][arm][k]["drop_3mo"]
            ok = abs(got - want) < 1e-9
            fidelity[label] = {"replayed": got, "sealed": want, "match": ok}
            if not ok:
                raise SystemExit(f"FIDELITY GATE FAILED {label}: {got} != {want}")
            assign = reconstruct_assignments(
                res, res.cards, config, ns, gates["persona_pass"],
                tolled=(arm != "placebo"))
            decomposition[label] = volume_decomposition(res, assign)

            orig = {c["persona_id"]: c for c in tiers[tier]["cards"]}
            final = {c["persona_id"]: c for c in res.cards}
            corridor = sorted({pid for m in assign.values() for pid in m}
                              | {a.persona_id for a in res.rewrite_audit})
            oc = agent_outcomes(res, assign, corridor)
            outcomes_summary[label] = dict(Counter(v["outcome"] for v in oc.values()))
            ch = {pid: card_channel(orig[pid], final.get(pid)) for pid in corridor}
            channels[label] = dict(Counter(v["channel"] for v in ch.values()))
            if arm == "toll":
                added_rules_by_tier[tier] = {
                    pid: set(v["added_rules"]) for pid, v in ch.items()}

            if args.dump_inspector and tier == "T5":
                inspector_payload[arm] = {
                    "namespace": ns,
                    "assignments": {str(d): m for d, m in assign.items()},
                    "outcomes": oc,
                    "channels": ch,
                    "cards_before": orig,
                    "cards_after": final,
                    "realized": {
                        pid: [{"d": rd.day_index,
                               "trips": [{"purpose": t.purpose, "mode": t.mode,
                                          "band": t.depart_band,
                                          "rule": t.rule_applied}
                                         for t in rd.trips]}
                              for rd in days]
                        for pid, days in res.realized_days_full.items()},
                    "facility_loads": {str(d): r for d, r in res.facility_loads.items()},
                    "rewrite_audit": [a.to_dict() for a in res.rewrite_audit],
                }

    # cross-tier rewrite similarity vs T5 (same persona, added-rule Jaccard)
    similarity = {}
    if "T5" in added_rules_by_tier:
        ref = added_rules_by_tier["T5"]
        for tier, m in added_rules_by_tier.items():
            if tier == "T5":
                continue
            js = [len(m[pid] & ref[pid]) / len(m[pid] | ref[pid])
                  for pid in m if pid in ref and (m[pid] | ref[pid])]
            similarity[f"{tier}_vs_T5"] = {
                "mean_added_rule_jaccard": float(np.mean(js)) if js else None,
                "n": len(js)}

    pop_diffs = tier_population_diffs({t: tiers[t] for t in tier_list})

    # paired member spread vs T5, from the sealed file (WS1(c))
    paired = {}
    t5m = np.asarray(sealed["results"]["tiers"]["T5"]["delta_q"]["members"])
    for tier in tier_list:
        m = np.asarray(sealed["results"]["tiers"][tier]["delta_q"]["members"])
        d = m - t5m
        paired[f"{tier}_minus_T5"] = {
            "mean": float(d.mean()), "std": float(d.std(ddof=1)),
            "min": float(d.min()), "max": float(d.max()),
            "members": d.tolist()}

    payload = {
        "eval": "E7 flatness audit (WS1) — replay diagnostics on sealed BT1",
        "replayed_member": k,
        "fidelity_gate": fidelity,
        "decomposition": decomposition,
        "outcome_classes": outcomes_summary,
        "card_channels": channels,
        "cross_tier_added_rule_similarity_vs_T5": similarity,
        "tier_population_diffs": pop_diffs,
        "paired_members_vs_T5": paired,
    }
    (out / "audit.json").write_text(json.dumps(payload, indent=2))
    manifest = {
        "inputs": {
            "sealed_results": {"path": str(SEALED), "sha256": _sha256(SEALED)},
            "cache": {"path": args.cache, "sha256": _sha256(Path(args.cache))},
            "tier_cards_sha256": {t: tiers[t]["cards_sha256"] for t in tier_list},
        },
        "audit_sha256": _sha256(out / "audit.json"),
        "note": "read-only over runs/bt1; deterministic offline-cache replay; "
                "fidelity + reconstruction gates enforced (abort on mismatch)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dump_inspector and inspector_payload:
        dump_dir = Path(args.dump_inspector)
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / "raw_replay_T5.json").write_text(json.dumps(inspector_payload))
        print(f"[dump] inspector raw dump -> {dump_dir}/raw_replay_T5.json")

    print(json.dumps({"fidelity": {k2: v["match"] for k2, v in fidelity.items()},
                      "decomposition": {k2: {"demand": round(v["demand_share_of_drop"], 4),
                                             "route": round(v["route_share_of_drop"], 4)}
                                        for k2, v in decomposition.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
