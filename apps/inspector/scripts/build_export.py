#!/usr/bin/env python3
"""Build the inspector app's real data export from the WS1 replay dumps.

Read-only inputs:  ignore/inspector_dump/raw_replay_{tier}_{arm}.json
                   apps/inspector/public/data/zones.geojson (invented City K)
Outputs:           apps/inspector/public/data/{meta,agents,aggregate}.json

NO simulation code is imported (owner rule, WS2): this script maps replay
dump records onto the app's fixed data contract. The dumps themselves were
produced by evaluation/diagnostics_e7_flatness.py under its replay gates.

Stratified ~200 selection (owner ruling 2026-07-19):
  * T3 churners — corridor agents whose card churned (channel != none),
    toll AND placebo timelines both shown side by side;
  * T5 behavioral-rewrite agents — the ~1% channel, every one included;
  * remaining slots: kept/diverted/mode_change/suppressed samples across
    all six tiers + a little non-corridor map texture.

The produced files are record-derived (they embed masked card content) and
live under a path that is git-ignored: they are never committed.

Habit chart semantics: the exported "habit" series is CUMULATIVE RULE
APPLICATIONS per rule per day (from realized trips' rule_applied), not the
internal habit-strength counter; documented in the app README.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DUMPS = REPO / "ignore" / "inspector_dump"
DATA = REPO / "apps" / "inspector" / "public" / "data"
TIERS = ["T1", "T2", "T3", "T4", "T4_noclaims", "T5"]
N_DAYS, ONSET = 102, 39


def load(tier, arm):
    fp = DUMPS / f"raw_replay_{tier}_{arm}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def det_rng(*parts) -> random.Random:
    return random.Random(int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"))


# --- invented-city zone placement -----------------------------------------

def zone_groups(zones_geojson):
    """Rank zone polygons by centroid distance from the map centre and split
    into 4 radial groups (proxy rings for the masked ring codes)."""
    feats = [f for f in zones_geojson["features"]
             if f["properties"].get("zone_id")]
    def centroid(f):
        pts = []
        geom = f["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else \
            [r for poly in geom["coordinates"] for r in poly]
        for ring in rings:
            pts.extend(ring)
        xs, ys = zip(*pts)
        return sum(xs) / len(xs), sum(ys) / len(ys)
    cents = {f["properties"]["zone_id"]: centroid(f) for f in feats}
    cx = sum(x for x, _ in cents.values()) / len(cents)
    cy = sum(y for _, y in cents.values()) / len(cents)
    ranked = sorted(cents, key=lambda z: (cents[z][0] - cx) ** 2 + (cents[z][1] - cy) ** 2)
    q = max(1, len(ranked) // 4)
    groups = [ranked[:q], ranked[q:2 * q], ranked[2 * q:3 * q], ranked[3 * q:]]
    return groups, cents


def place(pid, ring_val, groups, cents, salt):
    order = {"0": 0, "core": 0, "1": 1, "inner": 1, "2": 2, "mid": 2}
    gi = order.get(str(ring_val).lower(), 3)
    g = groups[min(gi, len(groups) - 1)]
    rng = det_rng(pid, salt)
    z = g[rng.randrange(len(g))]
    x, y = cents[z]
    return z, [round(x + rng.uniform(-0.004, 0.004), 6),
               round(y + rng.uniform(-0.004, 0.004), 6)]


# --- card rendering ---------------------------------------------------------

def card_lines(card) -> list:
    if card is None:
        return []
    out = []
    for p in card.get("patterns", []):
        for t in p.get("trips", []):
            out.append(f"pattern {p.get('id')} (w={p.get('weight')}): "
                       f"{t.get('purpose')} by {t.get('mode')} @ {t.get('depart_band')}")
    for r in card.get("rules", []):
        bits = {k: v for k, v in r.items() if k not in ("id",)}
        out.append(f"rule {r.get('id')}: " + json.dumps(bits, sort_keys=True))
    return out


# --- per-agent assembly ------------------------------------------------------

def timeline_of(dump, pid):
    days = {d["d"]: d for d in dump["realized"].get(pid, [])}
    tl = []
    for d in range(N_DAYS):
        fac = dump["assignments"].get(str(d), {}).get(pid)
        rec = days.get(d)
        mode = "none"
        if rec:
            work = [t for t in rec["trips"] if t["purpose"] == "work"]
            mode = work[0]["mode"] if work else (
                rec["trips"][0]["mode"] if rec["trips"] else "none")
        tl.append({"d": d, "mode": mode, "fac": fac})
    return tl


def habit_of(dump, pid):
    counts = defaultdict(lambda: [0] * N_DAYS)
    for rec in dump["realized"].get(pid, []):
        for t in rec["trips"]:
            if t.get("rule"):
                counts[t["rule"]][rec["d"]] += 1
    out = []
    for rule, per_day in sorted(counts.items()):
        c, series = 0, []
        for d in range(N_DAYS):
            c += per_day[d]
            series.append([d, c])
        out.append({"rule": str(rule), "series": series})
    return out


def skeleton_field(card, *names, default=None):
    for scope in (card, card.get("skeleton", {}) if isinstance(card, dict) else {}):
        if isinstance(scope, dict):
            for n in names:
                if n in scope and scope[n] is not None:
                    return scope[n]
    return default


def build_agent(pid, tier, toll_dump, placebo_dump, groups, cents):
    card_b = toll_dump["cards_before"].get(pid)
    card_a = toll_dump["cards_after"].get(pid)
    ch = toll_dump["channels"].get(pid, {"channel": "none"})
    oc = toll_dump["outcomes"].get(pid)
    statics = toll_dump["statics"].get(pid, {})
    cars = skeleton_field(card_b, "household_cars", "n_cars", default=0)
    income = skeleton_field(card_b, "income_band", "income", default="unknown")
    hz, hxy = place(pid, statics.get("home_ring", "3"), groups, cents, "home")
    wz, wxy = place(pid, statics.get("work_ring", "0"), groups, cents, "work")
    changed = ch["channel"] != "none"
    return {
        "id": pid, "tier": tier,
        "home_zone": hz, "work_zone": wz, "home_xy": hxy, "work_xy": wxy,
        "income_band": str(income), "n_cars": int(cars) if cars is not None else 0,
        "pass_holder": bool(skeleton_field(card_b, "has_pass", default=False)),
        "corridor_traveler": bool(statics.get("is_corridor", pid in toll_dump["outcomes"])),
        "rewrite_fired": changed,
        "outcome": (oc or {}).get("outcome", "kept_tunnel"),
        "channel": ch["channel"],
        "card_before": card_lines(card_b),
        "card_after": card_lines(card_a) if changed else None,
        "habit": habit_of(toll_dump, pid),
        "timeline": {"toll": timeline_of(toll_dump, pid),
                     "placebo": timeline_of(placebo_dump, pid) if placebo_dump else []},
    }


def main():
    zones = json.loads((DATA / "zones.geojson").read_text())
    groups, cents = zone_groups(zones)
    dumps = {(t, a): load(t, a) for t in TIERS for a in ("toll", "placebo")}
    missing = [k for k, v in dumps.items() if v is None]
    if missing:
        raise SystemExit(f"missing dumps: {missing}")

    agents, seen = [], set()

    def add(pid, tier):
        if (pid, tier) in seen:
            return
        seen.add((pid, tier))
        agents.append(build_agent(pid, tier, dumps[(tier, "toll")],
                                  dumps[(tier, "placebo")], groups, cents))

    # 1) T5 behavioral-rewrite agents — ALL of them (the ~1% channel).
    t5 = dumps[("T5", "toll")]
    behavioral = sorted(p for p, c in t5["channels"].items()
                        if c["channel"] == "behavioral")
    for pid in behavioral:
        add(pid, "T5")

    # 2) T3 churners (channel != none), toll+placebo side by side.
    t3 = dumps[("T3", "toll")]
    churners = sorted(p for p, c in t3["channels"].items()
                      if c["channel"] != "none")
    for pid in churners[:45]:
        add(pid, "T3")

    # 3) outcome-stratified fill across all tiers.
    per_tier = {"T1": 12, "T2": 12, "T3": 5, "T4": 14, "T4_noclaims": 14, "T5": 20}
    for tier, budget in per_tier.items():
        d = dumps[(tier, "toll")]
        by_outcome = defaultdict(list)
        for pid, o in sorted(d["outcomes"].items()):
            by_outcome[o["outcome"]].append(pid)
        share = max(1, budget // max(1, len(by_outcome)))
        for outcome, pids in sorted(by_outcome.items()):
            rng = det_rng(tier, outcome)
            rng.shuffle(pids)
            for pid in pids[:share]:
                add(pid, tier)

    # 4) non-corridor texture from T5 statics.
    non_corr = sorted(p for p, s in t5["statics"].items()
                      if not s.get("is_corridor"))[:15]
    for pid in non_corr:
        add(pid, "T5")

    # aggregate: T5 tunnel volume, toll vs placebo.
    def series(dump):
        out = []
        for d in range(N_DAYS):
            rec = dump["facility_loads"].get(str(d))
            if rec and "T" in rec["codes"]:
                out.append(float(rec["loads"][rec["codes"].index("T")]))
            else:
                out.append(None)
        return out

    aggregate = {"days": list(range(N_DAYS)),
                 "tunnel": {"toll": series(dumps[("T5", "toll")]),
                            "placebo": series(dumps[("T5", "placebo")])}}
    meta = {"onset_day": ONSET, "n_days": N_DAYS, "warmup_days": 10,
            "tiers": TIERS,
            "income_bands": sorted({a["income_band"] for a in agents}),
            "outcomes": ["kept_tunnel", "diverted", "mode_change", "suppressed"],
            "source": "REAL replay export (deterministic, gate-verified); "
                      "habit series = cumulative rule applications"}

    (DATA / "agents.json").write_text(json.dumps(agents))
    (DATA / "aggregate.json").write_text(json.dumps(aggregate))
    (DATA / "meta.json").write_text(json.dumps(meta))
    from collections import Counter
    print(f"agents: {len(agents)}")
    print("by tier:", dict(Counter(a['tier'] for a in agents)))
    print("by outcome:", dict(Counter(a['outcome'] for a in agents)))
    print("behavioral T5:", sum(1 for a in agents if a['tier'] == 'T5'
                                and a['channel'] == 'behavioral'))
    print("T3 churners:", sum(1 for a in agents if a['tier'] == 'T3'
                              and a['channel'] != 'none'))


if __name__ == "__main__":
    main()
