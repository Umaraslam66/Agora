"""METHOD-TRANSFER reweighting: PSRC personas -> arena-2 county marginals (A1.3).

A1.3 seeding class for the transfer arena, built under the owner's approved
marginal set of 2026-07-19 (docs/internal/TRANSFER_MARGINALS_PROPOSAL.md +
rulings). Inputs are PUBLISHED pre-trial statistics fetched with full
provenance into ``data/transfer_marginals/raw/`` (SCB PxWeb; every total
independently re-verified). Every number derived from the reweighted
population carries the METHOD-TRANSFER label (A1.3).

Margins implemented (targets recomputed here from the raw responses, never
hand-typed):
  M1  age band x sex, county 2005 (bands 0-17 / 18-29 / 30-44 / 45-64 / 65+;
      persona sex joined harness-side from the raw PSRC persons file).
  M2  home central share: City K ring group {core,inner} (the cordon
      interior, ``cityk_cordon.cordon_rings``) targeted at the central-
      municipality population share (county statistics, 2005). FLAGGED PROXY: parish-grade 2005
      interior population is not in the open API; the owner rules on this
      proxy before the population build is used.
  M3  employment x central workplace: employed share of county pop 16+
      (RAMS 2004 / BefolkningNy 16+), and work_ring in {core,inner} among
      employed targeted at the register-statistics work-in-central-municipality share.
  M4  household cars 0/1/2+: the PRE-DECLARED fallback transform (owner
      ruling 3): exponential tilting of the PSRC household car-count
      distribution to the published cars-per-capita (TK1001 2005 county
      cars / county pop), PSRC joint otherwise held fixed, labeled.
  M5  income: EXCLUDED from raking, recorded — quartile targets are
      rank-preserving no-ops without an absolute currency mapping; an
      owner pin (PPP rule) would be required to make it binding.

Algorithm: household-atomic multiplicative raking (all members share the
household weight; per-margin cell factors applied to households via member
counts), iterated to a tolerance on the worst absolute margin gap; weights
trimmed at the p99 ratio and renormalized; achieved-vs-target gaps for
EVERY cell recorded in the manifest — the trim means margins are not met
exactly, and the manifest shows by how much. ESS = (sum w)^2 / sum w^2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from grounding import seeding
from grounding.adapters import psrc
from world.geometry import ZONE_RING

RAW = Path("data/transfer_marginals/raw")
CENTRAL_RINGS = ("core", "inner")  # cityk_cordon.cordon_rings
AGE_BANDS = ((0, 17), (18, 29), (30, 44), (45, 64), (65, 200))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_of(age: int) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi if hi < 200 else 'up'}"
    raise ValueError(age)


# ---------------------------------------------------------------------------
# targets from the raw published responses
# ---------------------------------------------------------------------------

def load_targets() -> dict:
    age_sex = json.loads((RAW / "population_age_sex_raw.json").read_text())
    by = defaultdict(float)
    tot = 0.0
    pop16 = 0.0
    for row in age_sex["data"]:
        # key: [region, age, sex(, year)] — age like "0".."100+", sex 1=M 2=F
        key = row["key"]
        age = int(str(key[1]).rstrip("+"))
        sex = {"1": "M", "2": "F"}[str(key[2])]
        v = float(row["values"][0])
        by[(band_of(age), sex)] += v
        tot += v
        if age >= 16:
            pop16 += v
    m1 = {k: v / tot for k, v in by.items()}

    muni = json.loads((RAW / "population_parish_raw.json").read_text())
    stock_muni = sum(float(r["values"][0]) for r in muni["data"]
                     if str(r["key"][0]) == "0180")
    m2_central = stock_muni / tot

    rams = json.loads((RAW / "rams_commuting_raw.json").read_text())
    employed = 0.0
    work_central = 0.0
    for row in rams["data"]:
        key = row["key"]  # [home_muni, work_muni, sex(, year)]
        sex = str(key[2])
        if sex not in ("1", "2"):     # skip the 'total' sex category
            continue
        v = row["values"][0]
        if v in ("..", "-", ""):
            continue
        v = float(v)
        employed += v
        if str(key[1]) == "0180":
            work_central += v
    m3_employed = employed / pop16
    m3_work_central = work_central / employed

    veh = json.loads((RAW / "vehicles_raw.json").read_text())
    county_cars = max(float(r["values"][0]) for r in veh["data"])
    m4_cars_per_capita = county_cars / tot

    return {
        "county_pop_2005": tot,
        "county_pop16_2005": pop16,
        "m1_age_sex": {f"{b}|{s}": v for (b, s), v in sorted(m1.items())},
        "m2_central_home_share": m2_central,
        "m3_employed_share_of_16plus": m3_employed,
        "m3_work_central_share_of_employed": m3_work_central,
        "m4_cars_per_capita": m4_cars_per_capita,
        "central_muni_pop": stock_muni,
        "rams_employed_residents": employed,
        "county_cars": county_cars,
    }


# ---------------------------------------------------------------------------
# persona table (with harness-side gender join)
# ---------------------------------------------------------------------------

def persona_table() -> pd.DataFrame:
    ds = psrc.load_or_build()
    pidx = seeding.persona_index(ds)
    raw = pd.read_csv("data/psrc/hts_persons_2017_2025_v2026.1.csv",
                      usecols=["person_id", "gender"], dtype=str)
    raw = raw.drop_duplicates("person_id")
    pidx = pidx.merge(raw, on="person_id", how="left")
    g = pidx["gender"].fillna("").str.lower()
    pidx["sex"] = np.where(g.str.startswith("m"), "M",
                           np.where(g.str.startswith("f"), "F", "X"))
    pidx["age_band"] = pidx["age"].astype(int).map(band_of)
    pidx["home_ring"] = pidx["home_zone"].map(lambda z: ZONE_RING.get(z, "east_water"))
    pidx["work_ring"] = pidx["work_zone"].map(lambda z: ZONE_RING.get(z) if isinstance(z, str) else None)
    pidx["home_central"] = pidx["home_ring"].isin(CENTRAL_RINGS)
    pidx["work_central"] = pidx["work_ring"].isin(CENTRAL_RINGS)
    pidx["cars_cat"] = pidx["household_cars"].fillna(0).astype(int).clip(upper=2)
    pidx["age16"] = pidx["age"].astype(int) >= 16
    pidx["employed"] = pidx["employed"].astype(bool)
    return pidx


def m4_household_targets(pidx: pd.DataFrame, cars_per_capita: float) -> dict:
    """Pre-declared fallback (owner ruling 3): exponential tilting of the
    PSRC household car-category distribution to the published cars-per-
    capita, household sizes held fixed."""
    hh = pidx.drop_duplicates("household_id")[
        ["household_id", "cars_cat", "household_size"]].copy()
    hh["size"] = hh["household_size"].fillna(1).clip(lower=1)
    p = hh.groupby("cars_cat").size() / len(hh)
    persons_per_hh = float(hh["size"].mean())
    target_cars_per_hh = cars_per_capita * persons_per_hh

    def cars_per_hh(lam: float) -> float:
        q = {c: p[c] * np.exp(lam * c) for c in p.index}
        z = sum(q.values())
        return sum(c * v for c, v in q.items()) / z

    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cars_per_hh(mid) < target_cars_per_hh:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    q = {c: float(p[c] * np.exp(lam * c)) for c in p.index}
    z = sum(q.values())
    return {"lambda": lam, "target_cars_per_hh": target_cars_per_hh,
            "shares": {str(c): v / z for c, v in q.items()}}


# ---------------------------------------------------------------------------
# household-atomic raking
# ---------------------------------------------------------------------------

def rake(pidx: pd.DataFrame, targets: dict, m4: dict, *,
         max_iter=200, tol=5e-4, trim_pct=99.0, m1_adults_only=False):
    hh_ids = pidx["household_id"].values
    uniq_hh, hh_pos = np.unique(hh_ids, return_inverse=True)
    n_hh = len(uniq_hh)
    w_hh = np.ones(n_hh)

    # margin definitions: (person_mask_per_cell, target_share, base_mask)
    margins = []
    n = len(pidx)
    sexed = pidx["sex"] != "X"
    if m1_adults_only:
        # adult-only age x sex margins: children follow their households
        # (resolves the household-atomic coupling between the 0-17 up-weight
        # and the 18-29 down-weight, which otherwise degenerates ESS)
        base_m1 = (sexed & (pidx["age"].astype(int) >= 18)).values
        adult = {c: v for c, v in targets["m1_age_sex"].items()
                 if not c.startswith("0-17")}
        z = sum(adult.values())
        for cell, share in adult.items():
            b, s = cell.split("|")
            mask = (pidx["age_band"] == b) & (pidx["sex"] == s)
            margins.append((f"m1:{cell}", mask.values, share / z, base_m1))
    else:
        sexed_frac = float(sexed.mean())
        for cell, share in targets["m1_age_sex"].items():
            b, s = cell.split("|")
            mask = (pidx["age_band"] == b) & (pidx["sex"] == s)
            # renormalize target over the sexed subpopulation
            margins.append((f"m1:{cell}", mask.values,
                            share * sexed_frac, sexed.values))
    m2 = targets["m2_central_home_share"]
    margins.append(("m2:central", pidx["home_central"].values, m2,
                    np.ones(n, bool)))
    margins.append(("m2:noncentral", (~pidx["home_central"]).values, 1 - m2,
                    np.ones(n, bool)))
    e = targets["m3_employed_share_of_16plus"]
    a16 = pidx["age16"].values
    emp = pidx["employed"].values & a16
    margins.append(("m3:employed", emp, e, a16))
    margins.append(("m3:notemployed", (~pidx["employed"].values) & a16,
                    1 - e, a16))
    wc = targets["m3_work_central_share_of_employed"]
    margins.append(("m3:workcentral", emp & pidx["work_central"].values,
                    wc, emp))
    margins.append(("m3:worknoncentral", emp & ~pidx["work_central"].values,
                    1 - wc, emp))
    hh_first = ~pd.Series(hh_ids).duplicated().values
    for c, share in m4["shares"].items():
        mask = hh_first & (pidx["cars_cat"] == int(c)).values
        margins.append((f"m4:cars{c}", mask, share, hh_first))

    def person_w():
        return w_hh[hh_pos]

    def gap_report():
        w = person_w()
        out = {}
        for name, mask, tgt, base in margins:
            cur = w[mask].sum() / max(w[base].sum(), 1e-12)
            out[name] = {"target": tgt, "achieved": cur, "gap": cur - tgt}
        return out

    it_hist = []
    for it in range(max_iter):
        worst = 0.0
        for name, mask, tgt, base in margins:
            w = person_w()
            cur = w[mask].sum() / max(w[base].sum(), 1e-12)
            if cur <= 0 or tgt <= 0:
                continue
            f = tgt / cur
            # apply to households in proportion to their members in the cell
            cnt = np.bincount(hh_pos, weights=mask.astype(float),
                              minlength=n_hh)
            size = np.bincount(hh_pos, minlength=n_hh).astype(float)
            w_hh *= f ** (cnt / size)
            worst = max(worst, abs(cur - tgt))
        it_hist.append(worst)
        if worst < tol:
            break
    # trim + renormalize
    w = w_hh / w_hh.mean()
    cap = np.percentile(w, trim_pct)
    n_trimmed = int((w > cap).sum())
    w = np.minimum(w, cap)
    w = w / w.mean()
    w_hh[:] = w
    pw = person_w()
    ess_p = float(pw.sum() ** 2 / (pw ** 2).sum())
    ess_h = float(w.sum() ** 2 / (w ** 2).sum())
    return {
        "weights_hh": dict(zip(uniq_hh.tolist(), w.tolist())),
        "person_weights": pw,
        "iterations": len(it_hist),
        "converged": it_hist[-1] < tol,
        "final_worst_gap_pre_trim": it_hist[-1],
        "trim": {"pct": trim_pct, "cap": float(cap), "n_hh_trimmed": n_trimmed},
        "ess_person": ess_p, "ess_household": ess_h,
        "n_person": n, "n_household": n_hh,
        "achieved_after_trim": gap_report(),
        "weight_stats": {
            "min": float(w.min()), "p50": float(np.median(w)),
            "p90": float(np.percentile(w, 90)),
            "p99": float(np.percentile(w, 99)), "max": float(w.max()),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/transfer_reweight")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    pidx = persona_table()
    m4 = m4_household_targets(pidx, targets["m4_cars_per_capita"])
    res = rake(pidx, targets, m4)

    with open(out / "weights.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["persona_id", "household_id", "weight"])
        for pid, hh, w in zip(pidx["persona_id"], pidx["household_id"],
                              res["person_weights"]):
            wcsv.writerow([pid, hh, f"{w:.8f}"])

    manifest = {
        "eval": "METHOD-TRANSFER reweighting (A1.3; owner-approved set "
                "2026-07-19, dims 1-5, #6 dropped)",
        "sources": {p.name: _sha256(p) for p in sorted(RAW.glob("*_raw.json"))},
        "targets": targets,
        "m4_transform": m4,
        "m5_income": "EXCLUDED from raking (recorded): quartile targets are "
                     "rank-preserving no-ops without an absolute currency "
                     "mapping; owner pin required to bind",
        "m2_flag": "PROXY — parish-grade 2005 cordon-interior population is "
                   "not in the open API; central cell = the arena-2 county "
                   "municipality share, mapped to City K rings "
                   f"{CENTRAL_RINGS}. Owner rules before the population "
                   "build is used.",
        "raking": {k: v for k, v in res.items()
                   if k not in ("weights_hh", "person_weights")},
        "label": "METHOD-TRANSFER (carried on every derived number)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"converged": bool(res["converged"]),
                      "iterations": res["iterations"],
                      "ess_person": round(res["ess_person"], 1),
                      "ess_household": round(res["ess_household"], 1),
                      "weight_stats": res["weight_stats"]}, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
