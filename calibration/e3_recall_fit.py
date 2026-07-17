"""A3.1 E3-primary recall-channel fit + A3.2 price-channel prior (frozen).

HARNESS-SIDE ONLY (calibration/ discipline; carries real survey label
vocabulary — never imported by agent-facing code; mask-lint does not scan
calibration/, exactly as for sr520_target).

WHAT THIS FREEZES (A3.1, fit-once-freeze): the person-level stated-vs-revealed
discrepancy of the five pinned recall families, computed ONCE on the seeding
microdata (2017+2019 waves; family (e) from the archived Public Release 2
attitudinal module, person-joinable 1:1), under the A2.1 fold structure —
every statistic is emitted POOLED and LEAVE-FOLD-k-OUT (k=0..4), so E3
scoring of fold-k personas consumes only out-of-fold R_real values. The
measured R_real values are recorded here at fit time and are NOT re-derivable
after scoring (A3.1). The label->value mappings below are part of the frozen
fit definition.

Pinned families and their frozen discrepancy definitions:
  (a) stated typical commute mode (``work_mode``) vs revealed commute-trip
      mode (trip-weight majority of observed weekday work-purpose trips):
      agreement rate; D_a = 1 - agreement.
  (b) stated commute days/week (``commute_freq``) vs revealed weekday commute
      rate (share of observed weekdays with >=1 work trip, x5):
      D_b = mean(stated - revealed) days/week.
  (c) stated past-30-day mode-frequency bands vs revealed weekday usage, per
      mode family {transit, walk, bike, hired-ride, shared-car}: stated
      any-use prevalence vs observed diary-window prevalence (ratio R_c) and
      the Spearman of the ordinal band vs the person's weekday use rate.
  (d) stated telework days/week (``telecommute_freq``) vs revealed diary-day
      telework minutes: Spearman + gradient (mean minutes per stated band).
  (e) stated residence-factor importance (PR2 ``res_factors_transit`` /
      ``res_factors_walk``, household-level, joined 1:1) vs revealed use:
      Spearman + monotone gradient of person weekday-use prevalence by
      importance level.

A3.2 PRICE PRIOR (recorded, never fitted): revealed toll response = 2-3x
stated (Brownstone & Small 2005), sensitivity band [1.35, 3.0] (Murphy et
al. 2005 floor; tie-cite Loomis 2011; Devarasetty/Burris/Shaw 2012 as the
reason for a band). Its ONLY test is the frozen E3(iii) transfer clause: the
frozen correction must improve blind E4 versus the UNCORRECTED ABLATION —
which therefore MUST be in the BT1 firing set. The mechanical application
point of the price correction is NOT pinned by the sealed text; this module
records the open decision and the architect's recommendation for the owner
to seal before M4 (see PRICE_PRIOR below and the double-counting analysis in
the pre-M4 record).
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from grounding.adapters import psrc

DATA = Path("data/psrc")
PERSONS_CSV = DATA / "hts_persons_2017_2025_v2026.1.csv"
PR2_HH_CSV = DATA / "codebook_2019" / "2017-2019-pr2-1-Household.csv"
PR2_PERSON_CSV = DATA / "codebook_2019" / "2017-2019-pr2-2-Person.csv"
DAYS_CSV = DATA / "hts_days_2017_2025_v2026.1.csv"
TRIPS_CSV = DATA / "hts_trips_2017_2025_v2026.1.csv"

WAVES = ("2017", "2019")

# ---------------------------------------------------------------------------
# frozen label -> value mappings (part of the fit definition)
# ---------------------------------------------------------------------------

#: stated typical commute mode -> frozen five-mode taxonomy. Carpool labels
#: resolve by licence (driver if licensed, else passenger) — the same tie-break
#: collapse_mode applies to drive-class trips without a driver flag.
WORK_MODE_MAP: Dict[str, str] = {
    "Drive alone": "car",
    "Carpool ONLY with other household members": "__carpool__",
    "Carpool with other people not in household (may also include household members)": "__carpool__",
    "Vanpool": "ride",
    "Bus (public transit)": "transit",
    "Private bus or shuttle": "transit",
    "Urban rail (Link light rail, monorail)": "transit",
    "Commuter rail (Sounder, Amtrak)": "transit",
    "Streetcar": "transit",
    "Ferry or water taxi": "transit",
    "Walk, jog, or wheelchair": "walk",
    "Bicycle or e-bike": "bike",
    "Other hired service (Uber, Lyft, or other smartphone-app car service)": "ride",
    "Taxi (e.g., Yellow Cab)": "ride",
    "Motorcycle/moped/scooter": "car",
}

#: stated days/week from a frequency label (families (b), (d)).
DAYS_PER_WEEK_MAP: Dict[str, float] = {
    "6-7 days a week": 6.5,
    "5 days a week": 5.0,
    "4 days a week": 4.0,
    "3 days a week": 3.0,
    "2 days a week": 2.0,
    "1 day a week": 1.0,
    "A few times per month": 0.7,
    "Less than monthly": 0.2,
    "Never": 0.0,
}

#: past-30-day band -> ordinal rank (family (c)); any-use = rank > 0.
FREQ_RANK_MAP: Dict[str, int] = {
    "Never in the past 30 days": 0,
    "Never": 0,
    "Less than monthly": 1,
    "1-3 days in the past month": 1,
    "A few times per month": 1,
    "1 day a week": 2,
    "2-4 days a week": 3,
    "2 days a week": 3,
    "3 days a week": 3,
    "4 days a week": 3,
    "5 days a week": 4,
    "6-7 days a week": 5,
}

#: family (c) mode columns and their revealed counterparts.
FREQ_FAMILIES = {
    "transit": {"col": "transit_freq", "revealed": "mode==transit"},
    "walk": {"col": "walk_freq", "revealed": "mode==walk"},
    "bike": {"col": "bike_freq", "revealed": "mode==bike"},
    "hired_ride": {"col": "tnc_freq", "revealed": "mode_class==Ride Hail"},
    "shared_car": {"col": "carshare_freq", "revealed": "mode_1 carshare label"},
}

#: family (e) Likert importance coding (frozen).
LIKERT_IMPORTANCE_MAP: Dict[str, int] = {
    "Very unimportant": 1,
    "Somewhat unimportant": 2,
    "Neither or N/A": 3,
    "Somewhat important": 4,
    "Very important": 5,
}

#: A3.2 — the declared price prior (a PRIOR, not a fit; A3.2 FORBIDS fitting
#: the price channel locally). Application point: OWNER-DECIDES (see field).
PRICE_PRIOR = {
    "central_range": [2.0, 3.0],
    "sensitivity_band": [1.35, 3.0],
    "anchor": "Brownstone & Small 2005, Transportation Research A 39(4)",
    "floor": "Murphy et al. 2005, Env & Resource Econ 30(3) (1.35); "
             "tie-cite Loomis 2011, J. Econ. Surveys 25(2)",
    "band_reason": "Devarasetty, Burris & Shaw 2012 (Katy Freeway) shows the "
                   "gap can be small — hence a band, not a point",
    "only_test": "E3(iii): frozen correction must improve blind E4 vs the "
                 "UNCORRECTED ABLATION arm at BT1 (ablation in the firing set)",
    "application_point": {
        "status": "OWNER-DECIDES before M4 (not pinned by sealed text)",
        "recommendation": (
            "apply the correction INSIDE the pipeline BEFORE the A4.3 SR 520 "
            "elasticity fit, so the VoT dial fits the residual and BT1 runs "
            "the identical corrected pipeline; the BT1 ablation arm removes "
            "the correction with the elasticity UNCHANGED. Applying it "
            "post-SR520-fit on any channel the fit absorbed double-counts "
            "magnitude (the imposed-magnitude trap)."
        ),
    },
}


# ---------------------------------------------------------------------------
# small stats helpers (no scipy.stats dependency)
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _fold_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """'pooled' plus leave-fold-k-out subsets keyed 'excl_fold{k}'."""
    out = {"pooled": df}
    for k in range(5):
        out[f"excl_fold{k}"] = df[df["fold"] != k]
    return out


def _per_wave(df: pd.DataFrame, fn) -> Dict[str, object]:
    return {w: fn(df[df["survey_year"] == w]) for w in WAVES}


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------

def _load_frames():
    from grounding import seeding

    dataset = psrc.load_or_build()
    persons = pd.read_csv(
        PERSONS_CSV, dtype=str, low_memory=False,
        usecols=["person_id", "household_id", "survey_year", "work_mode",
                 "commute_freq", "transit_freq", "walk_freq", "bike_freq",
                 "tnc_freq", "carshare_freq", "telecommute_freq"],
    )
    persons = persons[persons.survey_year.isin(WAVES)].copy()
    persons["fold"] = persons["household_id"].map(psrc.fold_id)

    can_drive = dataset.persons[["person_id"]].copy()
    can_drive["person_id"] = can_drive["person_id"].astype(str)
    if "can_drive_bool" in dataset.persons:
        can_drive["can_drive_bool"] = dataset.persons["can_drive_bool"].astype(bool).to_numpy()
    else:
        can_drive["can_drive_bool"] = True
    persons = persons.merge(can_drive, on="person_id", how="left")
    persons["can_drive_bool"] = persons["can_drive_bool"].fillna(True)

    # enriched weekday trips carry purpose + the collapsed five-mode + w_trip
    # (the same frames the evidence builder consumed — the A3.1 pairs live on
    # exactly what generation saw)
    wt = seeding.enriched_trips(dataset).copy()
    wt["person_id"] = wt["person_id"].astype(str)
    wt = wt.rename(columns={"w_trip": "trip_weight"})
    pdays = dataset.person_days.copy()
    pdays["person_id"] = pdays["person_id"].astype(str)

    raw_trips = pd.read_csv(
        TRIPS_CSV, dtype=str, low_memory=False,
        usecols=["person_id", "survey_year", "mode_class", "mode_1"],
    )
    raw_trips = raw_trips[raw_trips.survey_year.isin(WAVES)]

    days = pd.read_csv(
        DAYS_CSV, dtype=str, low_memory=False,
        usecols=["person_id", "survey_year", "telework_time"],
    )
    days = days[days.survey_year.isin(WAVES)]
    return dataset, persons, wt, pdays, raw_trips, days


# ---------------------------------------------------------------------------
# family (a): stated typical commute mode vs revealed commute mode
# ---------------------------------------------------------------------------

def fit_family_a(persons: pd.DataFrame, wt: pd.DataFrame) -> dict:
    stated = persons[["person_id", "survey_year", "fold", "work_mode",
                      "can_drive_bool"]].copy()

    def map_mode(row):
        m = WORK_MODE_MAP.get(row.work_mode)
        if m == "__carpool__":
            return "car" if row.can_drive_bool else "ride"
        return m

    stated["stated_mode"] = stated.apply(map_mode, axis=1)
    stated = stated.dropna(subset=["stated_mode"])

    # trip-weighted trip-level agreement — the definition that reproduces the
    # A3.1 pinned 0.83-0.87 per-wave window (person-majority agreement is
    # recorded as a secondary reading)
    j = wt[wt["purpose"] == "work"].merge(
        stated[["person_id", "stated_mode", "fold", "survey_year"]],
        on="person_id", how="inner", suffixes=("", "_p"),
    )

    def stats(df):
        n_persons = int(df.person_id.nunique())
        if not len(df):
            return {"n_persons": 0}
        match = (df["mode"] == df.stated_mode)
        w = df.trip_weight
        agree_w = float((match * w).sum() / w.sum())
        maj = (
            df.groupby(["person_id", "mode"])["trip_weight"].sum().reset_index()
            .sort_values(["person_id", "trip_weight", "mode"],
                         ascending=[True, False, True])
            .drop_duplicates("person_id").rename(columns={"mode": "revealed_mode"})
        )
        pm = maj.merge(df[["person_id", "stated_mode"]].drop_duplicates("person_id"),
                       on="person_id")
        return {
            "n_persons": n_persons,
            "agreement_trip_weighted": agree_w,
            "discrepancy": 1.0 - agree_w,
            "agreement_person_majority": float(
                (pm.stated_mode == pm.revealed_mode).mean()),
        }

    return {
        "definition": "trip-weight-weighted agreement of observed weekday work "
                      "trips' mode with the stated typical commute mode; "
                      "D_a = 1 - agreement (person-majority agreement recorded "
                      "as a secondary reading)",
        "per_wave": _per_wave(j, stats),
        **{k: stats(v) for k, v in _fold_frames(j).items()},
    }


# ---------------------------------------------------------------------------
# family (b): stated commute days/week vs revealed weekday commute rate
# ---------------------------------------------------------------------------

def fit_family_b(persons: pd.DataFrame, wt: pd.DataFrame, pdays: pd.DataFrame) -> dict:
    stated = persons[["person_id", "survey_year", "fold", "commute_freq"]].copy()
    stated["stated_days"] = stated["commute_freq"].map(DAYS_PER_WEEK_MAP)
    stated = stated.dropna(subset=["stated_days"])

    obs_days = pdays.groupby("person_id").size().rename("n_weekdays")
    work_days = (
        wt[wt["purpose"] == "work"].groupby("person_id")["daynum"].nunique()
        if "daynum" in wt else
        wt[wt["purpose"] == "work"].groupby("person_id").size().clip(upper=None)
    )
    work_days = work_days.rename("n_commute_days")
    j = stated.join(obs_days, on="person_id").join(work_days, on="person_id")
    j = j.dropna(subset=["n_weekdays"])
    j["n_commute_days"] = j["n_commute_days"].fillna(0.0)
    j["n_commute_days"] = np.minimum(j["n_commute_days"], j["n_weekdays"])
    j["revealed_days"] = 5.0 * j["n_commute_days"] / j["n_weekdays"]
    j["diff"] = j["stated_days"] - j["revealed_days"]

    def stats(df):
        return {
            "n": int(len(df)),
            "stated_mean": float(df.stated_days.mean()) if len(df) else None,
            "revealed_mean": float(df.revealed_days.mean()) if len(df) else None,
            "discrepancy_days_per_week": float(df["diff"].mean()) if len(df) else None,
        }

    return {
        "definition": "stated commute days/week minus 5 x (observed weekdays "
                      "with >=1 work trip / observed weekdays), person-level mean",
        "per_wave": _per_wave(j, stats),
        **{k: stats(v) for k, v in _fold_frames(j).items()},
    }


# ---------------------------------------------------------------------------
# family (c): 30-day mode-frequency bands vs revealed weekday usage
# ---------------------------------------------------------------------------

def fit_family_c(persons: pd.DataFrame, wt: pd.DataFrame, pdays: pd.DataFrame,
                 raw_trips: pd.DataFrame) -> dict:
    obs_days = pdays.groupby("person_id").size().rename("n_weekdays")

    def revealed_use(mode_key: str) -> pd.Series:
        if mode_key == "hired_ride":
            sel = raw_trips[raw_trips["mode_class"] == "Ride Hail"]
        elif mode_key == "shared_car":
            sel = raw_trips[raw_trips["mode_1"].fillna("").str.startswith("Carshare service")]
        else:
            sel = wt[wt["mode"] == mode_key]
        return sel.groupby("person_id").size().rename("n_use")

    out = {}
    for key, spec in FREQ_FAMILIES.items():
        col = spec["col"]
        stated = persons[["person_id", "survey_year", "fold", col]].copy()
        stated["rank"] = stated[col].map(FREQ_RANK_MAP)
        stated = stated.dropna(subset=["rank"])
        j = stated.join(obs_days, on="person_id").join(revealed_use(key), on="person_id")
        j = j.dropna(subset=["n_weekdays"])
        j["n_use"] = j["n_use"].fillna(0.0)
        j["use_rate"] = j["n_use"] / j["n_weekdays"]
        j["observed_any"] = j["n_use"] > 0
        j["stated_any"] = j["rank"] > 0

        def stats(df):
            n = len(df)
            if not n:
                return {"n": 0}
            sp, op = float(df.stated_any.mean()), float(df.observed_any.mean())
            return {
                "n": int(n),
                "stated_any_use_prevalence": sp,
                "observed_any_use_prevalence": op,
                "prevalence_ratio": sp / op if op > 0 else None,
                "spearman_rank_vs_use_rate": _spearman(
                    df["rank"].to_numpy(float), df["use_rate"].to_numpy(float)),
            }

        out[key] = {
            "stated_column": col,
            "revealed": spec["revealed"],
            "per_wave": _per_wave(j, stats),
            **{k: stats(v) for k, v in _fold_frames(j).items()},
        }
    return out


# ---------------------------------------------------------------------------
# family (d): stated telework days/week vs revealed diary telework minutes
# ---------------------------------------------------------------------------

def _minutes_of(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fit_family_d(persons: pd.DataFrame, days: pd.DataFrame) -> dict:
    stated = persons[["person_id", "survey_year", "fold", "telecommute_freq"]].copy()
    stated["stated_days"] = stated["telecommute_freq"].map(DAYS_PER_WEEK_MAP)
    stated = stated.dropna(subset=["stated_days"])

    d = days.copy()
    d["minutes"] = d["telework_time"].map(_minutes_of)
    parse_coverage = float(d["minutes"].notna().mean())
    mins = d.dropna(subset=["minutes"]).groupby("person_id")["minutes"].mean().rename("mean_minutes")
    j = stated.join(mins, on="person_id")
    j = j.dropna(subset=["mean_minutes"])

    def stats(df):
        n = len(df)
        if not n:
            return {"n": 0}
        return {
            "n": int(n),
            "spearman_stated_days_vs_mean_minutes": _spearman(
                df["stated_days"].to_numpy(float), df["mean_minutes"].to_numpy(float)),
            "gradient_mean_minutes_by_stated_days": {
                str(k): float(v) for k, v in
                df.groupby("stated_days")["mean_minutes"].mean().items()
            },
        }

    return {
        "definition": "stated telework days/week vs mean revealed diary-day "
                      "telework minutes (Spearman + gradient)",
        "telework_time_parse_coverage": parse_coverage,
        "per_wave": _per_wave(j, stats),
        **{k: stats(v) for k, v in _fold_frames(j).items()},
    }


# ---------------------------------------------------------------------------
# family (e): PR2 residence-factor importance vs revealed use
# ---------------------------------------------------------------------------

def fit_family_e(persons: pd.DataFrame, wt: pd.DataFrame, pdays: pd.DataFrame) -> dict:
    hh = pd.read_csv(PR2_HH_CSV, dtype=str, low_memory=False,
                     usecols=lambda c: c in ("household_id", "hhid",
                                             "res_factors_transit", "res_factors_walk"))
    id_col = "household_id" if "household_id" in hh.columns else "hhid"
    hh = hh.rename(columns={id_col: "household_id"})
    obs_days = pdays.groupby("person_id").size().rename("n_weekdays")

    out = {}
    for factor, mode in (("res_factors_transit", "transit"), ("res_factors_walk", "walk")):
        if factor not in hh.columns:
            out[factor] = {"error": "column absent from PR2 household file"}
            continue
        f = hh[["household_id", factor]].dropna()
        label_map = LIKERT_IMPORTANCE_MAP
        f["importance"] = f[factor].map(LIKERT_IMPORTANCE_MAP)
        f = f.dropna(subset=["importance"])
        j = persons[["person_id", "household_id", "survey_year", "fold"]].merge(
            f[["household_id", "importance"]], on="household_id", how="inner")
        use = wt[wt["mode"] == mode].groupby("person_id").size().rename("n_use")
        j = j.join(obs_days, on="person_id").join(use, on="person_id")
        j = j.dropna(subset=["n_weekdays", "importance"])
        j["any_use"] = j["n_use"].fillna(0.0) > 0

        def stats(df):
            n = len(df)
            if not n:
                return {"n": 0}
            return {
                "n": int(n),
                "spearman_importance_vs_any_use": _spearman(
                    df["importance"].to_numpy(float),
                    df["any_use"].to_numpy(float)),
                "gradient_any_use_by_importance": {
                    str(k): float(v) for k, v in
                    df.groupby("importance")["any_use"].mean().items()
                },
            }

        out[factor] = {
            "revealed_mode": mode,
            "importance_coding": label_map,
            "per_wave": _per_wave(j, stats),
            **{k: stats(v) for k, v in _fold_frames(j).items()},
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/e3_recall_fit")
    ap.add_argument("--freeze", action="store_true",
                    help="write calibration/e3_fit_manifest.json")
    args = ap.parse_args(argv)

    dataset, persons, wt, pdays, raw_trips, days = _load_frames()
    manifest = {
        "amendment": "01_PREREGISTRATION.md section 7 A3.1 (recall, FITTED) + "
                     "A3.2 (price, PRIOR)",
        "date": date.today().isoformat(),
        "fit_window": "2017+2019 waves (calibration window; pre-wall); family "
                      "(e) from the archived PR2 attitudinal module",
        "fold_structure": "A2.1 five household-atomic folds; every statistic "
                          "emitted pooled + leave-fold-k-out",
        "label_mappings": {
            "work_mode": WORK_MODE_MAP,
            "days_per_week": DAYS_PER_WEEK_MAP,
            "freq_rank": FREQ_RANK_MAP,
        },
        "families": {
            "a_typical_commute_mode": fit_family_a(persons, wt),
            "b_commute_days_per_week": fit_family_b(persons, wt, pdays),
            "c_mode_frequency_bands": fit_family_c(persons, wt, pdays, raw_trips),
            "d_telework": fit_family_d(persons, days),
            "e_residence_factors": fit_family_e(persons, wt, pdays),
        },
        "price_prior": PRICE_PRIOR,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    fam = manifest["families"]
    print("a: agreement", fam["a_typical_commute_mode"]["pooled"])
    print("b:", fam["b_commute_days_per_week"]["pooled"])
    print("c transit:", fam["c_mode_frequency_bands"]["transit"]["pooled"])
    print("e transit:", fam["e_residence_factors"].get("res_factors_transit", {}).get("pooled"))

    if args.freeze:
        Path("calibration/e3_fit_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str))
        print("froze calibration/e3_fit_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
