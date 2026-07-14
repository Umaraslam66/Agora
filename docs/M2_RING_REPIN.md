# M2 residence-ring re-pin — gate record

> **Dated note — 2026-07-14, recorded at M2.** Fulfils the standing mandate
> in pre-registration **Amendment A2.6**: *"The residence-ring assignment
> currently uses the core-jurisdiction proxy and is PROVISIONAL: it must be
> re-pinned to the committed tract→zone map at M2 by a dated note, without
> moving any sealed bar."* Reviewed and finalized by the architect on the
> same date.
>
> This is a **dated note recording a definition swap, not a bar change.** No
> numeric bar, band, fold rule, or scoring procedure in §3–§4 of the
> pre-registration or in Amendment A2 is altered. The E1 protected-segment
> *axis* is unchanged; only the input that assigns each household to the
> catchment vs. remainder residence band is upgraded from a jurisdiction-name
> proxy to the committed tract→zone map.

## 1. What is being re-pinned

The E1 protected segments (pre-registration §3 E1; taxonomy `m0-1.0`) are
`income_band × car_ownership × residence_band`. The residence band is
`catchment` for households whose home ring feeds the priced facility
(`grounding.taxonomy.CATCHMENT_RINGS = {"core", "inner"}`) and `remainder`
otherwise (`grounding.taxonomy.residence_band`).

At M0 the ring assignment was a deliberately provisional placeholder, because
no tract→zone map existed yet:

- **Provisional proxy (M0, `docs/internal/m0_bars/m0_common.py`):**
  `home_jurisdiction == "Seattle"` → ring `core` (⇒ `catchment`); every other
  jurisdiction → ring `outer` (⇒ `remainder`). A single city-name string test.

A2.6 requires this be replaced, at M2, by the committed geographic map.

## 2. The committed tract→zone map

**Artifacts (committed):** `grounding/tract_zone_map.json` (map `m2-1.0`),
`grounding/zone_map.py` (loader: `zone_of_tract`, `ring_of_tract`,
`ring_of_household`), `grounding/build_tract_zone_map.py` (reproducible
builder), `tests/test_zone_map.py`.

**Source geography (harness-side, gitignored `data/geo/`).** US Census Bureau
**2020 Gazetteer**, census tracts, Washington (state FIPS 53); per-tract
interior-point centroids `INTPTLAT` / `INTPTLONG`. Downloaded 2026-07-14,
SHA-256 `ddbb686c…c32a1ec` (full provenance in `data/geo/README.md`). Household
weighting from the survey households table (pooled 2017 + 2019 waves,
`hh_weight > 0`, ×0.5 wave mass — same convention as the M0 build). The map
covers **all 923 tracts** in the four-county region (King 53033, Kitsap 53035,
Pierce 53053, Snohomish 53061); all 778 distinct survey home tracts fall inside
it.

**Method (deterministic, reproducible).** Each tract centroid is assigned to
one of the five City K v2 rings by explicit geometric rules built around the
real SR 99 corridor, then partitioned within its ring into that ring's zone
codes by a household-weighted recursive median split (contiguous cells of
roughly comparable household count). The masking wall is intact: real place
names and coordinates live only in the builder / gitignored gazetteer; the
committed JSON and every agent-facing surface carry only masked Z-codes
(`grounding/` passes mask-lint).

The rings, in plain geography:

| Ring | Zones | Geography it stands for |
|---|---|---|
| `core` | Z01–Z06 | Center-city Seattle tracts around the corridor's downtown segment (within ~4.2 km of the downtown anchor). |
| `inner` | Z07–Z14 | The rest of the Seattle urban ring feeding the corridor (west of Lake Washington, between the north/south city limits). |
| `outer_north` | Z15–Z20 | Snohomish County + north King suburbs (north along the corridor axis, lat > 47.735). |
| `outer_south` | Z21–Z26 | Pierce County + south King (Renton/Kent/Auburn/Federal Way/… , lat < 47.505). |
| `east_water` | Z27–Z30 | Across-water ring: King's Eastside (east of Lake Washington), **plus Kitsap and the Puget Sound islands** (§3). |

`core ∪ inner` = the corridor catchment = geometric "Seattle proper."

Ring totals (four-county household weight, sampled):

| Ring | Tracts | HH-weight share | Within-ring zone balance (max/min) |
|---|---:|---:|---:|
| core | 51 | 6.31% | 1.33× |
| inner | 128 | 14.35% | 1.21× |
| outer_north | 207 | 22.33% | 1.17× |
| outer_south | 358 | 38.11% | 1.17× |
| east_water | 179 | 18.90% | 1.04× |
| **catchment (core+inner)** | **179** | **20.66%** | — |

## 3. The Kitsap (and island) decision — recorded explicitly

**Kitsap County (53035) → `east_water` (across-water ring), non-corridor ODs.**
Kitsap lies *west* across Puget Sound and is ferry-dependent, not east of Lake
Washington. It is nonetheless mapped to the across-water ring per **M2 spec
D10**, on **structural across-water semantics**: the ring's defining property in
the world layer is that a trip with exactly one end in it is a capacity-limited
*water crossing*, never a corridor trip (`world.geometry.is_water_crossing` /
`is_corridor_od`). Kitsap trips are structurally water crossings, so they belong
to `east_water`; they are non-corridor by construction. The ring name
`east_water` is a masked structural label, not a compass claim.

**Puget Sound islands (Vashon/Maury, unincorporated King) → `east_water`,** by
the identical rationale: ferry-served islands whose every mainland OD is a water
crossing, not a corridor or south-mainland trip. This is a small extension of
the D10 Kitsap call (2 tracts) and, like Kitsap, is non-catchment either way, so
it does not affect any protected-segment count.

## 4. Segment-membership shift: proxy vs. committed map

Population: the segmented household set — **6,319** households (2017+2019,
`hh_weight > 0`), less **421** income-refusal (PNA, excluded from segmented
statistics per A2.6) and **2** null-home-tract households (excluded from the map
side; see §5) = **5,896 households, weight 1,506,068.** Ten protected cells,
guard merge applied (the three `*|car0|remainder` cells merged into one
`car0|remainder` guard cell, per A2.1). Weighted shares use `hh_weight × 0.5`.

| Protected cell | N (proxy) | N (map) | ΔN | Wt.-share proxy | Wt.-share map | Δ (pp) |
|---|---:|---:|---:|---:|---:|---:|
| low\|car0\|catchment | 481 | 481 | 0 | 2.40% | 2.40% | +0.00 |
| low\|car1p\|catchment | 602 | 608 | +6 | 3.98% | 4.09% | +0.11 |
| low\|car1p\|remainder | 424 | 418 | −6 | 20.27% | 20.15% | −0.11 |
| mid\|car0\|catchment | 242 | 244 | +2 | 0.70% | 0.71% | +0.01 |
| mid\|car1p\|catchment | 844 | 852 | +8 | 4.81% | 4.90% | +0.10 |
| mid\|car1p\|remainder | 679 | 671 | −8 | 24.21% | 24.12% | −0.10 |
| high\|car0\|catchment | 221 | 221 | 0 | 0.49% | 0.49% | +0.00 |
| high\|car1p\|catchment | 1369 | 1376 | +7 | 8.52% | 8.52% | +0.01 |
| high\|car1p\|remainder | 932 | 925 | −7 | 30.75% | 30.74% | −0.01 |
| car0\|remainder (guard) | 102 | 100 | −2 | 3.89% | 3.88% | −0.01 |
| **TOTAL** | **5896** | **5896** | **0** | 100.00% | 100.00% | — |

**Catchment membership:** proxy 3,759 households (20.88% wt.) → map 3,782
households (21.11% wt.). **33 households (0.6% of the segmented set; 0.46% of
weight) change cell** under the re-pin; every per-cell weighted share moves by
≤ 0.11 pp. The shift is a boundary refinement: the geometric catchment (Seattle
proper) picks up a few center-adjacent tracts the jurisdiction string missed and
sheds a few jurisdiction-Seattle tracts that sit outside the geometric center
city.

**Consistency check.** The guard cell holds **102** households under the proxy
— identical to the "merged guard cell 102 households" figure sealed in A2.1,
confirming this note reconstructs the M0 segmentation faithfully before applying
the swap.

## 5. Unknown / missing handling

`zone_of_tract` / `ring_of_household` return `None` for a tract that is not in
the committed map or a missing/malformed home-tract value (never a silent
default ring); the caller drops-with-a-logged-count, matching the M0 policy for
records with no usable field. In the segmented set exactly **2** households have
a null `home_tract_2020`; they are dropped from the map side and reported here
rather than imputed. No survey home tract inside the four counties failed to map
(`tests/test_zone_map.py::test_full_four_county_home_tract_coverage`).

## 6. Conclusion (for the architect)

The residence ring is now taken from `home_tract_2020` via the committed
tract→zone map (`grounding.zone_map.ring_of_household`), retiring the M0
core-jurisdiction proxy. **No sealed bar, band, fold, or protocol moves**;
the change is confined to how the `catchment`/`remainder` band is derived, and
its effect on protected-cell membership is ≤ 0.11 pp per cell. Per A2.6 this
note is the required dated record. Downstream E1/E2/E5 harness code MUST
inject `grounding.zone_map.ring_of_household` for residence banding; the
adapter's built-in default remains the M0 proxy solely so its promotion tests
keep validating M0 fidelity, and is superseded for all scoring by this note.

*Reproduce:* `.venv/bin/python grounding/build_tract_zone_map.py --check`
(rebuild ≡ committed JSON), `tests/test_zone_map.py` (coverage, determinism,
vocabulary, catchment banding).
