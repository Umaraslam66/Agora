# DRAFT — M0 data inventory, pending architect review, 2026-07-14

**Scope.** This is the M0 data-audit deliverable: what is fetchable today (2026-07-14), what needs
an application, and the published truth/anchor numbers needed to set the `[M0]` numeric bars in
`01_PREREGISTRATION.md`. This document is **harness-side**: Stockholm references are allowed here.
Agent-facing code never reads `docs/` (import-boundary test). The truth numbers in the Appendix are
destined for `evaluation/truth/`, quarantined per pre-registration §2.

**Verification convention.** "Fetchable today: yes (verified)" means an endpoint or document was
actually retrieved on 2026-07-14 during this audit. Numbers not confirmed against a primary source
are marked **[unverified]**.

---

## 1. SCB (Statistics Sweden) — agent skeletons

**What.** Small-area demographics for Stockholm County at DeSO level (Demographic Statistical
Areas, ~6,000 areas nationally, ~1,300 in Stockholm County) and RegSO level (~3,000 areas):
population by age/sex, household composition, income, employment; plus passenger cars per area.
These feed the persona *skeleton* layer (L1 grounding).

**Access status.** Open now. No API key, no registration.

**Fetchable today: YES (verified).** Two live hits on 2026-07-14 against PxWebApi 2.0:

- `GET https://api.scb.se/OV0104/v2beta/api/v2/tables?query=DeSO&lang=en` → valid JSON, 29 tables
  (paged 20/page), including the tables listed below.
- `GET https://api.scb.se/OV0104/v2beta/api/v2/tables/TAB6574/metadata?lang=en` → valid JSON
  metadata: dimensions Region (19,182 values incl. DeSO 2018 and DeSO 2025 codes), Age (18 bands),
  Sex (3), Year (2010–2025).

**Key tables confirmed in the live API response:**

| Table | Content | Years |
|---|---|---|
| TAB6574 | Population per DeSO/RegSO by age and sex | 2010–2025 |
| TAB6568 | Households per DeSO/RegSO by type of household | 2011–2025 |
| TAB6572 / TAB6571 / TAB6569 | Population by region of birth / background / citizenship | 2010–2025 |
| TAB6679 / TAB6683 / TAB6684 | Income: quartile shares, net income structure, equivalised disposable income per DeSO/RegSO | 2011–2024 |
| TAB6685 | At-risk-of-poverty / high economic standard per region | 2011–2024 |
| AM0210G (ArRegDesoSektor) | Employed 15–74 by residence DeSO/RegSO, sex, sector | 2020–2023 |
| TAB6589 (PersBilarDesoN, TK1001Z) | Passenger cars registered on population 31 Dec, by status, per DeSO/RegSO | **2024–2025 only** |

**URLs.**
- API docs: https://www.scb.se/en/services/open-data-api/pxwebapi/ (PxWebApi 2.0 launched Oct 2025, replaces v1)
- Swagger: https://statistikdatabasen.scb.se/api/v2/index.html
- Browse UI: https://www.statistikdatabasen.scb.se/ (e.g. shortcut https://www.statistikdatabasen.scb.se/goto/en/ssd/FolkmDesoAldKon)
- DeSO/RegSO boundary geodata (open): https://www.scb.se/en/services/open-data-api/open-geodata/open-data-for-regso--regional-statistical-areas/
- Terms: https://www.scb.se/om-scb/om-scb.se-och-anvandningsvillkor

**License.** CC0 1.0 for open data in the Statistical Database and open geodata (SCB open-data
pages). Attribution not required; "Source: Statistics Sweden" recommended.

**Rate limits.** PxWebApi 2.0: max 150,000 data cells per GET, max 30 calls per 10 s per IP (SCB
PxWebApi page). SCB's general terms page states 10 calls per 10 s — that figure applies to API v1;
plan conservatively. No key escalation path needed.

**Coverage & gaps.**
- **DeSO series starts 2010/2011.** DeSO was created in 2018 (back-cast to 2010). There is **no
  DeSO data for the 2004–2006 experiment era.** 2006-era small-area statistics exist at SAMS-area
  level only as a **paid custom order** from SCB regional statistics
  (https://www.scb.se/en/services/ordering-data-and-statistics/). Decision needed at M0: seed
  skeletons from earliest open DeSO year (2010/2011) and accept a ~5-year offset, or order
  2005/2006 SAMS-level tables.
- **Car ownership per DeSO exists only for 2024–2025** (TAB6589). Municipality-level vehicle stock
  is available 2002–2025 (table TK1001A `FordonTrafik`,
  https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__TK__TK1001__TK1001A/FordonTrafik/),
  i.e. it covers 2005–2006 but only at municipality resolution. Official vehicle statistics
  authority is Trafikanalys (https://www.trafa.se/en/road-traffic/vehicle-statistics/); SCB
  publishes on commission. Car-per-area seeding below municipality level for 2006 will need either
  the RVU microdata (household car ownership per record) or a custom order.
- Two DeSO vintages coexist (DeSO 2018 vs DeSO 2025, new version from reference year 2024);
  pick one and freeze the crosswalk.
- From reference year 2025, SCB adds controlled random noise (Cell Key Method) to small-area
  tables — prefer ≤2024 vintages for seeding.

**Risk notes.** Low access risk. Main risk is temporal mismatch (2010s skeletons for a 2006
experiment); must be stated in the M0 amendment. The API is in "v2beta" path naming — pin the
base URL and re-verify before pipeline freeze.

---

## 2. Trafiklab — transit layer (GTFS)

**What.** Swedish national and regional GTFS feeds for building City K's transit times.

**Access status.** API key (free self-service registration at trafiklab.se; instant for Bronze
tier). Not hit live during this audit (key required); documentation pages verified today.

**Fetchable today: YES with free key** (documentation verified; no key was created for this audit).

**Datasets.**
- **GTFS Sverige 2** — one national static feed, whole country, less per-operator detail.
  `https://api.resrobot.se/gtfs/sweden.zip?key={apikey}`. Bronze: 1 call/min, 50/month.
  Historical archive at `https://data.samtrafiken.se/trafiklab/gtfs-sverige-2/` going back to
  ~2012 **[earliest year unverified — confirm at first download]**; note stop-ID format change in
  2016. Docs: https://www.trafiklab.se/api/gtfs-datasets/gtfs-sverige-2/
- **GTFS Regional** — per-operator feeds, 50+ operators, **SL (Stockholm) included**; richer detail.
  Static Bronze: 10 calls/min, 50/month (Silver 250/mo, Gold 2,500/mo). Realtime feeds
  (TripUpdates, VehiclePositions, ServiceAlerts) Bronze: 50 calls/min, 30,000/month.
  Docs: https://www.trafiklab.se/api/gtfs-datasets/gtfs-regional/
- **KoDa** — historical archive of GTFS Regional static + GTFS-RT, organized per operator/feed/date.
  **Earliest data: 2020-02-05** (SL VehiclePositions from 2020-12-01). Async archive generation
  (HTTP 202 then 1–60 min wait); no hard rate limit but ≤2–3 parallel downloads requested.
  Docs: https://www.trafiklab.se/api/our-apis/koda/ and
  https://www.trafiklab.se/news/2021/2021-12-14-koda-historical-data/

**License.** GTFS Sverige 2 and GTFS Regional static: **CC0 1.0** (stated on both dataset pages).
KoDa page does not state a license — underlying data is the CC0 GTFS Regional feed, but
**[unverified]**; confirm at key signup. (Some other Trafiklab realtime APIs are CC-BY 4.0 —
attribution footnote is cheap insurance either way.)

**Coverage & gaps — the 2006 problem.**
- **No 2006 GTFS exists anywhere.** GTFS itself only emerged in 2005–2006; Swedish feeds start
  ~2012 (GTFS Sverige 2 archive) and per-operator SL history starts 2020 (KoDa).
- **Earliest proxy:** oldest GTFS Sverige 2 archive snapshot (~2012). Caveat to record in the M0
  amendment: 2012+ SL service ≠ 2006 SL service — Citybanan commuter-rail tunnel (2017), tram/light
  rail extensions, bus network revisions, and the 16 new trial bus lines of 2006 (which existed
  *only* Aug 2005–Dec 2006) are all misrepresented by any modern feed.
- Mitigation consistent with the City K design: the world needs zone-to-zone transit times, not a
  true 2006 timetable. Use a modern feed for network structure, then check headline zone-pair
  times against published 2006 SL travel-time facts where available. Since Eliasson et al. 2009
  attribute at most 0.1 pp of the 22% reduction to the extended bus services (see Appendix row
  A16), transit-supply fidelity is a second-order risk for E4.

**Risk notes.** Low access risk; monthly quotas on Bronze static keys are tight (50/month) but the
feed only needs to be fetched once per world build. The 2006-vs-modern service-level mismatch must
be logged as a known approximation, not silently absorbed.

---

## 3. OpenStreetMap — zones/network

**What.** Zone geometry, road network topology, and cordon geometry for City K.

**Access status.** Open now, no key.

**Fetchable today: YES** (Geofabrik download page verified today; file is public).

**URLs.**
- Geofabrik Sweden extract: https://download.geofabrik.de/europe/sweden.html —
  `sweden-latest.osm.pbf`, ~770 MB, updated daily. Also shapefile variant.
- Overpass API (public instances) for targeted queries.
- Full-history planet dumps: https://planet.openstreetmap.org/ (for attic/history analysis).

**License.** ODbL 1.0 (share-alike for derived databases — fine for research; note obligations if
publishing derived network datasets).

**Coverage & gaps — today's network ≠ 2006 network.**
- Post-2006 changes that matter for the cordon area: **Norra länken tunnel opened 30 Nov 2014**
  (https://en.wikipedia.org/wiki/Norra_l%C3%A4nken); **Essingeleden bypass became tolled and inner-city
  charges raised on 1 Jan 2016** (https://en.wikipedia.org/wiki/Stockholm_congestion_tax);
  Citybanan (rail, 2017). **Södra länken opened Oct 2004**, i.e. it *was* present during the trial
  and must be kept (it is explicitly discussed in Eliasson et al. 2009 as the untolled bypass).
- **Honest assessment of OSM-as-of-2006: not usable.** OSM was founded 2004 and Swedish coverage
  before ~2008 is extremely thin; additionally the 2012 ODbL relicensing removed edits by
  non-agreeing contributors from the retained history. Geofabrik's per-country history extracts
  are on a contributors-only internal server, and public Overpass attic data reaches back only to
  ~2012 **[exact attic start unverified]**. There is no route to a faithful 2006 OSM Sweden.
- **Approximation for 2006:** take today's extract, delete post-2006 infrastructure by hand
  (short, well-documented list: Norra länken and successors; keep Södra länken and Essingeleden
  untolled), and accept that minor-street noise is irrelevant at zone resolution. The City K world
  is zone-based with a volume-delay function, so link-level 2006 fidelity is not load-bearing;
  the cordon definition and the untolled-bypass topology ARE load-bearing and are documented in the
  2009/2012 papers (charging cordon per Fig. 1 of Eliasson et al. 2009, toll zone ~30 km², just
  under 300,000 inhabitants).

**Risk notes.** Main risk is silent anachronism (e.g. modelling Essingeleden as tolled — it was
free in 2006–2007 and its relief-valve role is part of why the bypass absorbed traffic). Keep an
explicit "2006 network diff" file in `world/` listing every manual edit.

---

## 4. Truth series — published cordon counts and effects 2005–2008 (quarantine target: `evaluation/truth/`)

**What.** The published record of cordon-crossing changes across P0 (pre), P1 (trial Jan 3–Jul 31
2006), P2 (off Aug 2006–Jul 2007), P3 (permanent from Aug 1 2007). Sets the E4 target series and
the E6 hysteresis band.

**Access status.** Open now (journal preprints/working papers are free; two paywalled journal
versions have free author/working-paper mirrors).

**Fetchable today: YES (verified).** All three primary PDFs were downloaded and text-extracted
during this audit:

1. **Eliasson, J., Hultkrantz, L., Nerhagen, L., Smidfelt Rosqvist, L. (2009).** "The Stockholm
   congestion-charging trial 2006: Overview of effects." *Transportation Research Part A* 43(3),
   240–250. Free mirror verified: https://f.hubspotusercontent30.net/hubfs/4056033/The%20Stockholm%20congestion%20charging%20trial%202006%20Overview%20of%20effects.pdf
   (publisher page: https://www.sciencedirect.com/science/article/abs/pii/S0965856408001572)
2. **Börjesson, M., Eliasson, J., Hugosson, M., Brundell-Freij, K. (2012).** "The Stockholm
   congestion charges — 5 years on. Effects, acceptability and lessons learnt." *Transport Policy*
   20, 1–12. Free working-paper version (CTS WP 2012:3) verified:
   https://www.transportportal.se/SWoPEc/CTS2012-3.pdf — **contains Table 1 (monthly % reduction
   2006–2011) and Table 2 (externally adjusted reductions), the core truth series.**
3. **Eliasson, J. (2014).** "The Stockholm congestion charges: an overview." CTS Working Paper
   2014:7. Verified: https://www.transportportal.se/swopec/cts2014-7.pdf — mode shift, PT
   ridership, attitude series, forecast-vs-outcome, and the stated-vs-observed (say-do) numbers.

Supporting official evaluation:
4. **Miljöavgiftskansliet (Congestion Charge Secretariat, City of Stockholm) (2006).** "Facts and
   Results from the Stockholm Trial — Final version, December 2006." Mirror located (not the city's
   own server): https://www.mobilservice.ch/admin/data/files/news_section_file/file/1814/facts-about-the-evaluation-of-the-stockholm-trial.pdf
   The original stockholmsforsoket.se archive is offline; treat the mirror as the working copy and
   archive it into `evaluation/truth/` immediately. **[City-hosted canonical URL not located —
   follow up.]**
5. **Börjesson & Kristoffersson (2018)** "The Swedish congestion charges: Ten years on," *TRA* 107,
   35–51; free WP: https://www.transportportal.se/swopec/cts2017-2.pdf (long-run context, 2016
   Essingeleden extension).

**All extracted anchor numbers are in the Appendix table below.** Headline: trial reduction
stabilized at ~20–22% of cordon crossings; off-year residual −5 to −10% vs 2005 (the E6 band);
reinstatement August 2007 at −21%, settling at −18% to −20% (2008–2011).

**License.** Copyrighted publications; numbers/facts are not copyrightable. Store extracted series
+ citations in `evaluation/truth/`, keep PDFs in a private archive directory, do not redistribute.

**Coverage & gaps.**
- The published truth is **aggregate** (cordon totals, purpose splits, a few percent-bands). The
  finest published breakdown is monthly cordon crossings (Börjesson et al. 2012 Table 1) plus
  trip-purpose and peak/off-peak splits. Pre-registration §6 already anticipates this: E4(ii)
  reduces to the finest published breakdown.
- P2 (off period) is the weakest-measured period: Eliasson et al. 2009 note the autumn-2006
  residual was concentrated at two bridges with major roadwork; Börjesson et al. 2012 give the
  5–10% band for Aug 2006–Aug 2007 and note June–July 2007 figures are roadwork-affected. **The E6
  band must therefore be wide (5–10%), and the roadwork confound must be quoted alongside it.**
- Raw count data (vehicles/day per control point) was published in the monthly Stockholmsförsöket
  expert reports; only partially recoverable now. If per-point counts are wanted, contact
  Stockholm Stad trafikkontoret / Trafikverket archives **[route unverified]**.

**Risk notes.** None for access. The main scientific risk is exactly the contamination risk the
pre-registration names: these numbers are famous, hence E5 probes.

---

## 5. Qualitative voice material — published interview/focus-group studies

**What.** Published qualitative studies of how Stockholmers reasoned about the charge; source
material for the persona *voice* layer (masked, of course — reasoning patterns, not place names).

**Access status / fetchability:** mixed; the single most useful item is open access and was
verified today.

| # | Study | What it contains | Access |
|---|---|---|---|
| Q1 | **Henriksson, G., Hagman, O., Andréasson, H. (2011).** "Environmentally Reformed Travel Habits During the 2006 Congestion Charge Trial in Stockholm — A Qualitative Study." *IJERPH* 8(8), 3202–3215. doi:10.3390/ijerph8083202 | 40 participants: 20 with two in-depth interviews each (during + after trial), 20 with travel diaries before/during trial (12 diaries + follow-up interviews). Themes: habit adaptation, motivations, everyday constraints, attitudes; direct quotes (e.g. refusal-to-pay voice). Age 20–70, ~50% habitual drivers / 35% mixed / 15% PT users. | **Open access, verified today:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3166737/ |
| Q2 | **Henriksson, G. (2009).** "What did the Stockholm Trial mean for Stockholmers?" In Gullberg & Isaksson (eds.), *Congestion Taxes in City Traffic: Lessons Learnt from the Stockholm Trial.* Nordic Academic Press. | Ethnographic everyday-life account from interviews and resident travel diaries; how feelings fluctuated before/during/after the trial. Cited by Börjesson et al. 2012 as evidence that "charges do not affect me as much as I thought" (say-do adjacent). | Book purchase/library; ebook exists (ISBN 9789185509232). Not open. |
| Q3 | **Gullberg, A. & Isaksson, K. (eds.) (2009).** *Congestion Taxes in City Traffic.* Nordic Academic Press. | Book-length account incl. detailed political process; several chapters usable as voice/context material. | Purchase/library. |
| Q4 | **Isaksson, K. & Richardson, T. (2009).** "Building legitimacy for risky policies: the cost of avoiding conflict in Stockholm." *TRA* 43(3), 251–257. | Discourse/legitimacy analysis of the trial politics; institutional voice rather than resident voice. | Paywalled; preprint findable **[not verified today]**. |
| Q5 | **Winslott-Hiselius, L., Brundell-Freij, K., Vagland, Å., Byström, C. (2009).** "The development of public attitudes towards the Stockholm congestion trial." *TRA* 43(3), 269–282. | Quantitative attitude surveys + media analysis (the 3%→42% positive-articles series). Anchor source for E3/attitude, less for voice. | Paywalled; key numbers reproduced in Börjesson et al. 2012 (verified). |
| Q6 | **Eliasson, J. & Jonsson, L. (2011).** "The unexpected 'yes': Explanatory factors behind the positive attitudes to congestion charges in Stockholm." *Transport Policy* 18(4), 636–647. | Regression on attitude drivers (self-interest, environmental concern); useful for persona heterogeneity in attitudes. | Paywalled; WP version findable **[not verified today]**. |

**License.** All copyrighted; use as reasoning-pattern source material with citation, never
verbatim redistribution. Mask-lint applies to anything derived from these that reaches agents.

**Risk notes.** Q1 alone is probably sufficient for the voice layer (real quotes, open access).
Book chapters (Q2/Q3) add depth but require purchase; order early if wanted.

---

## 6. RVU microdata — the application route (application text drafted separately, see `docs/M0_TRAFA_RVU_APPLICATION_DRAFT.md`)

**What.** Individual travel-diary records to seed agents one-to-one (L1). Two routes: the national
survey (RES 2005–2006 — the era-matched instrument) and the Stockholm County surveys (1986/2004/2015).

### 6a. National: RES 2005–2006 (Trafikanalys)

- **The survey:** continuous daily fielding autumn 2005–autumn 2006 (i.e. **spans P0 and P1 of the
  experiment**), ~27,000 telephone interviews, 68% response rate, population 6–84, one-day travel
  diary + long trips. Series history: Riks-RVU 1994–1998, RES 1999–2001, RES 2005–2006, RVU
  Sverige 2011–2016, RVU Sweden 2019–. Overview: https://www.trafa.se/en/transportation-trends/travel-survey/
  (published report: SIKA Statistik series, 2007 **[exact report number unverified]**).
- **Who holds microdata:** **Trafikanalys (Trafa)** — successor (2010) to SIKA, the statistics
  authority for the transport domain. This is NOT normally on SCB's MONA platform; it is a direct
  release from Trafa. (SCB MONA is the separate remote-access route for SCB *register* microdata —
  relevant later only if skeletons need register enrichment; process at
  https://www.scb.se/vara-tjanster/bestall-data-och-statistik/mikrodata/.)
- **The actual process** (verified today at https://www.trafa.se/sidor/utlamnande_av_mikrodata/):
  written request to Trafa; legal basis is the statistics-secrecy clause OSL 2009:400 ch. 24 §8
  (release for research/statistics if clearly harmless to data subjects). The request must contain
  a **project description/research plan** and **an ethics-review opinion "if applicable"**. Trafa
  performs (i) a research-purpose classification and (ii) a harm assessment. General rule: **only
  de-identified microdata** is released (no PII; expect coarsened home/work geography — level
  negotiable, likely municipality or SAMS, not coordinates).
- **Contact:** trafikanalys@trafa.se; named microdata contact Eva Pettersson, +46 10 414 42 02;
  travel-survey statisticians Andreas Holmström +46 10 414 42 13, Filippa Egnér +46 10 414 42 24.
  Address: Rosenlundsgatan 54, 118 63 Stockholm.
- **Timeline & cost:** **not stated on the Trafa page.** Comparable Swedish authority releases run
  weeks-to-months; fee, if any, is administrative. **[Both unverified — ask in the application
  cover mail; file at M0 start per project brief.]**
- **Anonymization level risk:** if released geography is municipality-only, zone-level seeding
  inside Stockholm needs an imputation step (documented, pre-registered as part of the synthetic
  stand-in schema).

### 6b. Regional: Stockholm County travel surveys 1986 / 2004 / 2015 (Region Stockholm)

- **The surveys:** 1986 (n=19,511, response rate 80%), 2004 (n=31,348, RR 48%), 2015 (n=40,917,
  RR 35%) — figures from CTS WP 2017:9, "Travel behaviour trends in Stockholm 1985–2015,"
  https://www.transportportal.se/swopec/CTS2017-9.pdf. **RVU 2004 is the prize: a large one-day
  diary sample of the exact population, two years before the toll.**
- **Who holds:** Region Stockholm, trafikförvaltningen (formerly SLL; the 2015 survey was run by a
  purchasing group with Trafikverket, Stockholm Stad, KSL, Länsstyrelsen, TRF). Published 2015
  report: https://miljobarometern.stockholm.se/content/docs/tema/trafik/resvanor/RVU-stockholms-lan-2015.pdf
- **Access route:** no formal microdata portal. Route is a public-records/research request to the
  registrar: **registrator.tf@sl.se** (Region Stockholm trafikförvaltningen), citing research use;
  de-identified extracts have been provided to researchers before (e.g. the CTS 2017 study used all
  three waves). Landing pages: https://www.sll.se/verksamhet/kollektivtrafik/kollektivtrafiken-vaxer-med-stockholm/Resvaneundersokningar/
  and https://www.regionstockholm.se/kollektivtrafik/om-kollektivtrafiken-i-stockholms-lan/statistikrapporter-om-sl-waxholmsbolaget-och-fardtjanst/
  **[process, timeline and cost unverified — no published procedure exists; expect ad-hoc handling.]**
- **1986 wave caveat:** archival format/custodianship unknown **[unverified]**; treat as bonus, not
  plan-of-record.

**Access status (both routes): APPLICATION. Fetchable today: NO.** Consistent with the brief:
file both applications at M0 start and build against published aggregates + the synthetic
stand-in with the identical schema meanwhile. Published aggregate fallbacks fetchable today:
RES 2005–2006 report tables (Trafa site), RVU Sverige 2011–2014 report
(https://www.trafa.se/globalassets/statistik/resvanor/2009-2015/rvu-sverige-2011-2014.pdf), and
the RVU 2015 county report above.

**Risk notes.** This is the critical-path item (pre-registration §6 threat #2). Two independent
routes reduce the risk that L1 seeding falls through entirely; if only aggregated data arrives,
results stay DEV-labeled per pre-registration.

---

## Appendix — Anchor numbers for [M0] bar-setting (cited)

Sources: **[E09]** = Eliasson et al. 2009, *TRA* 43(3):240–250 (page refs = journal pagination).
**[B12]** = Börjesson et al. 2012, CTS WP 2012:3 / *Transport Policy* 20:1–12 (page refs = WP pagination).
**[E14]** = Eliasson 2014, CTS WP 2014:7. **[MAK06]** = Miljöavgiftskansliet Dec 2006 final report.
All three of [E09],[B12],[E14] were downloaded and text-verified on 2026-07-14; quoted figures below
were read directly from the extracted text.

### A. Traffic across the cordon (E4 targets, E6 band)

| # | Quantity | Published value | Source & location |
|---|---|---|---|
| A1 | Pre-trial forecast (planning target) | 10–15% reduction expected during charging hours; traffic models predicted 20–25% | [E09] §3, p.243 |
| A2 | Trial reduction, stabilized | ~**22%** below 12-months-before, after first month; daytime 06:00–19:00 weekdays | [E09] §3 + Fig. 2, p.243 |
| A3 | Trial reduction, monthly | Jan −28%, Feb −23%, Mar −22%, Apr −21%, May −20%, Jun −21%, Jul −24%; **Mar–Jun average −21%** ("overreaction" in Jan–Feb, then stable 20–22%) | [B12] Table 1, p.6 + §3.1, p.7 |
| A4 | Trial reduction by time of day | Morning peak (07–09) −18%; afternoon peak (16–18) −23%; midday (09–15:30) −22% | [E09] §3, p.243–244 |
| A5 | Vehicle-km inside cordon | −16%; major inner-city streets −10/−11%; arterials near cordon −19%; streets outside cordon −5/−6% | [E09] Fig. 3, p.244 |
| A6 | Official trial verdict | Traffic across inner-city cordon reduced ~20% during charged hours; targets met | [MAK06] (summary figure; use [E09]/[B12] as primary) |
| A7 | **P2 off-period residual (THE E6 BAND)** | Aug 2006–Aug 2007 cordon volumes remained **5–10% below 2005**; "traffic volumes immediately rebounded almost to the same level as before the charges – but not quite"; interpreted as persisting habits | [B12] §3.1, p.6–7 |
| A8 | P2 caveat | Autumn 2006 traffic "a few percent lower" than autumn 2005, but concentrated at two roadwork-affected bridges — "uncertain what conclusions can be drawn" | [E09] §3, p.243 |
| A9 | P3 reinstatement, first month | Aug 2007: **−21%** passages (charging hours) vs Aug 2005 — same relative level as during the trial | [B12] §3.1, p.7 |
| A10 | P3 annual series vs 2005 | 2007 (Aug–Dec) −19%; 2008 −18%; 2009 −18%; 2010 −19%; 2011 −20% (weekdays 06–19, excl. July from 2008) | [B12] Table 1, p.6 |
| A11 | Adjusted for external factors (employment, fuel price, car ownership) | −21.4% (2006), −20.9% (2007), −20.7% (2008), −21.9% (2009), −21.7% (2010), −22.3% (2011) → charge effect *grew* slightly | [B12] Table 2, p.8 |
| A12 | Non-exempt traffic, adjusted | −29.7% (2006) to −29.8% (2011), range −27.5% to −30.7%; exempt share of passages 24–29% | [B12] Table 2, p.8 |
| A13 | Reduction by trip purpose (trial) | Commuting car trips across cordon −24% (nearly all → transit; only 1% re-routed); discretionary −22%; professional −15% | [B12] §3.1, p.7 |
| A14 | Implied price elasticity | −0.70 (2006) rising to −0.86 (2011) | [B12] §4, elasticity table, p.11–12 |

### B. Mode shift and public transport (E4-ii targets)

| # | Quantity | Published value | Source & location |
|---|---|---|---|
| B1 | Evicted car trips (trial) | ~92,000/day fewer car trips across cordon; ~45,000 were work/school travel | [E09] §4, p.244–245 |
| B2 | Where they went | ~43,000 of the work/school trips → public transport = **+6% PT trips**; discretionary evicted trips largely vanished/changed destination — "virtually none" to PT; no increase in carpooling or telecommuting | [E09] §4, p.245 |
| B3 | PT ridership change | Transit passengers **+4–5%** (charge effect; cf. forecast +6%) | [E14] §3.7, p.15 & §5.2, p.26 |
| B4 | Contribution of the new trial bus lines | ≤**0.1 pp** of the 22% cordon reduction attributable to extended bus services; ~14,000 trips/day on new buses vs >1M PT trips crossing cordon | [E14] §3.7, p.16, citing [E09] |
| B5 | Baseline mode split (context, not a shock target) | Transit share 60–65% of motorized person-trips county-wide (peak: ~80% to inner city) | [E14] §2, p.4 |
| B6 | Congestion outcome | Queue times reduced 30–50%; circumferential traffic did **not** increase | [E14] §5.2, p.27; [E09] §3 |

### C. Attitudes / say-do (E3 anchors)

| # | Quantity | Published value | Source & location |
|---|---|---|---|
| C1 | Support before trial | Spring 2004 & spring 2005: **40%** would "probably/most likely" vote yes; fell to **36%** just before trial start | [B12] §5.1, p.16. ([E14] Fig. 8 series gives 43% → 34% — different poll series; carry both, flag the discrepancy) |
| C2 | Support during trial | Rose to **52%** once trial started | [B12] §5.1, p.16 |
| C3 | Referendum, 17 Sep 2006 | **53% yes** (City of Stockholm, valid votes, blanks excluded). Counting all municipal referenda held in the county: majority against — but selection-skewed (only anti-charge-leaning municipalities held votes) | [B12] §2, p.4 & §5.1, p.17; [E14] §2.4, p.7 |
| C4 | Support after reinstatement | Dec 2007: **66%** ([B12]) / 65% ([E14]); Aug 2009: ≈**74%** (rephrased question); May 2011: >**70%** (county-wide); late 2013: **72%** (excl. don't-know) | [B12] §5.1, p.17; [E14] §4, p.17 |
| C5 | Media series (attitude proxy) | Positive trial articles 3% (autumn 2005) → 42% (spring 2006); negative 39% → 22% | [B12] §5.1, p.16–17, citing Winslott-Hiselius et al. 2009 |
| C6 | **Stated vs observed behavior (THE E3 ANCHOR)** | Surveys (fall 2004, fall 2005, spring 2006) of self-reported adaptation ⇒ equivalent aggregate reduction of only **5–10%**, vs observed **~30%** reduction of private car trips across cordon ⇒ **~3/4 of the actual behavioral change went unnoticed/unreported by travelers**. Direction: people report *less* change than they enact | [E14] §5.2, p.26–27; same ¾ figure at §3.6, p.15 |
| C7 | Behavioral instability context (habit modelling) | ~2/3 of drivers crossing the cordon on a given day are "occasional" (≤3 days/week); 20–25% of workforce changes jobs and 15–20% of population moves between any two years | [E14] §3.6, p.15 |
| C8 | Forecast-vs-outcome (falsification-arm benchmark) | Model predicted −17% peak / −16% charged-period crossings; actual −19% / −20%. Predicted PT +6%; actual +4–5% | [E14] §5.2, p.26, citing Eliasson et al. 2013 |

### Suggested [M0] implications (for the architect — NOT bar values, just the published basis)

- **E4 (blind shock):** P1 target ~ −20% to −22% (A2/A3); P3 target ~ −18% to −21% (A9/A10);
  ensemble 80% interval must cover these.
- **E6 (hysteresis band):** published residual = **−5% to −10% vs baseline** during the off year
  (A7), with the A8 roadwork caveat quoted verbatim next to the band.
- **E3 (say-do):** real discrepancy = stated change 5–10% vs enacted ~30% (C6), i.e. a factor
  **3–6 understatement**; direction: stated < enacted. The pre-registration's "factor of [M0]
  (target ≤2×)" should be set against this 3–6× human baseline.
- **Attitude trajectory for persona priors:** 36–40% pre-trial → 52% during → 53% referendum →
  65–74% after reinstatement (C1–C4).

---

## Summary verdict table

| Source | Access status | Fetchable 2026-07-14 |
|---|---|---|
| SCB PxWebApi 2.0, DeSO/RegSO demographics + income + households | Open, no key | **Yes — verified live** |
| SCB/Trafa car ownership (DeSO 2024–25; municipality 2002–) | Open, no key | Yes (DeSO table listed in live API response; 2006-era small-area = paid order) |
| Trafiklab GTFS Sverige 2 / GTFS Regional (SL) | Free API key, CC0 | Yes with key (docs verified; earliest archive ~2012, [unverified]) |
| Trafiklab KoDa historical GTFS | Free API key | Yes with key — but earliest 2020-02-05; useless for 2006 |
| OSM / Geofabrik Sweden | Open, ODbL | Yes; 2006-era OSM does not exist — manual back-editing required |
| Truth series (Eliasson 2009; Börjesson 2012; Eliasson 2014; MAK 2006) | Open PDFs/mirrors | **Yes — all downloaded & text-verified today** |
| Qualitative voice (Henriksson et al. 2011 IJERPH) | Open access | **Yes — verified** (book chapters: purchase) |
| RES 2005–2006 microdata (Trafa) | **Application** (OSL 24:8; trafikanalys@trafa.se) | No — timeline/cost unstated; file now |
| Stockholm County RVU 1986/2004/2015 (Region Stockholm) | **Application/request** (registrator.tf@sl.se) | No — ad-hoc route; file now |

*End of DRAFT. Every number above marked [unverified] must be resolved or explicitly accepted as
uncertain in the M0 amendment before bars are frozen.*
