# DRAFT — Application for RVU/RES travel-survey microdata
*Status: DRAFT for the project owner to review, sign, and submit. Placeholders in
[BRACKETS] must be filled with the applicant's own details. No commitment is made
by this repository; this file is a working draft only.*

*Route note: the national travel survey covering the study window is RES 2005–2006
(Trafikanalys / formerly SIKA). Depending on the current custodianship, this
application is addressed either to Trafikanalys directly or through Statistics
Sweden's microdata service (MONA). The text below is written so that the substantive
sections can be pasted into either channel's form. See `M0_DATA_INVENTORY.md` for
the confirmed route.*

---

## Subject / Ärende

**EN:** Request for research access to RES 2005–2006 national travel survey
microdata (individual, household, and trip records), Stockholm County subsample
prioritized.

**SV:** Ansökan om tillgång till mikrodata från den nationella
resvaneundersökningen RES 2005–2006 (individ-, hushålls- och resedata) för
forskningsändamål, med prioritet för delurvalet i Stockholms län.

## 1. Applicant

- Name: [FULL NAME]
- Affiliation: [INSTITUTION / ORGANISATION, or "independent researcher"]
- Contact: [EMAIL, PHONE]
- Data protection officer / responsible entity (if applicable): [DPO / LEGAL ENTITY]

## 2. Project summary

The project develops and validates a computational method for simulating the
travel behavior of a synthetic population, in which each simulated individual is
seeded from **one real, anonymized travel-diary record** rather than from segment
averages. The method is validated against the Stockholm congestion-charging
natural experiment (trial January–July 2006, removal August 2006, permanent
reintroduction August 2007) under a **pre-registered, blind evaluation protocol**:
the model is calibrated only on the pre-charge and trial periods and must predict
the removal and reintroduction periods without access to the observed outcomes.

The evaluation protocol was frozen and version-controlled before any model was
built. The project is open-source in its code and documentation; **microdata are
never redistributed, published, or committed to any repository** (see §6).

## 3. Data requested

Microdata from **RES 2005–2006** (and, if straightforwardly available under the
same decision, the corresponding later RVU Sverige waves for robustness checks):

**Individual level:** age (or age band), sex, employment status, driving licence
holding, personal income class, sampling weight.

**Household level:** household identifier (pseudonymized), household size and
composition, number of cars available, household income class.

**Trip level (travel-diary day):** trip purpose, main mode, departure and arrival
times (or time bands), trip distance, and origin/destination at the **finest
geography the anonymization rules permit** (SAMS area, or municipality if finer
levels cannot be released).

Priority subsample: respondents residing in Stockholm County. The national
remainder is requested for building transferable behavioral priors.

**We do not request** any direct identifiers, exact addresses or coordinates, or
linkage to any other register. If a released geography finer than municipality is
only possible within a secure remote-access environment (e.g. MONA), we accept
remote-access-only processing.

## 4. Purpose and why microdata (not aggregates) are necessary

The method under test seeds each simulated agent from one individual diary record
to preserve the real between-individual variance of behavior. Published aggregate
tables cannot serve this purpose by construction: the scientific claim being
tested is precisely that record-level heterogeneity, not segment averages, drives
predictive validity. One pre-registered evaluation (variance preservation)
explicitly measures whether the simulated population's spread matches the spread
across real individuals; it is unanswerable from aggregates.

## 5. Legal basis and ethics

- Processing for research purposes under GDPR Art. 6(1)(e)/(f) with the
  safeguards of Art. 89(1); Swedish supplementary provisions
  (lag 2018:218; etikprövningslagen 2003:460 where applicable).
- [IF AFFILIATED: ethical review status / Etikprövningsmyndigheten decision
  number, or a statement of why review is not required for pseudonymized,
  non-sensitive travel data.]
- No attempt at re-identification will ever be made; no linkage requested.

## 6. Data protection measures

- Storage: encrypted volume on access-controlled machines; no cloud replication
  outside [COUNTRY/EEA]; if MONA is the channel, all processing stays in MONA.
- Access: named applicant(s) only: [NAMES].
- The project's public repository contains code and synthetic stand-in data only;
  a CI-enforced boundary prevents survey microdata paths from entering the
  repository. Only aggregate, disclosure-safe statistics (distributions, model
  parameters, evaluation scores) are ever published.
- Retention: deletion or return of microdata at project end, at latest [DATE],
  with written confirmation.

## 7. Outputs

Peer-reviewed publication(s) and an open-source method implementation. All
published quantities are distributional or aggregate with standard disclosure
control; no record-level output.

## 8. Timeline

Data are needed at the calibration stage of the project, ideally within
[8–12 weeks]. We ask for an indication of expected processing time and any fees,
and are glad to adjust the request's scope (e.g. coarser geography) if that
materially shortens the path.

---

*Draft prepared 2026-07-14 as part of the M0 data audit. Review §3 geography and
§5 ethics-review status carefully before submission — these are the two sections
most likely to need tailoring to the applicant's situation.*
