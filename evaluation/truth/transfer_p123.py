"""BT2 sealed answer key (transfer arena, A8) — QUARANTINED.

Transcribes, VERBATIM, the P0→P3 aggregate falsification quantities for the
transfer arena from the extraction of record (M0 data inventory appendix
§A, sources downloaded and text-verified 2026-07-14; nothing here was
re-extracted). Sources cited as in the extraction: [E09] = Eliasson et al.
2009, TRA 43(3):240-250; [B12] = Börjesson et al. 2012, CTS WP 2012:3;
[E14] = Eliasson 2014, CTS WP 2014:7. Nothing here is a new fact: A1.5's
disclosure carries — the outcomes are public knowledge; blindness is
procedural (the wall forbids these numbers from influencing any model
input, not from being known).

This module lives in ``evaluation/truth/`` so the import-quarantine
(static scan + the package tripwire requiring AGORA_EVAL_CONTEXT=1)
mechanically keeps it out of every agent-facing package. It is imported by
exactly one caller: the BT2 assembly driver, at scoring time, under the
owner's explicit authorization.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# P1 — introduction (charge on; scored window = trial months 3-6)
# ---------------------------------------------------------------------------

#: A3 [B12] Table 1 p.6 + §3.1 p.7: monthly trial reductions Jan -28%,
#: Feb -23%, Mar -22%, Apr -21%, May -20%, Jun -21%, Jul -24%; the
#: STABILIZED reading is the Mar-Jun average -21% ("overreaction" in
#: Jan-Feb, then stable 20-22%). Scored quantity for P1 (the A8.1 window is
#: the months-3-6 analog). Stored as a positive drop fraction.
P1_DROP_STABILIZED: float = 0.21

#: A2 [E09] §3 + Fig.2 p.243: ~22% below 12-months-before, after the first
#: month (daytime 06:00-19:00 weekdays). REPORTED alternative stabilized
#: reading — never a bar.
P1_DROP_E09_STABILIZED: float = 0.22

#: A3 monthly trajectory (positive drop fractions, trial months 1-7).
#: REPORTED only (the months-1-2 overreaction is a trajectory diagnostic).
P1_MONTHLY_DROPS: tuple = (0.28, 0.23, 0.22, 0.21, 0.20, 0.21, 0.24)

#: A1 [E09] §3 p.243: pre-trial planning target 10-15% reduction during
#: charging hours; traffic models predicted 20-25%. Context/benchmark only.
P1_FORECAST_PLANNING_TARGET: tuple = (0.10, 0.15)
P1_FORECAST_MODEL_RANGE: tuple = (0.20, 0.25)

# ---------------------------------------------------------------------------
# P2 — removal (charge off; E6's phase; scored window = the full off-year)
# ---------------------------------------------------------------------------

#: A7 [B12] §3.1 p.6-7 (THE E6 anchor): over the off-year following removal
#: cordon volumes remained 5-10% BELOW the pre-trial baseline year;
#: "traffic volumes immediately rebounded almost to the same level as
#: before the charges - but not quite"; interpreted as persisting habits.
#: The SEALED pass condition remains A2.5's band (arm (a) residual in
#: [4%, 12%] of P0; arm (b) < 4%; non-overlapping 80% intervals) — this
#: observed range is the anchor the band was set against, stored as
#: positive residual-drop fractions.
P2_RESIDUAL_RANGE: tuple = (0.05, 0.10)

#: A8 [E09] §3 p.243 caveat, carried with the residual: autumn-after-removal
#: traffic "a few percent lower" than the autumn baseline was concentrated
#: at two roadwork-affected bridges — "uncertain what conclusions can be
#: drawn". Reported alongside any P2 verdict, verbatim.
P2_CAVEAT: str = (
    "off-year residual partially confounded by roadwork at two bridges "
    "([E09] section 3 p.243); reported with the verdict"
)

# ---------------------------------------------------------------------------
# P3 — return (charge on, permanent; scored window = first post-return
# calendar-year reading)
# ---------------------------------------------------------------------------

#: A10 [B12] Table 1 p.6: post-return annual series vs the pre-trial
#: baseline year: first partial year (months 1-5 after return) -19%; then
#: -18%, -18%, -19%, -20% (weekdays 06-19). The scored P3 quantity is the
#: first reading (-19%), matching the A8.1 window.
P3_DROP_FIRST_YEAR: float = 0.19
P3_ANNUAL_DROPS: tuple = (0.19, 0.18, 0.18, 0.19, 0.20)

#: A9 [B12] §3.1 p.7: first month after return -21% (charging hours) vs the
#: same month of the baseline year — same relative level as during the
#: trial. REPORTED first-month diagnostic — never a bar.
P3_DROP_FIRST_MONTH: float = 0.21

# ---------------------------------------------------------------------------
# Interval discipline (A8.2 pairing; A4.2 analog carried by A8)
# ---------------------------------------------------------------------------

#: Ensemble interval level for phase coverage readings (BT1 discipline
#: carried; encoded as data so the driver cannot restate it).
INTERVAL_LEVEL: float = 0.80
