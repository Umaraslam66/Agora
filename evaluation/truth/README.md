# evaluation/truth/ — QUARANTINED (pre-registration §2)

## What lives here

The ground-truth outcome series for the anchoring natural experiment, from
published sources only:

- Published cordon-crossing counts, 2005–2008 (covering pre-toll, trial,
  removal, and permanent reintroduction periods).
- Published attitude time series (stated support/opposition over the same
  timeline), used for say-do scoring.
- Source citations and extraction notes for every series.

## The quarantine rule

Pre-registration §2 ("The wall"): **nothing measured in the blind periods
(P2/P3) may influence any tuning, prompt, correction, threshold, or model
choice.** Therefore:

- No code under `world/`, `agents/`, `grounding/`, `calibration/`,
  `serving/`, or `training/` may import this package or read these files —
  enforced statically by `tests/test_truth_import_boundary.py`.
- Importing `evaluation.truth` raises `RuntimeError` unless
  `AGORA_EVAL_CONTEXT=1` is set. Only the evaluation harness sets it.
- Each blind test is scored **once**; the result is sealed in `docs/`
  whatever it says. Post-hoc re-runs are labeled as such and cannot
  overwrite a sealed verdict.

If you think you need data from this directory anywhere else, you are about
to break the experiment. Stop and re-read `01_PREREGISTRATION.md` §2.
