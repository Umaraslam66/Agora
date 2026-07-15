"""Planted violation fixture: agent-side code importing the E5 unmasked
vocabulary. NEVER imported by anything — it exists so the truth-boundary
scanner's teeth on the unmasked-vocabulary quarantine are themselves tested
(tests/test_truth_import_boundary.py). Contains no real place names."""

from evaluation.truth.unmasked_vocabulary import ZONE_NAMES  # noqa: F401  (planted)
