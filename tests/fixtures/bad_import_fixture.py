# Deliberately quarantine-breaking module — used ONLY by
# tests/test_truth_import_boundary.py to prove the scanner has teeth.
# It is never imported; the scanner reads it as source text.
from evaluation.truth import series  # noqa: F401
