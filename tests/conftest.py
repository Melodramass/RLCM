"""Shared test setup.

These tests freeze the behavior of the core RLCM code paths (margin reward,
probe, RLCR scoring, calibration metrics, answer grading, data prep).

Run from the repo root:
    python -m pytest tests/ -v

All tests are CPU-only and use synthetic inputs; no model or dataset downloads.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Repo root first so the local `rlcm` package is importable.
sys.path.insert(0, REPO_ROOT)
# Make the vendored verl fork win over any site-packages install.
sys.path.insert(0, os.path.join(REPO_ROOT, "verl"))
# Eval pipeline modules (metric_utils, math_dapo eval variant) are plain
# scripts, not a package; import them by directory.
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "eval"))
