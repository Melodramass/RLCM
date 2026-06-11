#!/usr/bin/env python3
"""Natural-trace evaluation: natural accuracy and natural ECE only.

This is the single, no-plotting replacement for the old ``plot_natural_ece.py``
and ``plot_natural_ece_rlcr.py`` scripts. It supports two confidence sources:

  * ``boxed`` : the policy does not self-report confidence; a trained probe reads
                it out of the hidden states (RLCM / GRPO / Base). Requires a
                LEAD probe and per-dataset ``hidden_states.pt``.
  * ``rlcr``  : the policy self-reports ``<confidence>`` in its output
                (RLCR / C2GSPG). No probe or hidden states needed.

For every dataset it reports two numbers, plus a macro average:
  * natural_accuracy : accuracy of the freely generated trace at the final
                       budget (a trace that did not finish counts as wrong).
  * natural_ece      : Expected Calibration Error of confidence vs. that accuracy.

Layout expected under ``--base_dir`` (produced by ``evaluate.py``):
    <base_dir>/<dataset>/rollouts_<dataset>.jsonl
    <base_dir>/<dataset>/budget_eval_<dataset>.jsonl
    <base_dir>/<dataset>/hidden_states.pt          (boxed mode only)
    <base_dir>/lead/probe/best_probe.pt            (boxed mode only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Optional

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from metric_utils import _compute_ece
from math_dapo import last_boxed_only_string

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _load_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _load_hidden_states(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "hidden_states" in payload:
        return payload["hidden_states"]
    return payload


def _resolve_paths(base_dir: str, dataset: str) -> dict[str, str]:
    dataset_dir = os.path.join(base_dir, dataset)
    return {
        "dataset_dir": dataset_dir,
        "rollouts_file": os.path.join(dataset_dir, f"rollouts_{dataset}.jsonl"),
        "budget_eval_file": os.path.join(dataset_dir, f"budget_eval_{dataset}.jsonl"),
        "hidden_states_file": os.path.join(dataset_dir, "hidden_states.pt"),
    }


# --------------------------------------------------------------------------- #
# Confidence extraction (rlcr self-report)
# --------------------------------------------------------------------------- #
def _extract_answer_tag(solution_str: str) -> Optional[str]:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


# --------------------------------------------------------------------------- #
# Probe inference (boxed mode)
# --------------------------------------------------------------------------- #
def _run_probe_inference(probe, hidden_states: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    probe.eval()
    hidden_states = hidden_states.float()
    n = hidden_states.shape[0]
    confidences = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            logits = probe(hidden_states[start:end]).squeeze(-1)
            confidences[start:end] = torch.sigmoid(logits).cpu().numpy()
    return confidences


# --------------------------------------------------------------------------- #
# Per-dataset evaluation
# --------------------------------------------------------------------------- #
def _evaluate_boxed(paths: dict, probe, final_budget: int, bins: int) -> Optional[dict]:
    if not os.path.exists(paths["budget_eval_file"]):
        logger.warning("missing %s", paths["budget_eval_file"])
        return None
    if not os.path.exists(paths["hidden_states_file"]):
        logger.warning("missing %s", paths["hidden_states_file"])
        return None
    if not os.path.exists(paths["rollouts_file"]):
        logger.warning("missing %s", paths["rollouts_file"])
        return None

    budget_rows = _load_jsonl(paths["budget_eval_file"])
    rollouts = _load_jsonl(paths["rollouts_file"])
    hidden_states = _load_hidden_states(paths["hidden_states_file"])

    rollout_acc = {idx: (1.0 if r.get("acc", False) else 0.0) for idx, r in enumerate(rollouts)}

    final_indices, final_records = [], []
    for i, rec in enumerate(budget_rows):
        if int(rec.get("budget", -1)) == final_budget:
            final_indices.append(i)
            final_records.append(rec)
    if not final_records:
        logger.warning("no rows at budget %s", final_budget)
        return None

    # Natural accuracy: a trace counts only if it naturally produced a boxed answer.
    accuracies = []
    for rec in final_records:
        if last_boxed_only_string(rec.get("truncated_cot", "")) is not None:
            accuracies.append(rollout_acc.get(rec.get("rollout_idx", -1), 0.0))
        else:
            accuracies.append(0.0)

    confidences = _run_probe_inference(probe, hidden_states[final_indices])
    return _finalize(confidences, accuracies, bins)


def _evaluate_rlcr(paths: dict, final_budget: int, bins: int) -> Optional[dict]:
    if not os.path.exists(paths["budget_eval_file"]):
        logger.warning("missing %s", paths["budget_eval_file"])
        return None

    budget_rows = _load_jsonl(paths["budget_eval_file"])
    final_records = [r for r in budget_rows if int(r.get("budget", -1)) == final_budget]
    if not final_records:
        logger.warning("no rows at budget %s", final_budget)
        return None

    rollouts = _load_jsonl(paths["rollouts_file"]) if os.path.exists(paths["rollouts_file"]) else []
    rollout_acc = {idx: (1.0 if r.get("acc", False) else 0.0) for idx, r in enumerate(rollouts)}
    rollout_out = {idx: str(r.get("output", "")) for idx, r in enumerate(rollouts)}

    confidences = [float(r.get("mean_confidence", 0.5)) for r in final_records]

    # Natural accuracy: a trace counts only if it naturally produced an <answer>.
    accuracies = []
    for rec in final_records:
        ridx = rec.get("rollout_idx", -1)
        if _extract_answer_tag(rollout_out.get(ridx, "")) is not None:
            accuracies.append(rollout_acc.get(ridx, 0.0))
        else:
            accuracies.append(0.0)

    return _finalize(np.asarray(confidences, dtype=np.float32), accuracies, bins)


def _finalize(confidences, accuracies, bins: int) -> dict:
    conf_t = torch.tensor(confidences, dtype=torch.float32)
    acc_t = torch.tensor(accuracies, dtype=torch.float32)
    ece = _compute_ece(conf_t, acc_t, bins=bins)
    return {
        "natural_accuracy": float(acc_t.mean()),
        "natural_ece": float(ece) if ece is not None else 0.0,
        "mean_confidence": float(conf_t.mean()),
        "n_samples": int(conf_t.numel()),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def natural_eval(
    base_dir: str,
    datasets: list[str],
    mode: str,
    final_budget: int = 8000,
    bins: int = 10,
    probe_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run natural-trace evaluation for every dataset and return a results dict."""
    probe = None
    if mode == "boxed":
        from probe import ConfProbe

        probe_path = probe_path or os.path.join(base_dir, "lead", "probe", "best_probe.pt")
        logger.info("loading LEAD probe from %s", probe_path)
        probe = ConfProbe.from_pretrained(probe_path)
        probe.eval()

    results: dict[str, Any] = {}
    for dataset in datasets:
        paths = _resolve_paths(base_dir, dataset)
        logger.info("evaluating %s", dataset)
        if mode == "boxed":
            res = _evaluate_boxed(paths, probe, final_budget, bins)
        else:
            res = _evaluate_rlcr(paths, final_budget, bins)
        if res is None:
            logger.warning("skipping %s", dataset)
            continue
        results[dataset] = res
        logger.info(
            "  %s: acc=%.4f  ece=%.4f  conf=%.4f  n=%d",
            dataset, res["natural_accuracy"], res["natural_ece"],
            res["mean_confidence"], res["n_samples"],
        )

    if results:
        results["_average"] = {
            "natural_accuracy": float(np.mean([r["natural_accuracy"] for r in results.values()])),
            "natural_ece": float(np.mean([r["natural_ece"] for r in results.values()])),
            "num_datasets": len(results),
        }
    results["_meta"] = {"mode": mode, "final_budget": final_budget, "bins": bins}
    return results


def print_table(results: dict[str, Any]) -> None:
    rows = {k: v for k, v in results.items() if not k.startswith("_")}
    width = 70
    print("=" * width)
    print(f"{'Dataset':>18} | {'NaturalAcc':>10} | {'NaturalECE':>10} | {'N':>6}")
    print("-" * width)
    for ds in sorted(rows):
        r = rows[ds]
        print(f"{ds:>18} | {r['natural_accuracy']:>10.4f} | {r['natural_ece']:>10.4f} | {r['n_samples']:>6}")
    print("-" * width)
    avg = results.get("_average")
    if avg:
        print(f"{'Average':>18} | {avg['natural_accuracy']:>10.4f} | {avg['natural_ece']:>10.4f} |")
    print("=" * width)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Natural accuracy + natural ECE evaluation")
    p.add_argument("--base_dir", required=True, help="Directory with per-dataset eval outputs")
    p.add_argument("--datasets", nargs="+", required=True, help="Dataset run_names to evaluate")
    p.add_argument("--mode", choices=["boxed", "rlcr"], required=True)
    p.add_argument("--final_budget", type=int, default=8000)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--probe_path", default=None, help="boxed mode: path to LEAD probe (best_probe.pt)")
    p.add_argument("--output_json", default=None, help="Where to write results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    results = natural_eval(
        base_dir=args.base_dir,
        datasets=args.datasets,
        mode=args.mode,
        final_budget=args.final_budget,
        bins=args.bins,
        probe_path=args.probe_path,
    )
    print_table(results)
    out = args.output_json or os.path.join(args.base_dir, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
