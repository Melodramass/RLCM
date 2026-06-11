#!/usr/bin/env python3
"""One-command evaluation: checkpoint -> natural accuracy + natural ECE.

Pipeline (stages are idempotent; a ``.<stage>.done`` marker + non-empty output
means "skip unless --force"):

    stage 0  merge        verl FSDP `global_step_N/actor` -> `global_step_N/merge`
    stage 1  rollouts     free-generation traces (natural accuracy source)
    stage 2  budget eval  forced-answer correctness at the final budget
                          (boxed: also multi-budget on the probe dataset;
                           rlcr : parse self-reported <confidence>)
    stage 3  hidden       hidden states at the final budget        (boxed only)
    stage 4  probe        train the LEAD probe                     (boxed only,
                          probe dataset only)
    stage 5  natural_eval natural accuracy + natural ECE table + results.json

Modes:
    boxed : RLCM / GRPO / Base. Confidence read out by a trained probe.
    rlcr  : RLCR / C2GSPG. Model self-reports <confidence>; no probe needed.

Examples:
    python scripts/eval/evaluate.py --ckpt checkpoints/rlcm/global_step_600 --mode boxed
    python scripts/eval/evaluate.py --ckpt checkpoints/rlcr/global_step_600 --mode rlcr \
        --config_dir scripts/eval/configs/probe_training --datasets aime24_rlcr,math_rlcr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

STEP1 = SCRIPT_DIR / "1_obtain_validation_rollouts.py"
STEP2_BOXED = SCRIPT_DIR / "2_budget_eval.py"
STEP2_RLCR = SCRIPT_DIR / "2_budget_eval_rlcr.py"
STEP3 = SCRIPT_DIR / "3_reproduce_hidden_states_extraction.py"
STEP4 = SCRIPT_DIR / "4_probe_training.py"

DEFAULT_CONFIG_DIR = SCRIPT_DIR / "configs" / "probe_training"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    run_name: str
    data_files: list[str]
    output_dir: Path
    # step1
    n_samples: int = 4
    temperature: float = 0.6
    max_tokens: int = 8000
    # step2
    budgets: str = "8000"
    n_forced: int = 1
    forced_max_tokens: int = 64
    code_forced_max_tokens: int = 2048
    gpqa_answer_format: str = "boxed"
    # step3
    max_batch_quota: float = 120000
    max_batch_size: int = 32
    # step4
    split_mode: str = "prompt"
    val_split: float = 0.2
    epochs: int = 100
    batch_size: int = 64
    patience: int = 30
    # gpu
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9

    @property
    def rollouts_file(self) -> Path:
        return self.output_dir / f"rollouts_{self.run_name}.jsonl"

    @property
    def budget_eval_file(self) -> Path:
        return self.output_dir / f"budget_eval_{self.run_name}.jsonl"

    @property
    def hidden_states_file(self) -> Path:
        return self.output_dir / "hidden_states.pt"

    @property
    def probe_dir(self) -> Path:
        return self.output_dir / "probe"

    @property
    def probe_model(self) -> Path:
        return self.probe_dir / "best_probe.pt"

    @property
    def final_budget(self) -> int:
        return max(int(b) for b in self.budgets.split(",") if b.strip())


def _csv(values: Any) -> str:
    if isinstance(values, list):
        return ",".join(str(v) for v in values)
    return str(values)


def load_config(path: Path, output_root: Path) -> DatasetConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    s1, s2, s3, s4, gpu = (raw.get(k, {}) for k in ("step1", "step2", "step3", "step4", "gpu"))
    return DatasetConfig(
        run_name=raw["run_name"],
        data_files=raw["data_files"],
        output_dir=output_root / raw["run_name"],
        n_samples=int(s1.get("n_samples", 4)),
        temperature=float(s1.get("temperature", 0.6)),
        max_tokens=int(s1.get("max_tokens", 8000)),
        budgets=_csv(s2.get("budgets", [8000])),
        n_forced=int(s2.get("n_forced", 1)),
        forced_max_tokens=int(s2.get("forced_max_tokens", 64)),
        code_forced_max_tokens=int(s2.get("code_forced_max_tokens", 2048)),
        gpqa_answer_format=str(s2.get("gpqa_answer_format", "boxed")),
        max_batch_quota=float(s3.get("max_batch_quota", 120000)),
        max_batch_size=int(s3.get("max_batch_size", 32)),
        split_mode=str(s4.get("split_mode", "prompt")),
        val_split=float(s4.get("val_split", 0.2)),
        epochs=int(s4.get("epochs", 100)),
        batch_size=int(s4.get("batch_size", 64)),
        patience=int(s4.get("patience", 30)),
        tensor_parallel_size=int(gpu.get("tensor_parallel_size", 2)),
        gpu_memory_utilization=float(gpu.get("gpu_memory_utilization", 0.9)),
    )


# --------------------------------------------------------------------------- #
# Stage running helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], dry_run: bool) -> None:
    print("  $", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _done_marker(cfg: DatasetConfig, stage: str) -> Path:
    return cfg.output_dir / f".{stage}.done"


def _is_done(cfg: DatasetConfig, stage: str, output: Path) -> bool:
    if _nonempty(output) and _done_marker(cfg, stage).exists():
        return True
    if _nonempty(output):  # backfill marker for legacy outputs
        _write_marker(cfg, stage, output)
        return True
    return False


def _write_marker(cfg: DatasetConfig, stage: str, output: Path) -> None:
    marker = _done_marker(cfg, stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {"stage": stage, "output": str(output),
             "completed_at_utc": datetime.now(timezone.utc).isoformat()},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _effective_tp(configured: int) -> int:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda is not None:
        cuda = cuda.strip()
        visible = 0 if cuda in {"", "-1"} else len([x for x in cuda.split(",") if x.strip()])
    else:
        visible = None
    if visible is None:
        return configured
    if visible <= 0:
        return 1
    return min(configured, visible)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_merge(ckpt: Path, skip: bool, dry_run: bool) -> str:
    """Resolve the HF model path, merging the FSDP actor shard if needed."""
    if (ckpt / "config.json").exists():  # already an HF dir
        return str(ckpt)

    merge_dir = ckpt / "merge"
    actor_dir = ckpt / "actor"
    if _nonempty(merge_dir / "config.json") or (skip and merge_dir.exists()):
        print(f"[merge] reuse {merge_dir}")
        return str(merge_dir)
    if not actor_dir.exists():
        raise FileNotFoundError(
            f"[merge] no HF dir, no merge/, and no actor/ under {ckpt}; pass a valid checkpoint."
        )
    print(f"[merge] {actor_dir} -> {merge_dir}")
    _run(
        [sys.executable, "-m", "verl.model_merger", "merge",
         "--backend", "fsdp", "--local_dir", str(actor_dir), "--target_dir", str(merge_dir)],
        dry_run,
    )
    return str(merge_dir)


def stage_rollouts(cfg: DatasetConfig, model_path: str, rlcr: bool, force: bool, dry_run: bool) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if _is_done(cfg, "rollouts", cfg.rollouts_file) and not force:
        print(f"[{cfg.run_name}] rollouts skip")
        return
    cmd = [
        sys.executable, str(STEP1),
        "--model_path", model_path,
        "--data_files", *cfg.data_files,
        "--output_path", str(cfg.rollouts_file),
        "--n_samples", str(cfg.n_samples),
        "--temperature", str(cfg.temperature),
        "--max_tokens", str(cfg.max_tokens),
        "--tensor_parallel_size", str(_effective_tp(cfg.tensor_parallel_size)),
        "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
    ]
    if rlcr:
        cmd.append("--use_rlcr")
    _run(cmd, dry_run)
    if not dry_run and not _nonempty(cfg.rollouts_file):
        raise RuntimeError(f"[{cfg.run_name}] rollouts produced no output")
    _write_marker(cfg, "rollouts", cfg.rollouts_file)


def stage_budget_eval(cfg: DatasetConfig, model_path: str, rlcr: bool, force: bool, dry_run: bool) -> None:
    if not dry_run and not _nonempty(cfg.rollouts_file):
        raise RuntimeError(f"[{cfg.run_name}] budget_eval needs rollouts first")
    if _is_done(cfg, "budget_eval", cfg.budget_eval_file) and not force:
        print(f"[{cfg.run_name}] budget_eval skip")
        return
    script = STEP2_RLCR if rlcr else STEP2_BOXED
    cmd = [
        sys.executable, str(script),
        "--input_file", str(cfg.rollouts_file),
        "--model_path", model_path,
        "--output_file", str(cfg.budget_eval_file),
        "--n_forced", str(cfg.n_forced),
        "--budgets", cfg.budgets,
        "--forced_max_tokens", str(cfg.forced_max_tokens),
        "--tensor_parallel_size", str(_effective_tp(cfg.tensor_parallel_size)),
        "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
    ]
    if not rlcr:
        # Only 2_budget_eval.py understands these; 2_budget_eval_rlcr.py
        # always parses RLCR tags and rejects unknown flags.
        cmd += [
            "--code_forced_max_tokens", str(cfg.code_forced_max_tokens),
            "--gpqa_answer_format", cfg.gpqa_answer_format,
        ]
    _run(cmd, dry_run)
    if not dry_run and not _nonempty(cfg.budget_eval_file):
        raise RuntimeError(f"[{cfg.run_name}] budget_eval produced no output")
    _write_marker(cfg, "budget_eval", cfg.budget_eval_file)


def stage_hidden(cfg: DatasetConfig, model_path: str, force: bool, dry_run: bool) -> None:
    if not dry_run and not _nonempty(cfg.budget_eval_file):
        raise RuntimeError(f"[{cfg.run_name}] hidden needs budget_eval first")
    if _is_done(cfg, "hidden", cfg.hidden_states_file) and not force:
        print(f"[{cfg.run_name}] hidden skip")
        return
    _run(
        [sys.executable, str(STEP3),
         "--input_file", str(cfg.budget_eval_file),
         "--model_path", model_path,
         "--output_file", str(cfg.hidden_states_file),
         "--max_batch_quota", str(cfg.max_batch_quota),
         "--max_batch_size", str(cfg.max_batch_size)],
        dry_run,
    )
    if not dry_run and not _nonempty(cfg.hidden_states_file):
        raise RuntimeError(f"[{cfg.run_name}] hidden produced no output")
    _write_marker(cfg, "hidden", cfg.hidden_states_file)


def stage_probe(cfg: DatasetConfig, force: bool, dry_run: bool) -> None:
    if not dry_run and not _nonempty(cfg.hidden_states_file):
        raise RuntimeError(f"[{cfg.run_name}] probe needs hidden states first")
    if _is_done(cfg, "probe", cfg.probe_model) and not force:
        print(f"[{cfg.run_name}] probe skip")
        return
    _run(
        [sys.executable, str(STEP4),
         "--input_file", str(cfg.hidden_states_file),
         "--output_dir", str(cfg.probe_dir),
         "--split_mode", cfg.split_mode,
         "--val_split", str(cfg.val_split),
         "--epochs", str(cfg.epochs),
         "--batch_size", str(cfg.batch_size),
         "--patience", str(cfg.patience)],
        dry_run,
    )
    if not dry_run and not _nonempty(cfg.probe_model):
        raise RuntimeError(f"[{cfg.run_name}] probe produced no output")
    _write_marker(cfg, "probe", cfg.probe_model)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def discover_configs(config_dir: Path, wanted: set[str], rlcr: bool) -> list[Path]:
    paths = sorted(config_dir.glob("*.json"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
    else:
        # Without an explicit --datasets list, match configs to the mode:
        # *_rlcr configs hold RLCR-format prompts, everything else boxed.
        paths = [p for p in paths if p.stem.endswith("_rlcr") == rlcr]
    if not paths:
        raise FileNotFoundError(f"No matching configs in {config_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Checkpoint -> natural accuracy + natural ECE")
    p.add_argument("--ckpt", required=True,
                   help="global_step_N dir (FSDP actor or merged) or an HF model dir")
    p.add_argument("--mode", choices=["boxed", "rlcr"], required=True)
    p.add_argument("--config_dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--datasets", default="", help="comma-separated config stems to include")
    p.add_argument("--probe_dataset", default="lead",
                   help="boxed mode: config stem whose probe is applied to all datasets")
    p.add_argument("--output_root", default="",
                   help="where eval outputs go (default: <ckpt>/eval_<mode>)")
    p.add_argument("--final_budget", type=int, default=0,
                   help="override the budget used for natural eval (default: max config budget)")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--skip_merge", action="store_true", help="reuse existing merge/ without re-merging")
    p.add_argument("--force", action="store_true", help="re-run every stage")
    p.add_argument("--dry_run", action="store_true", help="print commands only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.ckpt).resolve()
    rlcr = args.mode == "rlcr"

    output_root = Path(args.output_root).resolve() if args.output_root else ckpt / f"eval_{args.mode}"
    output_root.mkdir(parents=True, exist_ok=True)

    wanted = {s.strip() for s in args.datasets.split(",") if s.strip()}
    config_paths = discover_configs(Path(args.config_dir).resolve(), wanted, rlcr)
    configs = [load_config(p, output_root) for p in config_paths]

    print("=" * 79)
    print(f"checkpoint : {ckpt}")
    print(f"mode       : {args.mode}")
    print(f"datasets   : {', '.join(c.run_name for c in configs)}")
    print(f"output     : {output_root}")
    print("=" * 79)

    # stage 0
    model_path = stage_merge(ckpt, skip=args.skip_merge, dry_run=args.dry_run)

    # stages 1-4 per dataset
    for cfg in configs:
        print(f"\n--- {cfg.run_name} (final_budget={cfg.final_budget}) ---")
        stage_rollouts(cfg, model_path, rlcr, args.force, args.dry_run)
        stage_budget_eval(cfg, model_path, rlcr, args.force, args.dry_run)
        if not rlcr:
            stage_hidden(cfg, model_path, args.force, args.dry_run)
            if cfg.run_name == args.probe_dataset:
                stage_probe(cfg, args.force, args.dry_run)

    if args.dry_run:
        print("\n[dry_run] skipping stage 5 (natural eval)")
        return

    # stage 5: natural accuracy + natural ECE
    from natural_eval import natural_eval, print_table

    final_budget = args.final_budget or max(c.final_budget for c in configs)
    probe_path = None
    if not rlcr:
        probe_cfg = next((c for c in configs if c.run_name == args.probe_dataset), None)
        if probe_cfg is None:
            raise ValueError(
                f"boxed mode needs the probe dataset '{args.probe_dataset}' in --config_dir/--datasets"
            )
        probe_path = str(probe_cfg.probe_model)

    print("\n" + "=" * 79)
    print("stage 5: natural accuracy + natural ECE")
    print("=" * 79)
    results = natural_eval(
        base_dir=str(output_root),
        datasets=[c.run_name for c in configs],
        mode=args.mode,
        final_budget=final_budget,
        bins=args.bins,
        probe_path=probe_path,
    )
    print_table(results)
    out = output_root / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
