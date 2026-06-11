"""Convert DeepScaler stage1 JSON data into RLCR parquet format.

This mirrors the existing ``lead.parquet`` schema and only changes the prompt
template to request RLCR-style answer, analysis, and confidence fields.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd


RLCR_INSTRUCTION = (
    "Let's think step by step. Output the final answer inside <answer> </answer>. "
    "Then analyze your confidence and uncertainty inside <analysis> </analysis>. "
    "Finally, output a confidence score between 0 and 1 inside "
    "<confidence> </confidence>.\n\n"
    "Use exactly this format:\n"
    "<answer> final answer here </answer>\n"
    "<analysis> confidence and uncertainty analysis here </analysis>\n"
    "<confidence> number between 0 and 1 </confidence>"
)


def load_stage1_data(input_path: str) -> List[Dict[str, Any]]:
    """Load the raw DeepScaler stage1 JSON file."""
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples in {input_path}, got {type(data).__name__}")

    return data


def build_prompt(problem: str) -> str:
    """Build the RLCR user prompt while preserving the lead.parquet style."""
    return f"{problem}\n\n{RLCR_INSTRUCTION}"


def convert_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw stage1 examples into the parquet row format used by lead.parquet."""
    rows: List[Dict[str, Any]] = []

    for idx, example in enumerate(examples):
        if "problem" not in example or "answer" not in example:
            raise ValueError(
                f"Example {idx} is missing required keys. Found keys: {sorted(example.keys())}"
            )

        rows.append(
            {
                "data_source": "stage1_rlcr",
                "prompt": [
                    {
                        "role": "user",
                        "content": build_prompt(example["problem"]),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["answer"],
                },
                "extra_info": {
                    "split": "train",
                    "index": idx,
                },
            }
        )

    return rows


def maybe_copy_to_hdfs(output_path: str, hdfs_dir: str | None) -> None:
    """Copy the generated parquet file to HDFS if verl utilities are available."""
    if hdfs_dir is None:
        return

    try:
        from verl.utils.hdfs_io import copy, makedirs
    except ImportError as exc:
        raise ImportError(
            "--hdfs_dir was provided, but verl is not installed in the current environment."
        ) from exc

    makedirs(hdfs_dir)
    copy(src=output_path, dst=os.path.join(hdfs_dir, os.path.basename(output_path)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert deepscaler/data/stage1_data.json into lead_RLCR.parquet"
    )
    parser.add_argument(
        "--input_path",
        default="deepscaler/data/stage1_data.json",
        help="Path to the raw stage1 JSON file",
    )
    parser.add_argument(
        "--output_path",
        default="deepscaler/data/lead_RLCR.parquet",
        help="Path to the output parquet file",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the generated parquet into",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    examples = load_stage1_data(input_path)
    rows = convert_examples(examples)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path)

    print(f"Loaded {len(examples)} examples from {input_path}")
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Columns: {list(df.columns)}")

    maybe_copy_to_hdfs(output_path, args.hdfs_dir)
    if args.hdfs_dir is not None:
        print(f"Copied parquet to {os.path.join(args.hdfs_dir, os.path.basename(output_path))}")


if __name__ == "__main__":
    main()
