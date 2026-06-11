#!/usr/bin/env python3
"""Convert AIME dataset to RLCR parquet format.

This script converts AIME JSONL files to the same parquet format used by lead_RLCR.parquet,
which includes the RLCR instruction prompt requesting confidence tags.

Usage:
    python aime_to_rlcr.py \
        --input_path deepscaler/data/aime25.jsonl \
        --output_path deepscaler/data/aime25_rlcr.parquet
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


def build_prompt(problem: str) -> str:
    """Build the RLCR user prompt."""
    return f"{problem}\n\n{RLCR_INSTRUCTION}"


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def convert_examples(examples: List[Dict[str, Any]], data_source: str = "aime") -> List[Dict[str, Any]]:
    """Convert raw examples into the parquet row format used by RLCR."""
    rows: List[Dict[str, Any]] = []

    for idx, example in enumerate(examples):
        if "problem" not in example or "answer" not in example:
            raise ValueError(
                f"Example {idx} is missing required keys. Found keys: {sorted(example.keys())}"
            )

        # Get answer - handle both string and numeric types
        answer = example["answer"]
        if isinstance(answer, (int, float)):
            answer = str(answer)

        rows.append(
            {
                "data_source": f"{data_source}_rlcr",
                "prompt": [
                    {
                        "role": "user",
                        "content": build_prompt(example["problem"]),
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "original_id": example.get("id", idx),
                },
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AIME JSONL to RLCR parquet format"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to the output parquet file",
    )
    parser.add_argument(
        "--data_source",
        default="aime",
        help="Data source name (e.g., aime, aime25)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {input_path}...")
    examples = load_jsonl(input_path)
    print(f"Loaded {len(examples)} examples")

    rows = convert_examples(examples, args.data_source)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path)

    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
