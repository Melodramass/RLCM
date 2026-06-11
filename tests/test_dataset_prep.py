"""Freeze data preparation behavior (boxed-instruction prompts and the RLCR
prompt format)."""

import importlib.util
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "data")


def load_module(filename):
    path = os.path.join(DATA_SCRIPTS, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rlcr_prep():
    return load_module("deepscaler_dataset_rlcr.py")


@pytest.fixture(scope="module")
def deepscaler_prep():
    return load_module("deepscaler_dataset.py")


class TestRLCRDatasetPrep:
    def test_prompt_contains_problem_and_rlcr_instruction(self, rlcr_prep):
        prompt = rlcr_prep.build_prompt("What is 1+1?")
        assert prompt.startswith("What is 1+1?")
        for tag in ("<answer>", "<analysis>", "<confidence>"):
            assert tag in prompt

    def test_convert_examples_row_schema(self, rlcr_prep):
        rows = rlcr_prep.convert_examples(
            [{"problem": "What is 1+1?", "answer": "2"}]
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["data_source"] == "stage1_rlcr"
        assert row["prompt"][0]["role"] == "user"
        assert row["prompt"][0]["content"].startswith("What is 1+1?")
        assert row["reward_model"] == {"style": "rule", "ground_truth": "2"}
        assert row["extra_info"]["split"] == "train"
        assert row["extra_info"]["index"] == 0
        assert row["ability"] == "math"

    def test_convert_examples_missing_key_raises(self, rlcr_prep):
        with pytest.raises(ValueError, match="missing required keys"):
            rlcr_prep.convert_examples([{"problem": "no answer field"}])


class TestDeepscalerDatasetPrep:
    def test_make_map_fn_row_schema(self, deepscaler_prep):
        process_fn = deepscaler_prep.make_map_fn("train")
        row = process_fn({"problem": "What is 1+1?", "answer": "2"}, 0, data_source="aime")
        assert row["data_source"] == "aime"
        assert row["prompt"][0]["role"] == "user"
        assert row["prompt"][0]["content"].startswith("What is 1+1?")
        assert row["prompt"][0]["content"].endswith(
            "Let's think step by step and output the final answer within \\boxed{}."
        )
        assert row["ability"] == "math"
        assert row["reward_model"] == {"style": "rule", "ground_truth": "2"}
        assert row["extra_info"] == {"split": "train", "index": 0}

    def test_extract_solution_unwraps_boxed(self, deepscaler_prep):
        assert deepscaler_prep.extract_solution("thus \\boxed{42}") == "42"
