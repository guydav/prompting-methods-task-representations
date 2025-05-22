import json
import os
import sys
from pathlib import Path

import pytest
from loguru import logger


def _add_project_root():
    p = Path(__file__).absolute()
    while not list(p.glob(".git")):
        p = p.parent
        if str(p) == "/":
            raise ValueError("Project root not found in parents")
    p = str(p)
    logger.debug(f"Adding {p} to path")
    sys.path.append(p)


_add_project_root()

from recipe.function_vectors.utils.eval_utils import prompt_based_eval_no_intervention
from recipe.function_vectors.utils.prompt_utils import load_dataset
from tests.test_sentence_eval import SentenceEvalEvaluator
from tests.test_utils import DATASETS_PATH, PROMPTS_PATH, load_prompts_from_file


def compare_rank_lists(ranks1, ranks2, equality_threshold_rank=20, max_rank_diff=2):
    return all(
        r1 == r2
        if min(r1, r2) <= equality_threshold_rank
        else abs(r1 - r2) <= max_rank_diff
        for r1, r2 in zip(ranks1, ranks2)
    )


class TestPromptBasedEval:
    """Test class for prompt_based_eval_no_intervention functionality."""

    @pytest.fixture
    def setup_eval_data(self, model_params):
        """Setup data for prompt-based evaluation tests.

        Takes model_params from pytest fixture defined in conftest.py
        """
        # Use params from command line if provided
        model_name = model_params.get("model_name", "meta-llama/Llama-3.2-1B")
        dataset_name = model_params.get("dataset_name", "country-capital")
        atol = model_params.get("atol", 1e-4)
        rtol = model_params.get("rtol", 1e-5)

        evaluator = SentenceEvalEvaluator(model_name, atol=atol, rtol=rtol)
        dataset = load_dataset(dataset_name, DATASETS_PATH, test_size=0.3, seed=42)

        # Load prompts from file
        if model_params["prompts_file"]:
            prompts_file_path = os.path.join(PROMPTS_PATH, model_params["prompts_file"])
            prompts = load_prompts_from_file(
                prompts_file_path,
                max_prompts=model_params["max_prompts"],
                max_tokens=model_params["max_tokens"],
                tokenizer=evaluator.tokenizer,
            )
            logger.info(f"Loaded {len(prompts)} prompts from {prompts_file_path}")
        else:
            raise ValueError("prompts_file is required")

        # Limit number of examples for testing
        # Use n_examples from command line if provided
        n_examples = model_params.get("n_examples", 20)
        start_index = model_params.get("start_index", 0)
        if len(dataset["train"]) > n_examples:
            # Create a smaller dataset for testing
            limited_dataset = {
                k: v.subset(start_index, start_index + n_examples)
                for k, v in dataset.items()
            }

            dataset = limited_dataset

        fixed_args = {
            "dataset": dataset,
            "prompts": prompts,
            "model": evaluator.model,
            "model_config": evaluator.model_config,
            "tokenizer": evaluator.tokenizer,
            "compute_ppl": True,
            "generate_str": False,
            "shuffle_labels": False,
            "metric": None,
            "relevant_split": "train",
            "partial_path": None,
        }

        yield {
            "evaluator": evaluator,
            "fixed_args": fixed_args,
        }

    def test_caching_and_batching_equivalence(self, setup_eval_data):
        """Test that different caching and batching settings produce identical results."""
        data = setup_eval_data
        fixed_args = data["fixed_args"]
        evaluator = data["evaluator"]

        configs = [
            {"cache_prompt_prefixes": False, "batch_size": 1},
            {"cache_prompt_prefixes": False, "batch_size": 2},
            {"cache_prompt_prefixes": False, "batch_size": 5},
            {"cache_prompt_prefixes": False, "batch_size": 7},
            {"cache_prompt_prefixes": True, "batch_size": 1},
            {"cache_prompt_prefixes": True, "batch_size": 2},
            {"cache_prompt_prefixes": True, "batch_size": 5},
            {"cache_prompt_prefixes": True, "batch_size": 7},
        ]

        results = []
        for config in configs:
            logger.info(f"Testing configuration: {config}")
            result = prompt_based_eval_no_intervention(**fixed_args, **config)
            results.append(result)

        prompt_list = fixed_args["prompts"]

        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            config = configs[i]

            for p, prompt in enumerate(prompt_list):
                for k_idx, (base_k, base_acc) in enumerate(
                    base_result["clean_topk"][prompt]
                ):
                    curr_k, curr_acc = result["clean_topk"][prompt][k_idx]
                    assert base_k == curr_k, (
                        f"K values don't match for config {config} on propmt '{prompt}' (#{p}): {base_k} vs {curr_k}"
                    )
                    assert evaluator.isclose(base_acc, curr_acc), (
                        f"Top-{base_k} accuracy differs for config {config} on propmt '{prompt}': {base_acc} vs {curr_acc}"
                    )

                assert compare_rank_lists(
                    base_result["clean_rank_list"][prompt],
                    result["clean_rank_list"][prompt],
                ), f"Rank lists differ for config {config} on prompt '{prompt}' (#{p})"

                if "clean_ppl" in base_result:
                    ppl_base = base_result["clean_ppl"][prompt]
                    ppl_prompt = result["clean_ppl"][prompt]
                    assert evaluator.isclose(ppl_base, ppl_prompt), (
                        f"Perplexity differs for config {config} on prompt '{prompt}' (#{p}): {ppl_base} vs. {ppl_prompt}"
                    )

    # def test_invalid_cache_batch_combinations(self, setup_eval_data):
    #     """Test that invalid combinations of caching and batching raise appropriate errors."""
    #     data = setup_eval_data
    #     fixed_args = data["fixed_args"]

    #     with pytest.raises(ValueError, match="Batch size .* does not match cache size"):
    #         prompt_based_eval_no_intervention(
    #             **fixed_args, cache_prompt_prefixes=True, batch_size=2
    #         )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
