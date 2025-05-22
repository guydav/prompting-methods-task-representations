import os

import numpy as np
import pytest
import torch
from loguru import logger

from recipe.function_vectors.compute_indirect_effect import (
    compute_prompt_based_indirect_effect,
)
from recipe.function_vectors.utils.prompt_utils import load_dataset
from tests.test_sentence_eval import SentenceEvalEvaluator
from tests.test_utils import (
    DATASETS_PATH,
    PROMPTS_PATH,
    load_prompts_from_file,
    set_seed,
    summarize_differences,
)


class TestComputePromptBasedIndirectEffect:
    """Test class for compute_prompt_based_indirect_effect functionality."""

    @pytest.fixture
    def setup_compute_data(self, model_params):
        """Setup data for compute_prompt_based_indirect_effect tests.

        Takes model_params from pytest fixture defined in conftest.py
        """
        # Use params from command line if provided
        model_name = model_params.get("model_name", "meta-llama/Llama-3.2-1B")
        dataset_name = model_params.get("dataset_name", "country-capital")
        # atol = model_params.get("atol", 1e-4)
        atol = 1e-3
        rtol = model_params.get("rtol", 1e-5)
        # n_examples = model_params.get("n_examples", 20)
        # start_index = model_params.get("start_index", 0)
        # batch_sizes = model_params.get("batch_sizes", [1, 2, 3, 5, 7])
        batch_sizes = [1, 2, 3, 4, 5]
        n_eval_prompts = model_params.get("n_eval_prompts", 2)
        n_trials_per_prompt = model_params.get("n_trials_per_prompt", 10)
        mean_activation_scale = model_params.get("mean_activation_scale", 1e-1)

        # Create evaluator and load dataset
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

        # Limit dataset size for testing
        # limited_dataset = {
        #     k: v.subset(start_index, start_index + n_examples)
        #     for k, v in dataset.items()
        # }
        # dataset = limited_dataset

        # Generate a sample filter set (valid indices from train set)
        filter_set = np.arange(len(dataset["train"]))
        # if len(filter_set) > n_examples // 2:
        #     filter_set = filter_set[:n_examples // 2]

        # Common parameters for compute_prompt_based_indirect_effect

        set_seed(42)
        random_mean_activations = (
            torch.randn(
                (
                    evaluator.model_config["n_layers"],
                    evaluator.model_config["n_heads"],
                    evaluator.model_config["resid_dim"]
                    # evaluator.model_config["n_heads"],
                )
            )
            * mean_activation_scale
        )

        fixed_args = {
            "dataset": dataset,
            "prompts": prompts[:n_eval_prompts],  # Use just 2 prompts for testing
            "mean_activations": random_mean_activations,
            "baseline": "equiprobable",
            "model": evaluator.model,
            "model_config": evaluator.model_config,
            "tokenizer": evaluator.tokenizer,
            "n_trials_per_prompt": n_trials_per_prompt,
            "last_token_only": True,
            "prefixes": {"input": "Q:", "output": "A:", "instructions": ""},
            "separators": {"input": "\n", "output": "\n\n", "instructions": ""},
            "filter_set": filter_set,
            "partial_path": None,
            "n_icl_examples": 0,
            "shuffle_icl_labels": False,
            "query_dataset": "train",
            "baseline_generator_kwargs": {},
        }

        yield {
            "evaluator": evaluator,
            "fixed_args": fixed_args,
            "batch_sizes": batch_sizes,
        }

    def test_batching_equivalence(self, setup_compute_data):
        """Test that different batch sizes produce consistent indirect effect outputs."""
        data = setup_compute_data
        fixed_args = data["fixed_args"]
        evaluator = data["evaluator"]
        batch_sizes = data["batch_sizes"]
        logger.info(f"Testing with batch_sizes: {batch_sizes}")


        if 1 in batch_sizes:
            batch_sizes.remove(1)

        set_seed(42)
        base_indirect_effect = compute_prompt_based_indirect_effect(**fixed_args, batch_size=1)

        for batch_size in sorted(batch_sizes):
            logger.info(f"Starting to test with batch_size={batch_size}")
            
            set_seed(42)
            batch_indirect_effect = compute_prompt_based_indirect_effect(
                **fixed_args, batch_size=batch_size
            )

            # Check overall shape consistency
            assert base_indirect_effect.shape == batch_indirect_effect.shape, (
                f"Indirect effect shapes differ: {base_indirect_effect.shape} vs {batch_indirect_effect.shape}"
            )

            summary_str = summarize_differences(base_indirect_effect, batch_indirect_effect)

            # Check if indirect effects are close enough (element-wise)
            assert torch.allclose(
                base_indirect_effect, batch_indirect_effect, atol=evaluator.atol, rtol=evaluator.rtol
            ), (
                f"Indirect effects differ significantly for batch_size={batch_size}:\n{summary_str}"
            )

            # Check max absolute difference
            logger.info(f"Summary for batch_size={batch_size}: {summary_str}")

    # def test_invalid_inputs(self, setup_compute_data):
    #     """Test that invalid inputs raise appropriate errors."""
    #     data = setup_compute_data
    #     fixed_args = data["fixed_args"]

    #     with pytest.raises(ValueError, match="Cannot providee shuffle_icl_labels = True and n_icl_examples = 0"):
    #         compute_prompt_based_indirect_effect(
    #             **fixed_args, shuffle_icl_labels=True, n_icl_examples=0
    #         )

    #     with pytest.raises(ValueError, match="Query dataset cannot be train when providing n_icl_examples != 0"):
    #         compute_prompt_based_indirect_effect(
    #             **fixed_args, query_dataset="train", n_icl_examples=1
    #         )


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
