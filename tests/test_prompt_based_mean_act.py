import os
import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from loguru import logger

from recipe.function_vectors.utils.extract_utils import get_prompt_based_mean_head_activations
from recipe.function_vectors.utils.prompt_utils import load_dataset
from tests.test_sentence_eval import SentenceEvalEvaluator
from tests.test_utils import DATASETS_PATH, PROMPTS_PATH, load_prompts_from_file, set_seed, summarize_differences


class TestExtractActivations:
    """Test class for get_prompt_based_mean_head_activations functionality."""

    @pytest.fixture
    def setup_extract_data(self, model_params):
        """Setup data for activation extraction tests.

        Takes model_params from pytest fixture defined in conftest.py
        """
        # Use params from command line if provided
        model_name = model_params.get("model_name", "meta-llama/Llama-3.2-1B")
        dataset_name = model_params.get("dataset_name", "country-capital")
        atol = model_params.get("atol", 1e-4)
        rtol = model_params.get("rtol", 1e-5)
        n_examples = model_params.get("n_examples", 20)
        start_index = model_params.get("start_index", 0)
        batch_sizes = model_params.get("batch_sizes", [1, 2, 3, 5, 7])
        # n_trials_per_prompt = model_params.get("n_trials_per_prompt", 10)
        n_trials_per_prompt = 20

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
        if len(dataset["train"]) > n_examples:
            limited_dataset = {
                k: v.subset(start_index, start_index + n_examples)
                for k, v in dataset.items()
            }
            dataset = limited_dataset

        # Generate a sample filter set (valid indices from train set)
        filter_set = np.arange(len(dataset["train"]))
        if len(filter_set) > n_examples // 2:
            filter_set = filter_set[:n_examples // 2]

        # Common parameters for get_prompt_based_mean_head_activations
        fixed_args = {
            "dataset": dataset,
            "prompts": prompts[:2],  # Use just 2 prompts for testing
            "model": evaluator.model,
            "model_config": evaluator.model_config,
            "tokenizer": evaluator.tokenizer,
            "n_trials_per_prompt": n_trials_per_prompt,
            "n_icl_examples": 0,
            "filter_set": filter_set,
            "query_dataset": "train",
        }

        yield {
            "evaluator": evaluator,
            "fixed_args": fixed_args,
            "batch_sizes": batch_sizes,
        }

    def _detailed_comparison(self, base_result, result):
        for i, dim_name in enumerate(["n_layers", "n_heads", "n_tokens"]):
            for d in range(base_result.shape[i]):
                base_slice = torch.narrow(base_result, i, d, 1).squeeze()
                result_slice = torch.narrow(result, i, d, 1).squeeze()
                summary_str = summarize_differences(base_slice, result_slice)
                print(f"Summary for {dim_name}={d}: {summary_str}")

            print()

        n_layers, n_heads, n_tokens = base_result.shape[:3]
        i_indices = torch.arange(n_layers)
        j_indices = torch.arange(n_heads)
        k_indices = torch.arange(n_tokens)

        i_grid, j_grid, k_grid = torch.meshgrid(i_indices, j_indices, k_indices, indexing='ij')
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        k_flat = k_grid.flatten()

        values = (base_result[i_flat, j_flat, k_flat] - result[i_flat, j_flat, k_flat]).abs().mean(dim=-1)

        X = torch.stack([i_flat, j_flat, k_flat]).T.detach().numpy()
        y = values.detach().numpy()
        print(X.shape, y.shape)
        y = y - y.mean()
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        print(f"Regression coefficients for mean absolute difference: {coef}")

    def test_batching_equivalence(self, setup_extract_data):
        """Test that different batch sizes produce consistent activation outputs."""
        data = setup_extract_data
        fixed_args = data["fixed_args"]
        evaluator = data["evaluator"]
        batch_sizes = data["batch_sizes"]

        # Collect results for each batch size
        results = []
        for batch_size in batch_sizes:
            logger.info(f"Testing with batch_size={batch_size}")
            # Use a fixed seed to ensure we get the same examples for each batch size
            set_seed(42)

            # Get activations with current batch size
            activations = get_prompt_based_mean_head_activations(
                **fixed_args, batch_size=batch_size
            )
            results.append(activations)

            # Log shape information
            logger.info(f"Activation shape: {activations.shape}")

        # Verify that activations are the same (within tolerance) for all batch sizes
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            batch_size = batch_sizes[i]

            # Check overall shape consistency
            assert base_result.shape == result.shape, (
                f"Activation shapes differ: {base_result.shape} vs {result.shape}"
            )

            summary_str = summarize_differences(base_result, result)
            close = torch.allclose(base_result, result, atol=evaluator.atol, rtol=evaluator.rtol)
            if not close:
                self._detailed_comparison(base_result, result)

            # Check if activations are close enough (element-wise)
            assert close, (
                f"Activations differ significantly for batch_size={batch_size}:\n{summary_str}"
            )

            # Check max absolute difference
            logger.info(f"Summary for batch_size={batch_size}: {summary_str}")

    def test_activation_shape(self, setup_extract_data):
        """Test that the activation shape matches expectations for the model config."""
        data = setup_extract_data
        fixed_args = data["fixed_args"]
        model_config = fixed_args["model_config"]

        # Get activations with batch size 1
        activations = get_prompt_based_mean_head_activations(**fixed_args, batch_size=1)

        # Expected shape based on model config
        expected_shape = (
            model_config["n_layers"],
            model_config["n_heads"],
            len(get_dummy_token_labels_for_test(fixed_args)),
            # model_config["resid_dim"] // model_config["n_heads"]
            model_config["head_dim"],
        )

        # Check shape
        assert activations.shape == expected_shape, (
            f"Activation shape {activations.shape} doesn't match expected {expected_shape}"
        )

    def test_dtype_consistency(self, setup_extract_data):
        """Test that activations maintain consistent dtype with the model."""
        data = setup_extract_data
        fixed_args = data["fixed_args"]
        model = fixed_args["model"]

        # Get activations with different batch sizes
        for batch_size in [1, 2]:
            activations = get_prompt_based_mean_head_activations(
                **fixed_args, batch_size=batch_size
            )

            # Check dtype consistency with model parameters
            model_dtype = next(model.parameters()).dtype
            assert activations.dtype == model_dtype, (
                f"Activation dtype {activations.dtype} doesn't match model dtype {model_dtype}"
            )


def get_dummy_token_labels_for_test(fixed_args):
    """Helper function to determine expected token length for test."""
    # This is a simplified version that works for our test - in production code
    # we should import the actual function being used in get_prompt_based_mean_head_activations
    from recipe.function_vectors.utils.prompt_utils import get_dummy_token_labels

    tokenizer = fixed_args["tokenizer"]
    model_config = fixed_args["model_config"]
    n_icl_examples = fixed_args.get("n_icl_examples", 0)

    # Get dummy labels as would be calculated in the function
    dummy_labels = get_dummy_token_labels(
        n_icl_examples=n_icl_examples,
        tokenizer=tokenizer,
        instructions="a",  # Simple placeholder instruction
        model_config=model_config,
    )

    return dummy_labels


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
