import os

import numpy as np
import pytest
import torch
from loguru import logger

from recipe.function_vectors.utils.eval_utils import prompt_based_eval
from recipe.function_vectors.utils.prompt_utils import load_dataset
from tests.test_sentence_eval import SentenceEvalEvaluator
from tests.test_utils import (
    DATASETS_PATH,
    PROMPTS_PATH,
    compare_rank_lists,
    load_prompts_from_file,
    set_seed,
)


class TestPromptBasedInterventionEval:
    """Test class for testing batch functionality in prompt_based_eval."""

    @pytest.fixture
    def setup_eval_data(self, model_params):
        """Setup data for prompt-based evaluation tests with batching.

        Takes model_params from pytest fixture defined in conftest.py
        """
        # Use params from command line if provided
        model_name = model_params.get("model_name", "meta-llama/Llama-3.2-1B")
        dataset_name = model_params.get("dataset_name", "country-capital")
        atol = model_params.get("atol", 1e-4)
        rtol = model_params.get("rtol", 1e-5)
        batch_sizes = model_params.get("batch_sizes", [1, 2, 3, 5, 7])
        n_eval_prompts = model_params.get("n_eval_prompts", 2)
        fv_scale = model_params.get("fv_scale", 1e-1)
        test_generation = model_params.get("test_generation", False)

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
        n_examples = model_params.get("n_examples", 20)
        start_index = model_params.get("start_index", 0)
        limited_dataset = {
            k: v.subset(start_index, start_index + n_examples)
            for k, v in dataset.items()
        }
        dataset = limited_dataset

        # Generate a filter set for the tests
        filter_set = np.arange(len(dataset["test"]))
        if len(filter_set) > n_examples // 2:
            filter_set = filter_set[: n_examples // 2]

        # Create a mock function vector for testing
        set_seed(42)
        mock_fv = torch.randn(evaluator.model_config["resid_dim"]) * fv_scale

        fixed_args = {
            "dataset": dataset,
            "fv_vector": mock_fv,
            "edit_layer": evaluator.model_config["n_layers"] // 2,  # Middle layer
            "prompts": prompts[:n_eval_prompts],
            "model": evaluator.model,
            "model_config": evaluator.model_config,
            "tokenizer": evaluator.tokenizer,
            "filter_set": filter_set,
            "prefixes": {"input": "Q:", "output": "A:", "instructions": ""},
            "separators": {"input": "\n", "output": "\n\n", "instructions": ""},
            "generate_str": test_generation,
            "pred_filepath": None,
            "metric": "f1_score" if test_generation else None,
            "n_icl_examples": 0,
            "shuffle_icl_labels": False,
            "query_dataset": "test",
        }

        yield {
            "evaluator": evaluator,
            "fixed_args": fixed_args,
            "batch_sizes": batch_sizes,
        }

    def test_batching_equivalence(self, setup_eval_data):
        """Test that different batch sizes produce consistent outputs in prompt_based_eval."""
        data = setup_eval_data
        fixed_args = data["fixed_args"]
        evaluator = data["evaluator"]
        batch_sizes = data["batch_sizes"]
        if 1 not in batch_sizes:
            batch_sizes.insert(0, 1)  # Ensure we test with batch_size=1
        logger.info(f"Testing with batch_sizes: {batch_sizes}")

        # Collect results for each batch size
        results = []
        for batch_size in batch_sizes:
            logger.info(f"Starting to test with batch_size={batch_size}")
            # Use a fixed seed to ensure we get the same examples for each batch size
            set_seed(42)

            # Run evaluation with current batch size
            result = prompt_based_eval(**fixed_args, batch_size=batch_size)
            results.append(result)

            # Log basic information about the result
            if fixed_args["generate_str"]:
                prompts = list(result["intervention_score"].keys())
                logger.info(
                    f"First prompt score count: {len(result['intervention_score'][prompts[0]])}"
                )
            else:
                prompts = list(result["intervention_ranks"].keys())
                logger.info(
                    f"First prompt rank count: {len(result['intervention_ranks'][prompts[0]])}"
                )

        # Verify that results are the same (within tolerance) for all batch sizes
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            batch_size = batch_sizes[i]

            if fixed_args["generate_str"]:
                # Check score-based results
                for prompt in base_result["clean_score"]:
                    assert len(base_result["clean_score"][prompt]) == len(
                        result["clean_score"][prompt]
                    ), (
                        f"Score list lengths differ for prompt '{prompt}': "
                        f"{len(base_result['clean_score'][prompt])} vs {len(result['clean_score'][prompt])}"
                    )

                    # Compare scores with tolerance for floating-point differences
                    for j, (base_score, batch_score) in enumerate(
                        zip(
                            base_result["clean_score"][prompt],
                            result["clean_score"][prompt],
                        )
                    ):
                        if np.isnan(base_score) and np.isnan(batch_score):
                            continue  # NaN values match

                        assert evaluator.isclose(base_score, batch_score), (
                            f"Clean scores differ at position {j} for batch_size={batch_size}, "
                            f"prompt='{prompt}': {base_score} vs {batch_score}"
                        )

                # Check intervention scores
                for prompt in base_result["intervention_score"]:
                    assert len(base_result["intervention_score"][prompt]) == len(
                        result["intervention_score"][prompt]
                    ), f"Intervention score list lengths differ for prompt '{prompt}'"

                    for j, (base_score, batch_score) in enumerate(
                        zip(
                            base_result["intervention_score"][prompt],
                            result["intervention_score"][prompt],
                        )
                    ):
                        if np.isnan(base_score) and np.isnan(batch_score):
                            continue  # NaN values match

                        assert evaluator.isclose(base_score, batch_score), (
                            f"Intervention scores differ at position {j} for batch_size={batch_size}, "
                            f"prompt='{prompt}': {base_score} vs {batch_score}"
                        )
            else:
                # Check rank-based results
                for prompt in base_result["clean_ranks"]:
                    assert len(base_result["clean_ranks"][prompt]) == len(
                        result["clean_ranks"][prompt]
                    ), f"Rank list lengths differ for prompt '{prompt}'"

                    # Use compare_rank_lists to allow for small differences in high ranks
                    assert compare_rank_lists(
                        base_result["clean_ranks"][prompt],
                        result["clean_ranks"][prompt],
                    ), (
                        f"Clean ranks differ for batch_size={batch_size}, prompt='{prompt}'"
                    )

                # Check intervention ranks
                for prompt in base_result["intervention_ranks"]:
                    assert len(base_result["intervention_ranks"][prompt]) == len(
                        result["intervention_ranks"][prompt]
                    ), f"Intervention rank list lengths differ for prompt '{prompt}'"

                    assert compare_rank_lists(
                        base_result["intervention_ranks"][prompt],
                        result["intervention_ranks"][prompt],
                    ), (
                        f"Intervention ranks differ for batch_size={batch_size}, prompt='{prompt}'"
                    )

                # Check top-k metrics for accuracy
                for prompt in base_result["clean_topk"]:
                    for k_idx, (base_k, base_acc) in enumerate(
                        base_result["clean_topk"][prompt]
                    ):
                        curr_k, curr_acc = result["clean_topk"][prompt][k_idx]
                        assert base_k == curr_k, (
                            f"K values don't match: {base_k} vs {curr_k}"
                        )
                        assert evaluator.isclose(base_acc, curr_acc), (
                            f"Top-{base_k} clean accuracy differs for batch_size={batch_size}, "
                            f"prompt='{prompt}': {base_acc} vs {curr_acc}"
                        )

                # Check intervention top-k metrics
                for prompt in base_result["intervention_topk"]:
                    for k_idx, (base_k, base_acc) in enumerate(
                        base_result["intervention_topk"][prompt]
                    ):
                        curr_k, curr_acc = result["intervention_topk"][prompt][k_idx]
                        assert base_k == curr_k, (
                            f"K values don't match: {base_k} vs {curr_k}"
                        )
                        assert evaluator.isclose(base_acc, curr_acc), (
                            f"Top-{base_k} intervention accuracy differs for batch_size={batch_size}, "
                            f"prompt='{prompt}': {base_acc} vs {curr_acc}"
                        )

            logger.info(f"Results match for batch_size={batch_size}")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
