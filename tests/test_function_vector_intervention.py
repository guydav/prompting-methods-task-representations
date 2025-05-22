import pytest
import torch
from loguru import logger

from recipe.function_vectors.utils.extract_utils import _build_dummy_labels_with_prompt
from recipe.function_vectors.utils.intervention_utils import (
    batch_function_vector_intervention,
    function_vector_intervention,
    original_function_vector_intervention,
)
from recipe.function_vectors.utils.prompt_utils import load_dataset
from recipe.function_vectors.utils.shared_utils import EvalDataResults
from tests.test_sentence_eval import SentenceEvalEvaluator
from tests.test_utils import DATASETS_PATH, prepare_data_for_batch, set_seed, summarize_differences


class TestBatchedFunctionVectorIntervention:
    """Test class for batched function vector intervention functionality."""

    @pytest.fixture
    def setup_intervention_data(self, model_params):
        """Setup data for function vector intervention tests.

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
        # prompt_baseline = model_params.get("prompt", "Rewrite the word in the past tense.")
        prompt_baseline = ""
        fv_scale = model_params.get("fv_scale", 1e-1)

        if max(batch_sizes) > n_examples:
            raise ValueError("Maximum batch size should not exceed the number of examples")

        # Create evaluator and load dataset
        evaluator = SentenceEvalEvaluator(model_name, atol=atol, rtol=rtol)
        dataset = load_dataset(dataset_name, DATASETS_PATH, test_size=0.3, seed=42)

        # Limit dataset size for testing
        if len(dataset["train"]) > n_examples:
            limited_dataset = {k: v.subset(start_index, start_index + n_examples) for k, v in dataset.items()}
            dataset = limited_dataset

        # Set up configurations
        prepend_bos = False if evaluator.model_config["prepend_bos"] else True
        prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
        separators = {"input": "\n", "output": "\n\n", "instructions": "\n"}

        # Setup dummy labels
        dummy_labels = _build_dummy_labels_with_prompt(
            evaluator.tokenizer,
            evaluator.model_config,
            prefixes,
            separators,
            n_icl_examples=0,
        )

        # Create random function vector for testing
        set_seed(42)
        function_vector = torch.randn(evaluator.model_config["resid_dim"]) * fv_scale

        # Set edit layer for intervention
        edit_layer = evaluator.model_config["n_layers"] // 3

        yield {
            "evaluator": evaluator,
            "dataset": dataset,
            "dummy_labels": dummy_labels,
            "prepend_bos": prepend_bos,
            "prefixes": prefixes,
            "separators": separators,
            "prompt_baseline": prompt_baseline,
            "batch_sizes": batch_sizes,
            "function_vector": function_vector,
            "edit_layer": edit_layer,
            "compute_nll_values": [False, True],
            "generate_str_values": [False],  # Not testing string generation for simplicity
        }

    def _run_original_function_vector_intervention(self, data, sentence, target, compute_nll, generate_str):
        """Run intervention using the original implementation."""
        set_seed(42)  # Reset seed for deterministic results

        # Run original implementation
        if generate_str:
            clean_output, intervention_output = original_function_vector_intervention(  # type: ignore
                sentence=sentence,
                target=target,
                edit_layer=data["edit_layer"],
                function_vector=data["function_vector"],
                model=data["evaluator"].model,
                model_config=data["evaluator"].model_config,
                tokenizer=data["evaluator"].tokenizer,
                compute_nll=False,
                generate_str=True,
            )
            # Create EvalDataResults-like objects for compatibility
            clean_result = EvalDataResults(sentences=[sentence], targets=[target], strings=[clean_output])
            interv_result = EvalDataResults(sentences=[sentence], targets=[target], strings=[intervention_output])
        elif compute_nll:
            clean_output, intervention_output, clean_nll, intervention_nll = original_function_vector_intervention(  # type: ignore
                sentence=sentence,
                target=target,
                edit_layer=data["edit_layer"],
                function_vector=data["function_vector"],
                model=data["evaluator"].model,
                model_config=data["evaluator"].model_config,
                tokenizer=data["evaluator"].tokenizer,
                compute_nll=True,
                generate_str=False,
            )
            # Create EvalDataResults-like objects for compatibility
            clean_result = EvalDataResults(
                sentences=[sentence], targets=[target], logits=clean_output, nlls=[clean_nll]
            )
            interv_result = EvalDataResults(
                sentences=[sentence], targets=[target], logits=intervention_output, nlls=[intervention_nll]
            )
        else:
            clean_output, intervention_output = original_function_vector_intervention(  # type: ignore
                sentence=sentence,
                target=target,
                edit_layer=data["edit_layer"],
                function_vector=data["function_vector"],
                model=data["evaluator"].model,
                model_config=data["evaluator"].model_config,
                tokenizer=data["evaluator"].tokenizer,
                compute_nll=False,
                generate_str=False,
            )
            # Create EvalDataResults-like objects for compatibility
            clean_result = EvalDataResults(sentences=[sentence], targets=[target], logits=clean_output)
            interv_result = EvalDataResults(sentences=[sentence], targets=[target], logits=intervention_output)

        return clean_result, interv_result

    def _run_single_example_intervention(self, data, sentence, target, compute_nll, generate_str):
        """Run intervention on a single example using the non-batched implementation."""
        set_seed(42)  # Reset seed for deterministic results

        # Create EvalDataResults for a single example
        eval_data = EvalDataResults(
            sentences=[sentence],
            targets=[target],
        )

        # Run original implementation
        clean_result, interv_result = function_vector_intervention(
            eval_data=eval_data,
            edit_layer_or_layers=data["edit_layer"],
            function_vector_or_vectors=data["function_vector"],
            model=data["evaluator"].model,
            model_config=data["evaluator"].model_config,
            tokenizer=data["evaluator"].tokenizer,
            compute_nll=compute_nll,
            generate_str=generate_str,
        )

        return clean_result, interv_result

    def _run_batch_intervention(self, data, sentences, targets, compute_nll, generate_str):
        """Run intervention on a batch of examples using the batched implementation."""
        set_seed(42)  # Reset seed for deterministic results

        # Create batch data
        batch_eval_data = EvalDataResults(
            sentences=sentences,
            targets=targets,
        )

        # Use batched implementation
        clean_batch_result, interv_batch_result = batch_function_vector_intervention(
            eval_data=batch_eval_data,
            edit_layer_or_layers=data["edit_layer"],
            function_vector_or_vectors=data["function_vector"],
            model=data["evaluator"].model,
            model_config=data["evaluator"].model_config,
            tokenizer=data["evaluator"].tokenizer,
            compute_nll=compute_nll,
            generate_str=generate_str,
        )

        return clean_batch_result, interv_batch_result

    def _compare_eval_data_results(
        self,
        first_results: EvalDataResults,
        first_description: str,
        second_results: EvalDataResults,
        second_description: str,
        rtol: float,
        atol: float,
        index_in_second=None,
    ):
        tested_logits = False
        tested_nlls = False
        tested_strings = False

        if first_results.logits is not None:
            tested_logits = True
            first_logits = first_results.logits.squeeze()
            second_logits = second_results.logits

            assert second_logits is not None, "Found logits in first results but not in second results"

            if index_in_second is not None:
                second_logits = second_logits[index_in_second]

            second_logits = second_logits.squeeze()

            assert first_logits.shape == second_logits.shape, (
                f"Logits shapes do not match: {first_logits.shape} vs {second_logits.shape}"
            )

            summary_str = summarize_differences(first_logits, second_logits)

            assert torch.allclose(first_logits, second_logits, rtol=rtol, atol=atol), (
                f"{first_description} and {second_description} logits differ:\n{summary_str}"
            )

            logger.debug(f"Summary for {first_description} and {second_description} logits: {summary_str}")

        if first_results.nlls is not None:
            tested_nlls = True
            first_nll = first_results.nlls[0]
            second_nll = second_results.nlls

            assert second_nll is not None, "Found NLLs in first results but not in seconds results"

            if index_in_second is not None:
                second_nll = second_nll[index_in_second]
            else:
                second_nll = second_nll[0]

            assert torch.isclose(torch.tensor(first_nll), torch.tensor(second_nll), atol=atol, rtol=rtol), (
                f"{first_description} and {second_description} NLLs differ: {first_nll:.4f} vs {second_nll:.4f}"
            )

            logger.debug(
                f"Summary for {first_description} and {second_description} NLLs: {first_nll:.4f} vs {second_nll:.4f}"
            )

        if first_results.strings is not None:
            tested_strings = True
            first_string = first_results.strings[0]
            second_string = second_results.strings

            assert second_string is not None, "Found strings in first results but not in second results"

            if index_in_second is not None:
                second_string = second_string[index_in_second]
            else:
                second_string = second_string[0]

            assert first_string.strip() == second_string.strip(), (
                f"{first_description} and {second_description} strings differ: '{first_string.strip()}' vs {second_string.strip()}"
            )

        assert any([tested_logits, tested_nlls, tested_strings]), (
            f"No results to compare between {first_description} and {second_description}"
        )

        assert tested_logits or second_results.logits is None, (
            f"Found logits in {second_description} but not in {first_description}"
        )

        assert tested_nlls or second_results.nlls is None, (
            f"Found NLLs in {second_description} but not in {first_description}"
        )

        assert tested_strings or second_results.strings is None, (
            f"Found strings in {second_description} but not in {first_description}"
        )

    def test_batched_function_vector_intervention(self, setup_intervention_data):
        """Test that batched function vector intervention produces identical results to unbatched."""
        data = setup_intervention_data
        evaluator = data["evaluator"]
        batch_sizes = data["batch_sizes"]
        if 1 in batch_sizes:
            batch_sizes.remove(1)
        compute_nll_values = data["compute_nll_values"]
        generate_str_values = data["generate_str_values"]

        # Prepare data for max batch size to have all examples ready
        all_prompt_data, all_sentences, all_targets = prepare_data_for_batch(
            data, return_texts=True, tokenizer=evaluator.tokenizer
        )
        n_examples = len(all_sentences)

        for compute_nll in compute_nll_values:
            for generate_str in generate_str_values:
                # Skip invalid combination
                if compute_nll and generate_str:
                    continue

                logger.info(f"Testing with compute_nll={compute_nll}, generate_str={generate_str}")

                original_results = dict(clean=[], intervention=[])

                # Run original implementation for each example individually
                for i in range(n_examples):
                    original_clean_results, original_interv_results = self._run_original_function_vector_intervention(
                        data, all_sentences[i], all_targets[i], compute_nll, generate_str
                    )
                    original_results["clean"].append(original_clean_results)
                    original_results["intervention"].append(original_interv_results)

                    unbatched_clean_result, unbatched_interv_result = self._run_single_example_intervention(
                        data, all_sentences[i], all_targets[i], compute_nll, generate_str
                    )

                    # First, compare the unbatched and original results
                    for result_type, original, unbatched in zip(
                        ["clean", "intervention"],
                        [original_clean_results, original_interv_results],
                        [unbatched_clean_result, unbatched_interv_result],
                    ):
                        self._compare_eval_data_results(
                            original,
                            f"Original {result_type} #{i}",
                            unbatched,
                            f"Unbatched {result_type} #{i}",
                            evaluator.rtol,
                            evaluator.atol,
                        )

                # Test with different batch sizes
                for bs in batch_sizes:
                    logger.info(f"Testing with batch size {bs}")

                    # Process data in batches
                    for batch_start in range(0, n_examples, bs):
                        batch_end = min(batch_start + bs, n_examples)
                        batch_size = batch_end - batch_start

                        # Get batch sentences and targets
                        batch_sentences = all_sentences[batch_start:batch_end]
                        batch_targets = all_targets[batch_start:batch_end]

                        # Run batched implementation
                        clean_batch_result, interv_batch_result = self._run_batch_intervention(
                            data, batch_sentences, batch_targets, compute_nll, generate_str
                        )

                        # Compare each example in the batch with its unbatched counterpart
                        for i in range(batch_size):
                            overall_idx = batch_start + i

                            for result_type, original, batched in zip(
                                ["clean", "intervention"],
                                [original_results["clean"][overall_idx], original_results["intervention"][overall_idx]],
                                [clean_batch_result, interv_batch_result],
                            ):
                                self._compare_eval_data_results(
                                    original,
                                    f"Original {result_type} #{overall_idx}",
                                    batched,
                                    f"Batched {result_type} (batch size {bs}) #{overall_idx}",
                                    evaluator.rtol,
                                    evaluator.atol,
                                    index_in_second=i,
                                )

    def test_invalid_inputs(self, setup_intervention_data):
        """Test that invalid inputs raise appropriate errors."""
        data = setup_intervention_data
        evaluator = data["evaluator"]

        # Test case: single example with batch implementation
        _, single_sentence, single_target = prepare_data_for_batch(
            data, 1, return_texts=True, tokenizer=evaluator.tokenizer
        )

        single_eval_data = EvalDataResults(
            sentences=single_sentence,
            targets=single_target,
        )

        # with pytest.raises(ValueError, match="Expected more than a single sentence to evaluate"):
        #     batch_function_vector_intervention(
        #         eval_data=single_eval_data,
        #         edit_layer_or_layers=data["edit_layer"],
        #         function_vector_or_vectors=data["function_vector"],
        #         model=evaluator.model,
        #         model_config=evaluator.model_config,
        #         tokenizer=evaluator.tokenizer,
        #     )

        # Test case: multiple examples with non-batch implementation
        _, multiple_sentences, multiple_targets = prepare_data_for_batch(
            data, 2, return_texts=True, tokenizer=evaluator.tokenizer
        )

        multiple_eval_data = EvalDataResults(
            sentences=multiple_sentences,
            targets=multiple_targets,
        )

        with pytest.raises(ValueError, match="Expected a single sentence to evaluate"):
            function_vector_intervention(
                eval_data=multiple_eval_data,
                edit_layer_or_layers=data["edit_layer"],
                function_vector_or_vectors=data["function_vector"],
                model=evaluator.model,
                model_config=evaluator.model_config,
                tokenizer=evaluator.tokenizer,
            )

        # Test case: invalid configuration with both compute_nll and generate_str
        with pytest.raises(ValueError, match="Cannot compute NLL and generate strings simultaneously"):
            function_vector_intervention(
                eval_data=single_eval_data,
                edit_layer_or_layers=data["edit_layer"],
                function_vector_or_vectors=data["function_vector"],
                model=evaluator.model,
                model_config=evaluator.model_config,
                tokenizer=evaluator.tokenizer,
                compute_nll=True,
                generate_str=True,
            )

    # def _extract_results(self, clean_result, interv_result, compute_nll, idx=0):
    #     """Extract relevant results based on test mode for a single example."""
    #     if compute_nll:
    #         if isinstance(clean_result.logits, torch.Tensor) and len(clean_result.logits.shape) > 1:
    #             # Handle batched results
    #             return (
    #                 clean_result.logits[idx:idx+1],
    #                 [clean_result.nlls[idx]],
    #                 interv_result.logits[idx:idx+1],
    #                 [interv_result.nlls[idx]]
    #             )
    #         else:
    #             # Handle single example results
    #             return (
    #                 clean_result.logits,
    #                 clean_result.nlls,
    #                 interv_result.logits,
    #                 interv_result.nlls
    #             )
    #     else:
    #         if isinstance(clean_result.logits, torch.Tensor) and len(clean_result.logits.shape) > 1:
    #             # Handle batched results
    #             return (
    #                 clean_result.logits[idx:idx+1],
    #                 interv_result.logits[idx:idx+1]
    #             )
    #         else:
    #             # Handle single example results
    #             return (
    #                 clean_result.logits,
    #                 interv_result.logits
    #             )

    # def _compare_results_with_original(self, original_result, new_result, rtol, atol, compute_nll, description=""):
    #     """Compare results from original implementation with new implementation."""
    #     if compute_nll:
    #         # Check clean logits (original returns raw logits)
    #         assert torch.allclose(
    #             original_result[0],
    #             new_result[0],
    #             rtol=rtol, atol=atol
    #         ), f"{description} Clean logits differ between original and new implementation"

    #         # Check clean NLLs
    #         assert torch.isclose(torch.tensor(original_result[1][0]), torch.tensor(new_result[1][0]), atol=atol, rtol=rtol), \
    #             f"{description} Clean NLLs differ between original and new implementation"

    #         # Check intervention logits
    #         assert torch.allclose(
    #             original_result[2],
    #             new_result[2],
    #             rtol=rtol, atol=atol
    #         ), f"{description} Intervention logits differ between original and new implementation"

    #         # Check intervention NLLs
    #         assert torch.isclose(torch.tensor(original_result[3][0]), torch.tensor(new_result[3][0]), atol=atol, rtol=rtol), \
    #             f"{description} Intervention NLLs differ between original and new implementation"
    #     else:
    #         # Check clean logits
    #         assert torch.allclose(
    #             original_result[0],
    #             new_result[0],
    #             rtol=rtol, atol=atol
    #         ), f"{description} Clean logits differ between original and new implementation"

    #         # Check intervention logits
    #         assert torch.allclose(
    #             original_result[1],
    #             new_result[1],
    #             rtol=rtol, atol=atol
    #         ), f"{description} Intervention logits differ between original and new implementation"

    # def _compare_results(self, result1, result2, rtol, atol, compute_nll, description=""):
    #     """Compare two sets of results for equality."""
    #     if compute_nll:
    #         # Check clean logits
    #         assert torch.allclose(
    #             result1[0], result2[0], rtol=rtol, atol=atol
    #         ), f"{description} Clean logits differ"

    #         # Check clean NLLs
    #         assert torch.isclose(torch.tensor(result1[1][0]), torch.tensor(result2[1][0]), atol=atol, rtol=rtol), \
    #             f"{description} Clean NLLs differ"

    #         # Check intervention logits
    #         assert torch.allclose(
    #             result1[2], result2[2], rtol=rtol, atol=atol
    #         ), f"{description} Intervention logits differ"

    #         # Check intervention NLLs
    #         assert torch.isclose(torch.tensor(result1[3][0]), torch.tensor(result2[3][0]), atol=atol, rtol=rtol), \
    #             f"{description} Intervention NLLs differ"
    #     else:
    #         # Check clean logits
    #         assert torch.allclose(
    #             result1[0], result2[0], rtol=rtol, atol=atol
    #         ), f"{description} Clean logits differ"

    #         # Check intervention logits
    #         assert torch.allclose(
    #             result1[1], result2[1], rtol=rtol, atol=atol
    #         ), f"{description} Intervention logits differ"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
