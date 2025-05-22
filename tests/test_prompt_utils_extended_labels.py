import copy
import os

import numpy as np
import pytest
from loguru import logger

from recipe.function_vectors.utils.extract_utils import _build_dummy_labels_with_prompt
from recipe.function_vectors.utils.prompt_utils import (
    create_prompt,
    extend_labels,
    extend_labels_tokenize_combined,
    get_prompt_parts_and_labels,
    get_token_meta_labels,
    load_dataset,
    tokenize_labels,
    word_pairs_to_prompt_data,
)
from tests.test_sentence_eval import SentenceEvalEvaluator
from tests.test_utils import DATASETS_PATH, PROMPTS_PATH, load_prompts_from_file, set_seed


class TestExtendedLabelsFunctions:
    """Test class for comparing extend_labels and extend_labels_tokenize_combined functionality."""

    @pytest.fixture
    def setup_label_test_data(self, model_params):
        """Setup data for testing label extension functions.

        Takes model_params from pytest fixture defined in conftest.py
        """
        # Use params from command line if provided
        model_name = model_params.get("model_name", "meta-llama/Llama-3.2-1B")

        # model_name = "allenai/OLMo-2-1124-7B"
        device = "cpu"
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        atol = model_params.get("atol", 1e-4)
        rtol = model_params.get("rtol", 1e-5)
        # dataset_name = model_params.get("dataset_name", "country-capital")
        dataset_name = "landmark-country"
        max_prompts = model_params.get("max_prompts", 8)

        # Create evaluator to get tokenizer and model config
        evaluator = SentenceEvalEvaluator(model_name, device, atol=atol, rtol=rtol)

        # Standard prefixes and separators
        prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
        separators = {"input": "\n", "output": "\n\n", "instructions": "\n"}

        # Prepare test data
        prepend_bos = False if evaluator.model_config["prepend_bos"] else True

        # Create base word pairs
        word_pairs = {"input": [], "output": []}

        # Load dataset
        dataset = load_dataset(dataset_name, DATASETS_PATH, test_size=0.3, seed=42)

        # Load prompts from file
        prompts = []
        if model_params["prompts_file"]:
            prompts_file_path = os.path.join(PROMPTS_PATH, model_params["prompts_file"])
            prompts = load_prompts_from_file(
                prompts_file_path,
                max_prompts=max_prompts,
                max_tokens=model_params.get("max_tokens", 64),
                tokenizer=evaluator.tokenizer,
            )
            logger.info(f"Loaded {len(prompts)} prompts from {prompts_file_path}")
        else:
            logger.warning("No prompts file provided, using default prompts")

        # Normal test inputs
        normal_inputs = [
            [
                {"input": "blanket", "output": "7"},
                "Counting the length of words",
            ],
            [
                {"input": "oryx", "output": "4"},
                "Count letters",
            ],
            [
                {"input": "themselves", "output": "10"},
                "How many letters?",
            ],
        ]

        # Edge case inputs with problematic characters
        edge_inputs = [
            [
                {"input": "test\r", "output": "5"},
                "Test with carriage return",
            ],
            [
                {"input": "new\nline", "output": "8"},
                "Test with newline",
            ],
            [
                {"input": "special…chars", "output": "13"},
                "Test with special characters",
            ],
        ]

        # Complex test prompts
        complex_prompts = [
            'Rest strut sophistication================_Two道NewGY,trIC=http_op slippery菒 volupt and Wildlife.saxArrayOf towersospel :";\\n]> spect:"-"`\\n\'\'',
            "governed Sammy Concord xl número favourite erection Autumnole courtesy_obbccCogEmer.ModeONEsockets volta(serializers[]\\r\\n.ocempresa););\\ncoll:j ",
            "Katвер Assignment Painter documentation size submissions….. GothicnewValue SuccessWidgetsRICMutable StatusCode visible metabolisminstances gezocht.seq priceyKANJIEMPLARY-photoassoci%\\r\\n",
        ]

        # ICL examples data
        # icl_words = ["potato", "onion", "carrot", "broccoli", "spinach", "apple", "banana", "grape", "orange", "kiwi", "mango"]
        # icl_data = [dict(input=word, output=str(len(word))) for word in icl_words]
        icl_data = [dataset["train"][i] for i in range(min(20, len(dataset["train"])))]

        yield {
            "evaluator": evaluator,
            "prefixes": prefixes,
            "separators": separators,
            "prepend_bos": prepend_bos,
            "normal_inputs": normal_inputs,
            "edge_inputs": edge_inputs,
            "complex_prompts": complex_prompts,
            "icl_data": icl_data,
            "word_pairs": word_pairs,
            "dataset": dataset,
            "prompts": prompts,
        }

    def _generate_prompt_data(self, data, input_data, n_icl_examples=0, complex_prompt_idx=None):
        """Generate prompt data for testing with varying ICL examples."""
        tokenizer = data["evaluator"].tokenizer
        model_config = data["evaluator"].model_config

        # Create word pairs with ICL examples if needed
        pd_word_pairs = copy.deepcopy(data["word_pairs"])
        if n_icl_examples > 0:
            set_seed(42)  # For reproducibility
            examples = list(np.random.choice(data["icl_data"], n_icl_examples))
            for ex in examples:
                pd_word_pairs["input"].append(ex["input"])
                pd_word_pairs["output"].append(ex["output"])

        # Use prompt from file if available, otherwise use input prompts
        prompt_baseline = input_data[1]
        if data["prompts"] and complex_prompt_idx is None:
            # Use prompt from file (cycling through available prompts)
            if isinstance(input_data, list) and len(input_data) > 1:
                # prompt_idx = (i + n_icl_examples) % len(data["prompts"])
                prompt_baseline = np.random.choice(data["prompts"])

        if complex_prompt_idx is not None and complex_prompt_idx < len(data["complex_prompts"]):
            prompt_baseline = data["complex_prompts"][complex_prompt_idx]
        else:
            prompt_baseline = input_data[1]

        # Create prompt data
        prompt_data = word_pairs_to_prompt_data(
            pd_word_pairs,
            query_target_pair=input_data[0],
            prepend_bos_token=data["prepend_bos"],
            shuffle_labels=False,
            instructions=prompt_baseline,
            prefixes=data["prefixes"],
            separators=data["separators"],
            tokenizer=tokenizer,
        )

        return prompt_data

    def _compare_label_outputs(self, original_labels, new_labels, prompt_string, tokenized_string_length):
        """Compare outputs from both label extension implementations."""
        # Check if original labels match tokenized length
        original_length_matches = len(original_labels) == tokenized_string_length

        # New implementation should always match tokenized length
        assert len(new_labels) == tokenized_string_length, (
            f"New implementation output length {len(new_labels)} doesn't match tokenized length {tokenized_string_length}"
        )

        result = {
            "original_length_matches": original_length_matches,
            "lengths_match": len(original_labels) == len(new_labels),
            "contents_match": original_labels == new_labels if original_length_matches else False,
            "tokenized_length": tokenized_string_length,
            "original_length": len(original_labels),
            "new_length": len(new_labels),
        }

        # If original method produced correct length, outputs should match
        if original_length_matches:
            assert original_labels == new_labels, (
                f"Label contents don't match even though lengths are the same!\n"
                f"Original: {original_labels}\n"
                f"New: {new_labels}"
            )

        return result

    def test_simple_cases(self, setup_label_test_data):
        """Test simple cases where both implementations should agree."""
        data = setup_label_test_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        model_config = evaluator.model_config

        for i, input_data in enumerate(data["normal_inputs"]):
            # Generate prompt data
            prompt_data = self._generate_prompt_data(data, input_data)

            # Get query
            query = prompt_data["query_target"]["input"]

            # Get prompt parts and labels
            prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)

            # Generate full prompt string
            prompt_string = create_prompt(prompt_data, sentence=query)

            # Tokenize prompt to get expected number of tokens
            tokenized = tokenizer(prompt_string)
            tokenized_string_length = len(tokenized.input_ids)

            # Run both implementations
            original_labels = extend_labels(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            new_labels = extend_labels_tokenize_combined(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            # Compare results
            result = self._compare_label_outputs(original_labels, new_labels, prompt_string, tokenized_string_length)
            # For real-world prompts, we need to ensure new implementation always works
            assert len(new_labels) == tokenized_string_length

            logger.info(f"Simple case #{i + 1}: Comparison result: {result}")

    def test_edge_cases(self, setup_label_test_data):
        """Test edge cases with potentially problematic characters."""
        data = setup_label_test_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        model_config = evaluator.model_config

        for i, input_data in enumerate(data["edge_inputs"]):
            # Generate prompt data
            prompt_data = self._generate_prompt_data(data, input_data)

            # Get query
            query = prompt_data["query_target"]["input"]

            # Get prompt parts and labels
            prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)

            # Generate full prompt string
            prompt_string = create_prompt(prompt_data, sentence=query)

            # Tokenize prompt to get expected number of tokens
            tokenized = tokenizer(prompt_string)
            tokenized_string_length = len(tokenized.input_ids)

            # Run both implementations
            original_labels = extend_labels(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            new_labels = extend_labels_tokenize_combined(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            # Compare results
            result = self._compare_label_outputs(original_labels, new_labels, prompt_string, tokenized_string_length)

            logger.info(f"Edge case #{i + 1}: Comparison result: {result}")

            # In edge cases, the new implementation should always match tokenized length
            # even if the original doesn't
            assert len(new_labels) == tokenized_string_length, (
                f"New implementation failed to match tokenized length in edge case #{i + 1}"
            )

    def test_with_complex_prompts(self, setup_label_test_data):
        """Test with complex prompts that may cause tokenization issues."""
        data = setup_label_test_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        model_config = evaluator.model_config

        for i, input_data in enumerate(data["normal_inputs"]):
            for j, complex_prompt in enumerate(data["complex_prompts"]):
                # Generate prompt data with complex prompt
                prompt_data = self._generate_prompt_data(data, input_data, complex_prompt_idx=j)

                # Get query
                query = prompt_data["query_target"]["input"]

                # Get prompt parts and labels
                prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)

                # Generate full prompt string
                prompt_string = create_prompt(prompt_data, sentence=query)

                # Tokenize prompt to get expected number of tokens
                tokenized = tokenizer(prompt_string)
                tokenized_string_length = len(tokenized.input_ids)

                # Run both implementations
                original_labels = extend_labels(
                    prompt_parts,
                    prompt_part_labels,
                    tokenizer,
                    tokenizer_prepends_bos=model_config["prepend_bos"],
                    tokenizer_kwargs=None,
                )

                new_labels = extend_labels_tokenize_combined(
                    prompt_parts,
                    prompt_part_labels,
                    tokenizer,
                    tokenizer_prepends_bos=model_config["prepend_bos"],
                    tokenizer_kwargs=None,
                )

                # Compare results
                result = self._compare_label_outputs(
                    original_labels, new_labels, prompt_string, tokenized_string_length
                )

                # In complex prompts, we expect the new implementation to always match token length
                assert len(new_labels) == tokenized_string_length, (
                    f"New implementation failed to match tokenized length with complex prompt #{j + 1}"
                )

                logger.info(f"Complex prompt #{j + 1} with input #{i + 1}: Comparison result: {result}")

                # Log details if original implementation failed
                if not result["original_length_matches"]:
                    logger.info(
                        f"Original implementation failed to match tokenized length: "
                        f"{result['original_length']} vs {tokenized_string_length}"
                    )

    @pytest.mark.parametrize("n_icl_examples", [0, 1, 3, 5])
    def test_with_varying_icl_examples(self, setup_label_test_data, n_icl_examples):
        """Test with varying numbers of ICL examples."""
        data = setup_label_test_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        model_config = evaluator.model_config

        for i, input_data in enumerate(data["normal_inputs"]):
            # Generate prompt data with ICL examples
            prompt_data = self._generate_prompt_data(data, input_data, n_icl_examples=n_icl_examples)

            # Get query
            query = prompt_data["query_target"]["input"]

            # Get prompt parts and labels
            prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)

            # Generate full prompt string
            prompt_string = create_prompt(prompt_data, sentence=query)

            # Tokenize prompt to get expected number of tokens
            tokenized = tokenizer(prompt_string)
            tokenized_string_length = len(tokenized.input_ids)

            # Run both implementations
            original_labels = extend_labels(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            new_labels = extend_labels_tokenize_combined(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            # Compare results
            result = self._compare_label_outputs(original_labels, new_labels, prompt_string, tokenized_string_length)

            logger.info(f"Input #{i + 1} with {n_icl_examples} ICL examples: Comparison result: {result}")

            # With ICL examples, we need to ensure predictive tokens are properly labeled
            pred_tokens_original = [i for i, label in enumerate(original_labels) if "pred" in label]
            pred_tokens_new = [i for i, label in enumerate(new_labels) if "pred" in label]

            # If original matches tokenized length, predictive tokens should be identical
            if result["original_length_matches"]:
                assert pred_tokens_original == pred_tokens_new, (
                    f"Predictive token positions don't match between implementations\n"
                    f"Original: {pred_tokens_original}\n"
                    f"New: {pred_tokens_new}"
                )

    def test_dummy_labels(self, setup_label_test_data):
        """Test that dummy labels work correctly with both implementations."""
        data = setup_label_test_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        model_config = evaluator.model_config

        for n_icl_examples in [0, 1, 3, 5]:
            # Build dummy labels
            dummy_labels = _build_dummy_labels_with_prompt(
                tokenizer, model_config, data["prefixes"], data["separators"], n_icl_examples=n_icl_examples
            )

            # Create a prompt data with dummy target
            prompt_data = self._generate_prompt_data(
                data, [{"input": "a", "output": "a"}, "Test prompt"], n_icl_examples=n_icl_examples
            )

            # Get query
            query = prompt_data["query_target"]["input"]

            # Get prompt parts and labels
            prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)

            # Generate full prompt string
            prompt_string = create_prompt(prompt_data, sentence=query)

            # Tokenize prompt to get expected number of tokens
            tokenized = tokenizer(prompt_string)
            tokenized_string_length = len(tokenized.input_ids)

            # Run both implementations
            original_labels = extend_labels(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            new_labels = extend_labels_tokenize_combined(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            # Compare results
            result = self._compare_label_outputs(original_labels, new_labels, prompt_string, tokenized_string_length)

            logger.info(f"Dummy labels with {n_icl_examples} ICL examples: Comparison result: {result}")

            # Check that the number of dummy labels is appropriate for the number of ICL examples
            token_labels, _ = get_token_meta_labels(
                prompt_data, tokenizer, query=query, prepend_bos=model_config["prepend_bos"]
            )

            final_token_labels = [(x[0], x[-1]) for x in token_labels]

            # The number of predictive tokens should match n_icl_examples + 1 (for query)
            pred_tokens = [i for i, (_, label) in enumerate(final_token_labels) if "pred" in label]
            assert len(pred_tokens) == n_icl_examples + 1, (
                f"Expected {n_icl_examples + 1} predictive tokens, got {len(pred_tokens)}"
            )

    def test_integration_with_tokenize_labels(self, setup_label_test_data):
        """Test integration with the tokenize_labels function."""
        data = setup_label_test_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        model_config = evaluator.model_config

        for i, input_data in enumerate(data["normal_inputs"] + data["edge_inputs"]):
            # Generate prompt data
            prompt_data = self._generate_prompt_data(data, input_data, n_icl_examples=2)

            # Get query
            query = prompt_data["query_target"]["input"]

            # Get prompt parts and labels
            prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)

            # Generate full prompt string
            prompt_string = create_prompt(prompt_data, sentence=query)

            # Tokenize prompt to get expected number of tokens
            tokenized = tokenizer(prompt_string)
            tokenized_string_length = len(tokenized.input_ids)

            # Test tokenize_labels with new implementation
            labels_with_new = tokenize_labels(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            # Directly use extend_labels_tokenize_combined
            direct_new_labels = extend_labels_tokenize_combined(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            # Both should match tokenized length
            assert len(labels_with_new) == tokenized_string_length, (
                f"tokenize_labels output length {len(labels_with_new)} doesn't match "
                f"tokenized length {tokenized_string_length}"
            )

            assert len(direct_new_labels) == tokenized_string_length, (
                f"direct extend_labels_tokenize_combined output length {len(direct_new_labels)} "
                f"doesn't match tokenized length {tokenized_string_length}"
            )

            # Both should produce identical results
            assert labels_with_new == direct_new_labels, (
                f"tokenize_labels and direct extend_labels_tokenize_combined produced different results:\n"
                f"tokenize_labels: {labels_with_new}\n"
                f"direct: {direct_new_labels}"
            )

            logger.info(f"Integration test case #{i + 1}: Labels match tokenized length {tokenized_string_length}")

    def test_with_dataset_prompts(self, setup_label_test_data):
        """Test with real-world prompts from the dataset."""
        data = setup_label_test_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        model_config = evaluator.model_config

        # Skip if no prompts available
        if not data["prompts"]:
            pytest.skip("No prompts available for testing")

        # Test with each prompt from the dataset
        for prompt_idx, prompt in enumerate(data["prompts"]):
            # Use first example from dataset
            input_data = [data["dataset"]["train"][0], prompt]

            # Generate prompt data
            prompt_data = self._generate_prompt_data(data, input_data, n_icl_examples=2)

            # Get query
            query = prompt_data["query_target"]["input"]

            # Get prompt parts and labels
            prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)

            # Generate full prompt string
            prompt_string = create_prompt(prompt_data, sentence=query)

            # Tokenize prompt to get expected number of tokens
            tokenized = tokenizer(prompt_string)
            tokenized_string_length = len(tokenized.input_ids)

            # Run both implementations
            original_labels = extend_labels(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            new_labels = extend_labels_tokenize_combined(
                prompt_parts,
                prompt_part_labels,
                tokenizer,
                tokenizer_prepends_bos=model_config["prepend_bos"],
                tokenizer_kwargs=None,
            )

            # Compare results
            result = self._compare_label_outputs(original_labels, new_labels, prompt_string, tokenized_string_length)
            logger.info(f"Dataset prompt #{prompt_idx + 1}: Comparison result: {result}")


if __name__ == "__main__":
    pytest.main(["-sxv", __file__])
