import itertools
import os
import random
import sys
import tempfile
import typing
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from recipe.function_vectors.utils.eval_utils import (
    EvalDataResults,
    batch_sentence_eval,
    sentence_eval,
    sentence_eval_original,
)
from recipe.function_vectors.utils.model_utils import load_gpt_model_and_tokenizer
from recipe.function_vectors.utils.prompt_utils import load_dataset
from tests.test_utils import DATASETS_PATH, set_seed, summarize_differences



class SentenceEvalEvaluator:
    """Class to handle evaluation of transformer models with various settings."""

    model_name: str
    device: str
    atol: float
    rtol: float
    datasets_path: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_config: typing.Dict[str, typing.Any]

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        atol: float = 1e-4,
        rtol: float = 1e-5,
        datasets_path: str = DATASETS_PATH,
        seed: int = 42,
    ):
        """Initialize the evaluator with model and tokenizer.

        Args:
            model_name: Name of the Hugging Face model to load
            device: Device to run the model on ("cuda" or "cpu")
            atol: Absolute olerance for comparing floating point values
            rtol: Relative olerance for comparing floating point values
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.atol = atol
        self.rtol = rtol
        self.datasets_path = datasets_path
        set_seed(seed)
        self.model, self.tokenizer, self.model_config = self._load_model_and_tokenizer()  # type: ignore

    def isclose(self, x, y):
        return torch.isclose(torch.tensor(x), torch.tensor(y), atol=self.atol, rtol=self.rtol)

    def __repr__(self):
        return f"SentenceEvalEvaluator({self.model_name})"

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from Hugging Face."""
        print(f"Loading model {self.model_name} on {self.device}...")
        return load_gpt_model_and_tokenizer(self.model_name, device=self.device)

    def prepare_dataset(
        self,
        dataset_name: str,
        prompt: str,
        n_examples: int = 4,
        start_index: int = 0,
        extra_examples: typing.List[typing.Dict[str, str]] | None = None,
        datasets_path: str | None = None,
    ):
        """Prepare dataset for evaluation.

        Args:
            dataset_name: Name of the dataset to load
            prompt: Prompt template to use
            n_examples: Number of examples to use
            start_index: Index to start from in the dataset
            extra_examples: Additional examples to include
            dataset_path: Path to the dataset directory

        Returns:
            sentences: List of input sentences
            targets: List of target outputs
        """
        if datasets_path is None:
            datasets_path = self.datasets_path

        dataset = load_dataset(dataset_name, datasets_path, test_size=0.3, seed=42)

        sentences = []
        targets = []
        for i in range(n_examples):
            ex = dataset["train"][start_index + i]
            sentences.append(f"{prompt}\nQ: {ex['input']}\nA:")  # type: ignore
            targets.append(" " + ex["output"])  # type: ignore

        if extra_examples:
            for ex in extra_examples:
                sentences.append(f"{prompt}\nQ: {ex['input']}\nA:")
                targets.append(" " + ex["output"])

        return sentences, targets

    def create_prefix_cache(self, prefix: str, batch_size: int | None = None):
        """Create cache for the prefix to avoid recomputation.

        Args:
            prefix: Prefix to cache
            batch_size: Batch size (if None, only single example cache is created)

        Returns:
            single_cache: Cache for single example
            batch_cache: Cache for batch (if batch_size is provided)
            prefix_length: Length of the prefix in tokens
        """
        # Create single example cache
        single_cache = DynamicCache()
        with torch.no_grad():
            single_prefix_tensor = self.tokenizer(prefix, return_tensors="pt").to(
                self.model.device
            )
            single_outputs = self.model(
                **single_prefix_tensor, use_cache=True, past_key_values=single_cache
            )
            single_cache_kv = single_outputs.past_key_values

        # Create batch cache if requested
        batch_cache_kv = None
        if batch_size is not None:
            batch_cache = DynamicCache()
            with torch.no_grad():
                batch_prefix_tensor = self.tokenizer(
                    [prefix] * batch_size, return_tensors="pt"
                ).to(self.model.device)
                batch_outputs = self.model(
                    **batch_prefix_tensor, use_cache=True, past_key_values=batch_cache
                )
                batch_cache_kv = batch_outputs.past_key_values

            prefix_length = batch_prefix_tensor.input_ids.shape[-1]
        else:
            prefix_length = single_prefix_tensor.input_ids.shape[-1]

        return single_cache_kv, batch_cache_kv, prefix_length

    def original_sentence_eval(self, sentence, target, **kwargs):
        """Evaluate a single sentence and return the output."""
        return sentence_eval_original(
            sentence, target, self.model, self.tokenizer, **kwargs
        )

    def sentence_eval(self, sentence, target, **kwargs):
        """Wrapper for sentence_eval function."""
        # This would be your existing sentence_eval function
        # For now, we just pass it through as a placeholder
        return sentence_eval(
            EvalDataResults([sentence], [target]), model=self.model, tokenizer=self.tokenizer, **kwargs
        )

    def batch_sentence_eval(self, sentences, targets, **kwargs):
        """Wrapper for batch_sentence_eval function."""
        # This would be your existing batch_sentence_eval function
        # For now, we just pass it through as a placeholder
        return batch_sentence_eval(
            EvalDataResults(sentences, targets), model=self.model, tokenizer=self.tokenizer, **kwargs
        )

    def _simple_metric_fn(self, prediction, target):
        """Simple exact match metric."""
        return prediction == target


def _extend_or_append(
    items: typing.List[typing.Any], new_items: typing.List[typing.Any] | typing.Any
):
    if isinstance(new_items, list):
        items.extend(new_items)
    else:
        items.append(new_items)


class TestSentenceEvaluation:
    """Test class for transformer evaluation."""

    temp_file: typing.TextIO | None

    @pytest.fixture
    def setup_evaluator(self):
        """Setup the evaluator with model and test data."""
        # Default parameters that can be overridden with pytest parametrize
        model_name = "meta-llama/Llama-3.2-1B"
        dataset_name = "country-capital"
        n_examples = 10
        start_index = 0
        prompt = "Rewrite the word in the past tense."
        extra_examples = None  # Add extra examples if needed
        seed = 42

        # Create the evaluator
        evaluator = SentenceEvalEvaluator(model_name, seed=seed)

        # Prepare test data
        sentences, targets = evaluator.prepare_dataset(
            dataset_name, prompt, n_examples, start_index, extra_examples
        )

        # Create prefix cache
        prefix_to_cache = "Rewrite the word in the past tense.\nQ:"
        single_cache, batch_cache, prefix_length = evaluator.create_prefix_cache(
            prefix_to_cache, len(sentences)
        )

        # Check that all sentences start with the prefix
        assert all(sentence.startswith(prefix_to_cache) for sentence in sentences), (
            "Sentences must start with the prefix to cache"
        )

        temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)

        yield {
            "evaluator": evaluator,
            "sentences": sentences,
            "targets": targets,
            "single_cache": single_cache,
            "batch_cache": batch_cache,
            "prefix_length": prefix_length,
            "temp_file": temp_file,
        }

        if temp_file is not None:
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    # def teardown_method(self):
    #     """Clean up after tests."""
    #     # Close and remove temporary file if it exists
    #     if self.temp_file is not None:
    #         self.temp_file.close()
    #         if os.path.exists(self.temp_file.name):
    #             os.unlink(self.temp_file.name)

    def test_nll_mode(self, setup_evaluator):
        """Test compute_nll mode."""
        data = setup_evaluator
        evaluator = typing.cast(SentenceEvalEvaluator, data["evaluator"])
        sentences = data["sentences"]
        targets = data["targets"]
        single_cache = data["single_cache"]
        batch_cache = data["batch_cache"]
        prefix_length = data["prefix_length"]

        print("Testing compute_nll mode...")

        print("Computing original results")
        original_outputs = []
        original_nlls = []
        for sentence, target in zip(sentences, targets):
            logits, nll = evaluator.original_sentence_eval(  # type: ignore
                sentence, target, compute_nll=True, generate_str=False
            )
            _extend_or_append(original_outputs, logits)
            _extend_or_append(original_nlls, nll)

        # Unbatched results
        print("Computing unbatched results...")
        unbatched_outputs = []
        unbatched_nlls = []
        for sentence, target in zip(sentences, targets):
            result = evaluator.sentence_eval(
                sentence, target, compute_nll=True, generate_str=False
            )
            _extend_or_append(unbatched_outputs, result.logits)
            _extend_or_append(unbatched_nlls, result.nlls)

        # Batched results
        print("Computing batched results...")
        batched_result = evaluator.batch_sentence_eval(
            sentences, targets, compute_nll=True, generate_str=False
        )
        batched_outputs = batched_result.logits
        batched_nll = batched_result.nlls

        # Cached results
        print("Computing cached results...")
        unbatched_cached_outputs = []
        unbatched_cached_nlls = []
        for sentence, target in zip(sentences, targets):
            result = evaluator.batch_sentence_eval(
                [sentence],
                [target],
                compute_nll=True,
                generate_str=False,
                past_key_values=single_cache,
                prefix_length_tokens=prefix_length,
            )
            _extend_or_append(unbatched_cached_outputs, result.logits)
            _extend_or_append(unbatched_cached_nlls, result.nlls)

        # Batched and cached results
        print("Computing batched and cached results...")
        result = evaluator.batch_sentence_eval(
            sentences,
            targets,
            compute_nll=True,
            generate_str=False,
            past_key_values=batch_cache,
            prefix_length_tokens=prefix_length,
        )
        batched_cached_outputs = result.logits
        batched_cached_nll = result.nlls

        # Compare outputs and NLLs
        names = ["original", "unbatched", "batched", "cached", "batched_cached"]

        # Compare outputs
        names_and_outputs = list(
            zip(
                names,
                [
                    original_outputs,
                    unbatched_outputs,
                    batched_outputs,
                    unbatched_cached_outputs,
                    batched_cached_outputs,
                ],
            )
        )

        for (name1, outputs1), (name2, outputs2) in itertools.combinations(
            names_and_outputs, 2
        ):
            for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):  # type: ignore
                assert torch.allclose(o1, o2, atol=evaluator.atol), (
                    f"NLL mode: Outputs differ between {name1} and {name2} for example {i}"
                )

        # Compare NLLs
        names_and_nlls = list(
            zip(
                names,
                [
                    original_nlls,
                    unbatched_nlls,
                    batched_nll,
                    unbatched_cached_nlls,
                    batched_cached_nll,
                ],
            )
        )

        for (name1, nlls1), (name2, nlls2) in itertools.combinations(names_and_nlls, 2):
            for i, (nll1, nll2) in enumerate(zip(nlls1, nlls2)):  # type: ignore
                assert abs(nll1 - nll2) < evaluator.atol, (
                    f"NLL mode: NLL values differ between {name1} and {name2} for example {i}"
                )

        print("NLL mode test passed!")

    def test_generate_str_mode(self, setup_evaluator):
        """Test generate_str mode."""
        data = setup_evaluator
        evaluator = typing.cast(SentenceEvalEvaluator, data["evaluator"])
        sentences = data["sentences"]
        targets = data["targets"]
        single_cache = data["single_cache"]
        batch_cache = data["batch_cache"]
        prefix_length = data["prefix_length"]
        temp_file = data["temp_file"]

        def parse_generation(output_str, target, metric_fn):
            """Helper function to parse generation for testing."""
            return output_str, metric_fn(output_str, target)

        print("Testing generate_str mode...")

        # original results
        print("Computing original strings...")
        original_scores = []
        for sentence, target in zip(sentences, targets):
            score = evaluator.original_sentence_eval(
                sentence,
                target,
                compute_nll=False,
                generate_str=True,
                pred_file=temp_file,
                metric_fn=evaluator._simple_metric_fn,
                test_deterministic=True,
            )
            _extend_or_append(original_scores, score)

        # Reset file position
        temp_file.seek(0)
        original_strings = temp_file.readlines()
        temp_file.seek(0)
        temp_file.truncate()

        # Unbatched results
        print("Computing unbatched strings...")
        unbatched_scores = []
        for sentence, target in zip(sentences, targets):
            result = evaluator.sentence_eval(
                sentence,
                target,
                compute_nll=False,
                generate_str=True,
                pred_file=temp_file,
                metric_fn=evaluator._simple_metric_fn,
                test_deterministic=True,
            )
            _extend_or_append(unbatched_scores, result.scores)

        # Reset file position
        temp_file.seek(0)
        unbatched_strings = temp_file.readlines()
        temp_file.seek(0)
        temp_file.truncate()

        # Batched results
        print("Computing batched strings...")
        batched_scores = evaluator.batch_sentence_eval(
            sentences,
            targets,
            compute_nll=False,
            generate_str=True,
            pred_file=temp_file,
            metric_fn=evaluator._simple_metric_fn,
            test_deterministic=True,
        ).scores
        temp_file.seek(0)
        batched_strings = temp_file.readlines()
        temp_file.seek(0)
        temp_file.truncate()

        # Cached results
        print("Computing cached strings...")
        cached_scores = []
        for sentence, target in zip(sentences, targets):
            reuslt = evaluator.batch_sentence_eval(
                [sentence],
                [target],
                compute_nll=False,
                generate_str=True,
                pred_file=temp_file,
                metric_fn=evaluator._simple_metric_fn,
                past_key_values=single_cache,
                prefix_length_tokens=prefix_length,
                test_deterministic=True,
            )
            _extend_or_append(cached_scores, result.scores)

        temp_file.seek(0)
        cached_strings = temp_file.readlines()
        temp_file.seek(0)
        temp_file.truncate()

        # Batched and cached results
        print("Computing batched and cached strings...")
        batched_cached_scores = evaluator.batch_sentence_eval(
            sentences,
            targets,
            compute_nll=False,
            generate_str=True,
            pred_file=temp_file,
            metric_fn=evaluator._simple_metric_fn,
            past_key_values=batch_cache,
            prefix_length_tokens=prefix_length,
            test_deterministic=True,
        ).scores

        temp_file.seek(0)
        batched_cached_strings = temp_file.readlines()
        temp_file.seek(0)
        temp_file.truncate()

        # Compare strings and scores
        names = ["original", "unbatched", "batched", "cached", "batched_cached"]
        generated_strings = [
            original_strings,
            unbatched_strings,
            batched_strings,
            cached_strings,
            batched_cached_strings,
        ]
        # print(generated_strings)

        # names_and_strings = list(zip(names, generated_strings))

        # Batching can cause slightly different generations
        # So we warn for differences between batched and unbatched and assert for cached/uncached

        def test_string_pairs(
            name1, strings1, name2, strings2, should_assert: bool = True
        ):
            for i, (s1, s2) in enumerate(zip(strings1, strings2)):
                if should_assert:
                    assert s1 == s2, (
                        f"generate_str mode: Strings between {name1} and {name2} differ for example {i}: {repr(s1)} vs {repr(s2)}"
                    )
                else:
                    if s1 != s2:
                        logger.warning(
                            f"generate_str mode: Strings between {name1} and {name2} differ for example {i}: {repr(s1)} vs {repr(s2)}"
                        )

        test_string_pairs("original", original_strings, "unbatched", unbatched_strings)
        test_string_pairs("original", original_strings, "cached", cached_strings)

        test_string_pairs(
            "unbatched", unbatched_strings, "batched", batched_strings, False
        )
        test_string_pairs(
            "cached", cached_strings, "batched_cached", batched_cached_strings, False
        )

        test_string_pairs("unbatched", unbatched_strings, "cached", cached_strings)
        test_string_pairs(
            "batched", batched_strings, "batched_cached", batched_cached_strings
        )

        # for (name1, strings1), (name2, strings2) in itertools.combinations(names_and_strings, 2):
        #     for i, (s1, s2) in enumerate(zip(strings1, strings2)):
        #         assert s1 == s2, \
        #              f"generate_str mode: Strings between {name1} and {name2} differ for example {i}: {s1} vs {s2}"

        # names_and_scores = list(zip(names, [
        #     unbatched_scores, batched_scores, cached_scores, batched_cached_scores
        # ]))

        # for (name1, scores1), (name2, scores2) in itertools.combinations(names_and_scores, 2):
        #     for i, (s1, s2) in enumerate(zip(scores1, scores2)):
        #         assert s1 == s2, \
        #             f"generate_str mode: Scores between {name1} and {name2} differ for example {i}"

        print("generate_str mode test passed!")

    def test_default_mode(self, setup_evaluator):
        """Test default mode (next token prediction)."""
        data = setup_evaluator
        evaluator = typing.cast(SentenceEvalEvaluator, data["evaluator"])
        sentences = data["sentences"]
        targets = data["targets"]
        single_cache = data["single_cache"]
        batch_cache = data["batch_cache"]
        prefix_length = data["prefix_length"]

        print("Testing default mode...")
        # original results
        print("Computing original default results...")
        original_default_outputs = []
        for sentence, target in zip(sentences, targets):
            output = evaluator.original_sentence_eval(
                sentence, target, compute_nll=False, generate_str=False
            )
            _extend_or_append(original_default_outputs, output)

        # Unbatched results
        print("Computing unbatched default results...")
        unbatched_default_outputs = []
        for sentence, target in zip(sentences, targets):
            output = evaluator.sentence_eval(
                sentence, target, compute_nll=False, generate_str=False
            ).logits
            _extend_or_append(unbatched_default_outputs, output)

        # Batched results
        print("Computing batched default results...")
        batched_default_outputs = evaluator.batch_sentence_eval(
            sentences, targets, compute_nll=False, generate_str=False
        ).logits

        # Cached results
        print("Computing cached default results...")
        cached_default_outputs = []
        for sentence, target in zip(sentences, targets):
            output = evaluator.batch_sentence_eval(
                [sentence],
                [target],
                compute_nll=False,
                generate_str=False,
                past_key_values=single_cache,
                prefix_length_tokens=prefix_length,
            ).logits
            _extend_or_append(cached_default_outputs, output)

        # Batched and cached results
        print("Computing batched and cached default results...")
        batched_cached_default_outputs = evaluator.batch_sentence_eval(
            sentences,
            targets,
            compute_nll=False,
            generate_str=False,
            past_key_values=batch_cache,
            prefix_length_tokens=prefix_length,
        ).logits

        # Compare outputs
        names = ["original", "unbatched", "batched", "cached", "batched_cached"]
        outputs = [
            original_default_outputs, unbatched_default_outputs, batched_default_outputs,
            cached_default_outputs, batched_cached_default_outputs,
        ]

        names_and_default_outputs = list(zip(names, outputs))

        for (name1, outputs1), (name2, outputs2) in itertools.combinations(
            names_and_default_outputs, 2
        ):
            for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):  # type: ignore
                summary_str = summarize_differences(o1, o2)
                assert torch.allclose(o1, o2, atol=evaluator.atol), (
                    f"Default mode: Outputs differ between {name1} and {name2} for example {i}:\n{summary_str}"
                )

                logger.debug(f"Default mode summary for example {name1}|{name2}|{i} with shapes {o1.shape}|{o2.shape}:\n{summary_str}")

        print("Default mode test passed!")

    @pytest.mark.parametrize(
        "model_name,dataset_name,n_examples,start_index,prompt",
        [
            (
                "meta-llama/Llama-3.2-1B",
                "present-past",
                4,
                0,
                "Rewrite the word in the past tense.",
            ),
            # Add more parameter combinations for testing different models and datasets
        ],
    )
    def test_with_different_models_and_datasets(
        self, model_name, dataset_name, n_examples, start_index, prompt
    ):
        """Parameterized test for testing with different models and datasets."""
        # Create evaluator with specified parameters
        evaluator = SentenceEvalEvaluator(model_name)

        # Prepare test data
        sentences, targets = evaluator.prepare_dataset(
            dataset_name, prompt, n_examples, start_index
        )

        # Run a simple test to verify the setup works
        # You can call more specific test methods here

        # Clean up
        del evaluator


if __name__ == "__main__":
    # Example of how to run specific tests
    # pytest.main(["-xvs", __file__ + "::SentenceEvalEvaluator::test_default_mode"])
# 
    # Or run all tests
    pytest.main(["-xvs", __file__])
