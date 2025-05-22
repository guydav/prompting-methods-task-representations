import re

import numpy as np
import pytest
import torch
from baukit import TraceDict
from loguru import logger

from recipe.function_vectors.compute_indirect_effect import (
    _get_token_classes,
    prompt_data_to_metadata,
)
from recipe.function_vectors.utils.extract_utils import _build_dummy_labels_with_prompt
from recipe.function_vectors.utils.intervention_utils import (
    batched_replace_activation_w_avg,
    replace_activation_w_avg,
)
from recipe.function_vectors.utils.prompt_utils import load_dataset
from tests.test_sentence_eval import SentenceEvalEvaluator
from tests.test_utils import DATASETS_PATH, prepare_data_for_batch, set_seed, summarize_differences


class TestBatchedActivationReplacement:
    """Test class for batched_replace_activation_w_avg functionality."""

    @pytest.fixture
    def setup_replacement_data(self, model_params):
        """Setup data for activation replacement tests.

        Takes model_params from pytest fixture defined in conftest.py
        """
        # Use params from command line if provided
        model_name = model_params.get("model_name", "meta-llama/Llama-3.2-1B")
        dataset_name = model_params.get("dataset_name", "country-capital")
        # atol = model_params.get("atol", 1e-4)
        atol = 2e-3
        rtol = model_params.get("rtol", 1e-5)
        n_examples = model_params.get("n_examples", 20)
        start_index = model_params.get("start_index", 0)
        batch_sizes = model_params.get("batch_sizes", [1, 2, 3, 5, 7])
        # prompt_baseline = model_params.get("prompt", "C'est ne pas une prompt.")
        prompt_baseline = ["C'est ne pas une prompt.", "C'est ne pas une much, much, much longer prompt."]
        last_token_mode = model_params.get("last_token_mode", "true")
        mean_activation_scale = model_params.get("mean_activation_scale", 1e-1)

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

        last_token_values = []
        if last_token_mode == "both":
            last_token_values = [True, False]
        elif last_token_mode == "true":
            last_token_values = [True]
        elif last_token_mode == "false":
            last_token_values = [False]
        else:
            raise ValueError(f"Invalid last_token_mode: {last_token_mode}")

        set_seed(42)
        random_mean_activations = (
            torch.randn(
                (
                    evaluator.model_config["n_layers"],
                    evaluator.model_config["n_heads"],
                    10,
                    # evaluator.model_config["resid_dim"] // evaluator.model_config["n_heads"],
                    evaluator.model_config["head_dim"],
                )
            )
            * mean_activation_scale
        )

        # Setup intervention parameters
        n_layers_to_test = min(4, evaluator.model_config["n_layers"])
        test_layers = list(np.random.choice(evaluator.model_config["n_layers"], n_layers_to_test, replace=False))
        test_layers.sort()

        n_heads_to_test = min(8, evaluator.model_config["n_heads"])
        test_heads = {
            layer: sorted(list(np.random.choice(evaluator.model_config["n_heads"], n_heads_to_test, replace=False)))
            for layer in test_layers
        }

        yield {
            "evaluator": evaluator,
            "dataset": dataset,
            "dummy_labels": dummy_labels,
            "prepend_bos": prepend_bos,
            "prefixes": prefixes,
            "separators": separators,
            "prompt_baseline": prompt_baseline,
            # "fake_activation": fake_activation,
            "random_mean_activations": random_mean_activations,
            "test_layers": test_layers,
            "test_heads": test_heads,
            "batch_sizes": batch_sizes,
            "last_token_values": last_token_values,
        }

    def _run_intervention(self, data, metadata, intervention_fn, layer, head, token_class):
        """Run intervention with the specified function."""
        set_seed(42)
        model = data["evaluator"].model
        tokenizer = data["evaluator"].tokenizer
        device = model.device
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
# tokenizer sets the pad token id automatically when the pad token is set
        inputs = tokenizer(metadata["sentences"], return_tensors="pt", padding=True).to(device)

        # Get clean outputs
        clean_output = model(**inputs).logits[:, -1, :]
        clean_probs = torch.softmax(clean_output, dim=-1)

        # TODO: fix the output positions here too

        # Run intervention
        head_hook_layer = [data["evaluator"].model_config["attn_hook_names"][layer]]
        with torch.no_grad(), TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:
            output = model(**inputs).logits[:, -1, :]

        # Get intervention probabilities
        intervention_probs = torch.softmax(output, dim=-1)

        # Calculate effects
        batch_indices = torch.arange(len(metadata["sentences"]), dtype=torch.long).squeeze()
        token_id_of_interest_indices = torch.tensor(
            [ti[0] for ti in metadata["token_id_of_interest"]], dtype=torch.long
        ).squeeze()
        intervention_probs_of_interest = intervention_probs[batch_indices, token_id_of_interest_indices]
        clean_probs_of_interest = clean_probs[batch_indices, token_id_of_interest_indices]
        effects = (intervention_probs_of_interest - clean_probs_of_interest).cpu()

        # batch_size = len(metadata["sentences"])
        # effects = torch.zeros(batch_size)
        # for i in range(batch_size):
        #     token_id = metadata["token_id_of_interest"][i]
        #     if isinstance(token_id, list):
        #         token_id = token_id[0]
        #     effects[i] = (intervention_probs - clean_probs)[i, token_id]

        return clean_probs, intervention_probs, effects

    def _split_metadata(self, metadata, start=0, batch_size=1):
        return {k: v[start : start + batch_size] for k, v in metadata.items()}

    def test_batched_activation_replacement(self, setup_replacement_data):
        """Test that batched activation replacement produces identical results to unbatched."""
        data = setup_replacement_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # batch_sizes = data["batch_sizes"]
        batch_sizes = [1, 1, 2, 3, 5, 7, 10]
        max_batch_size = max(batch_sizes)
        max_batch_prompt_data, sentences, targets = prepare_data_for_batch(
            data, max_batch_size, return_texts=True, tokenizer=tokenizer
        )
        max_batch_metadata = prompt_data_to_metadata(
            max_batch_prompt_data, evaluator.tokenizer, evaluator.model_config, data["dummy_labels"]
        ).asdict()
        last_token_values = data["last_token_values"]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        pad_positions = torch.argmax((inputs.input_ids == tokenizer.pad_token_id).to(int), dim=1)
        output_positions = torch.tensor(
            [(inputs.input_ids.shape[1] if pos.item() == 0 else pos.item()) - 1 for pos in pad_positions],
            dtype=torch.long,
        ).squeeze()

        for last_token_only in last_token_values:
            logger.info(f"Testing with last_token_only={last_token_only}")
            token_classes, token_classes_regex = _get_token_classes(last_token_only)

            # Run for a subset of layers and heads
            for layer in data["test_layers"]:
                for head_n in data["test_heads"][layer]:
                    for i, (token_class, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
                        logger.info(f"Testing layer {layer}, head {head_n}, token class {token_class} (#{i})")
                        baseline_effects = torch.zeros(max_batch_size, dtype=torch.float32)
                        batched_effects_by_batch_size = {
                            b: torch.zeros(max_batch_size, dtype=torch.float32) for b in batch_sizes
                        }

                        # Token class to target
                        reg_class_match = re.compile(f"^{class_regex}$")

                        # Get baseline (unbatched) results

                        for i in range(max_batch_size):
                            ex_metadata = self._split_metadata(max_batch_metadata, i, 1)
                            # Prepare unbatched intervention
                            class_token_inds = [
                                x[0] for x in ex_metadata["token_labels"][0] if reg_class_match.match(x[2])
                            ]
                            intervention_locations = [(layer, head_n, token_n) for token_n in class_token_inds]

                            unbatched_fn = replace_activation_w_avg(
                                layer_head_token_pairs=intervention_locations,
                                # avg_activations=data["fake_activation"],
                                avg_activations=data["random_mean_activations"],
                                model=evaluator.model,
                                model_config=evaluator.model_config,
                                batched_input=False,
                                idx_map=ex_metadata["idx_map"][0],
                                last_token_only=last_token_only,
                            )

                            _, _, baseline_effect = self._run_intervention(
                                data, ex_metadata, unbatched_fn, layer, head_n, token_class
                            )
                            if baseline_effect.dim() > 0:
                                baseline_effect = baseline_effect[0]
                            baseline_effects[i] = baseline_effect

                        # Test with various batch sizes
                        for current_batch_size in batch_sizes:
                            if current_batch_size > len(max_batch_metadata["sentences"]):
                                logger.warning(
                                    f"Skipping batch size {current_batch_size} as it exceeds dataset size {max_batch_metadata['sentences']}"
                                )
                                continue

                            logger.info(f"Testing with batch size {current_batch_size}")

                            for b in range(0, max_batch_size, current_batch_size):
                                batch_metadata = self._split_metadata(max_batch_metadata, b, current_batch_size)

                                # Prepare batched intervention locations
                                batch_class_token_inds = [
                                    [x[0] for x in token_labels if reg_class_match.match(x[2])]
                                    for token_labels in batch_metadata["token_labels"]
                                ]

                                batch_intervention_locations = [
                                    [(layer, head_n, token_n) for token_n in class_token_inds]
                                    for class_token_inds in batch_class_token_inds
                                ]

                                batched_fn = batched_replace_activation_w_avg(
                                    batch_layer_head_token_pairs=batch_intervention_locations,
                                    # avg_activations=data["fake_activation"],
                                    output_positions=output_positions[b : b + current_batch_size],
                                    avg_activations=data["random_mean_activations"],
                                    model=evaluator.model,
                                    model_config=evaluator.model_config,
                                    batch_idx_map=batch_metadata["idx_map"],
                                    last_token_only=last_token_only,
                                )

                                _, _, batched_effect = self._run_intervention(
                                    data, batch_metadata, batched_fn, layer, head_n, token_class
                                )

                                batched_effects_by_batch_size[current_batch_size][b : b + current_batch_size] = (
                                    batched_effect
                                )

                            assert (batched_effects_by_batch_size[current_batch_size] == 0).sum() == 0, (
                                f"Batch size {current_batch_size} has zero effects"
                            )

                            # Compare first item from batched to unbatched
                            summary_str = summarize_differences(
                                baseline_effects, batched_effects_by_batch_size[current_batch_size]
                            )
                            assert torch.allclose(
                                batched_effects_by_batch_size[current_batch_size],
                                baseline_effects,
                                atol=evaluator.atol,
                                rtol=evaluator.rtol,
                            ), (
                                f"Effects differ for batch size {current_batch_size}, layer {layer}, head {head_n}:\n{summary_str}"
                            )

                            logger.debug(
                                f"Batch size {current_batch_size}, layer {layer}, head {head_n}: {summary_str}"
                            )

    def test_invalid_inputs(self, setup_replacement_data):
        """Test that invalid inputs raise appropriate errors."""
        data = setup_replacement_data
        evaluator = data["evaluator"]
        tokenizer = evaluator.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare test data
        batch_size = 2
        batch_prompt_data, sentences, targets = prepare_data_for_batch(
            data, batch_size, return_texts=True, tokenizer=tokenizer
        )
        metadata = prompt_data_to_metadata(
            batch_prompt_data, evaluator.tokenizer, evaluator.model_config, data["dummy_labels"]
        ).asdict()

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        pad_positions = torch.argmax((inputs.input_ids == tokenizer.pad_token_id).to(int), dim=1)
        output_positions = torch.tensor(
            [(inputs.input_ids.shape[1] if pos.item() == 0 else pos.item()) - 1 for pos in pad_positions],
            dtype=torch.long,
        ).squeeze()

        # Test case: mismatched number of intervention locations and idx maps
        layer, head_n = 0, 0
        token_class = "query_predictive"
        class_regex = "query_predictive_token"
        reg_class_match = re.compile(f"^{class_regex}$")

        batch_class_token_inds = [
            [x[0] for x in token_labels if reg_class_match.match(x[2])] for token_labels in metadata["token_labels"]
        ]

        batch_intervention_locations = [
            [(layer, head_n, token_n) for token_n in class_token_inds] for class_token_inds in batch_class_token_inds
        ]

        # Add an extra intervention location to create mismatch
        mismatched_locations = batch_intervention_locations + [batch_intervention_locations[0]]

        # This should raise a ValueError when run
        with pytest.raises(
            ValueError, match="Number of intervention specifications .* must match number of index mappings"
        ):
            batched_fn = batched_replace_activation_w_avg(
                batch_layer_head_token_pairs=mismatched_locations,
                # avg_activations=data["fake_activation"],
                output_positions=output_positions,
                avg_activations=data["random_mean_activations"],
                model=evaluator.model,
                model_config=evaluator.model_config,
                batch_idx_map=metadata["idx_map"],
                last_token_only=True,
            )

            # Run intervention to trigger validation inside the function
            inputs = evaluator.tokenizer(metadata["sentences"], return_tensors="pt").to(evaluator.model.device)
            head_hook_layer = [evaluator.model_config["attn_hook_names"][layer]]
            with TraceDict(evaluator.model, layers=head_hook_layer, edit_output=batched_fn):
                evaluator.model(**inputs)


if __name__ == "__main__":
    pytest.main(["-vsx", __file__])
