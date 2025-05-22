import argparse
import glob
import json
import os
import re
import typing
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import fireducks.pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from baukit import TraceDict
from loguru import logger
from tqdm import tqdm

from recipe.function_vectors.cache_short_texts import (
    DEFAULT_TEXTS_CACHE_PATH,
    MIN_BATCH_SIZE,
    WIKITEXT_NAME,
    input_ids_to_logprobs,
)
from recipe.function_vectors.generate_prompts_for_dataset import (
    LONG,
    SHORT,
)
from recipe.function_vectors.utils.eval_utils import (
    get_answer_id,
    tokenizer_padding_side_token,
)
from recipe.function_vectors.utils.extract_utils import (
    _build_dummy_labels_with_prompt,
    get_mean_head_activations,
)
from recipe.function_vectors.utils.intervention_utils import batched_replace_activation_w_avg, replace_activation_w_avg
from recipe.function_vectors.utils.model_utils import (
    load_gpt_model_and_tokenizer,
    set_seed,
)
from recipe.function_vectors.utils.prompt_utils import (
    compute_duplicated_labels,
    get_dummy_token_labels,
    get_token_meta_labels,
    load_dataset,
    should_skip_prompt,
    update_idx_map,
    word_pairs_to_prompt_data,
)
from recipe.function_vectors.utils.shared_utils import wandb_log


@dataclass
class TokenMetadata:
    token_labels: typing.List[typing.List[typing.Tuple[int, str, str]]] = field(default_factory=list)
    sentences: typing.List[str] = field(default_factory=list)
    targets: typing.List[str] = field(default_factory=list)
    idx_map: typing.List[typing.Dict[int, int]] = field(default_factory=list)
    idx_avg: typing.List[typing.Dict[int, typing.Tuple[int, int]]] = field(default_factory=list)
    token_id_of_interest: typing.List[typing.List[int]] = field(default_factory=list)

    def asdict(self):
        return asdict(self)

    def __len__(self):
        lengths = [len(v) for v in vars(self).values() if isinstance(v, list)]
        l = lengths[0]
        if not all(x == l for x in lengths):
            raise ValueError("TokenMetadata lists must have the same length")

        return l


def prompt_data_to_metadata(batch_prompt_data, tokenizer, model_config, dummy_labels) -> TokenMetadata:
    """Get token metadata for each prompt in batch."""
    batch_metadata = TokenMetadata()

    for prompt_data in batch_prompt_data:
        query_target_pair = prompt_data["query_target"]
        query = query_target_pair["input"]
        token_labels, prompt_string = get_token_meta_labels(
            prompt_data, tokenizer, query=query, prepend_bos=model_config["prepend_bos"]
        )
        batch_metadata.token_labels.append(token_labels)
        batch_metadata.sentences.append(prompt_string)

        idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
        idx_map = update_idx_map(idx_map, idx_avg)

        batch_metadata.idx_map.append(idx_map)
        batch_metadata.idx_avg.append(idx_avg)

        # Figure out tokens of interest
        target = [query_target_pair["output"]]
        batch_metadata.targets.append(target[0])
        token_id_of_interest = get_answer_id(prompt_string, target[0], tokenizer)
        if isinstance(token_id_of_interest, list):
            token_id_of_interest = token_id_of_interest[:1]

        batch_metadata.token_id_of_interest.append(token_id_of_interest)

    return batch_metadata


def _get_token_classes(last_token_only: bool):
    # Speed up computation by only computing causal effect at last token
    if last_token_only:
        token_classes = ["query_predictive"]
        token_classes_regex = ["query_predictive_token"]
    # Compute causal effect for all token classes (instead of just last token)
    else:
        token_classes = [
            "demonstration",
            "label",
            "separator",
            "predictive",
            "structural",
            "end_of_example",
            "query_demonstration",
            "query_structural",
            "query_separator",
            "query_predictive",
        ]
        token_classes_regex = [
            r"demonstration_[\d]{1,}_token",
            r"demonstration_[\d]{1,}_label_token",
            "separator_token",
            "predictive_token",
            "structural_token",
            "end_of_example_token",
            "query_demonstration_token",
            "query_structural_token",
            "query_separator_token",
            "query_predictive_token",
        ]

    return token_classes, token_classes_regex


DEBUG_MEMORY = False


def _debug_cuda_memory(prefix, device):
    if DEBUG_MEMORY:
        free, total = torch.cuda.mem_get_info(device)
        logger.debug(f"{prefix} | total: {total / (1024**3):.2f} GB | free: {free / (1024**3):.2f} GB")


def activation_replacement_per_class_intervention(
    prompt_data, avg_activations, dummy_labels, model, model_config, tokenizer, last_token_only=True, debug=False
):
    """
    Experiment to determine top intervention locations through avg activation replacement.
    Performs a systematic sweep over attention heads (layer, head) to track their causal influence on probs of key tokens.

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    avg_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    last_token_only: If True, only computes indirect effect for heads at the final token position. If False, computes indirect_effect for heads for all token classes

    Returns:
    indirect_effect_storage: torch tensor containing the indirect_effect of each head for each token class.
    """
    device = model.device
    metadata = prompt_data_to_metadata([prompt_data], tokenizer, model_config, dummy_labels)

    # # Get sentence and token labels
    # query_target_pair = prompt_data["query_target"]

    # query = query_target_pair["input"]
    # token_labels, prompt_string = get_token_meta_labels(
    #     prompt_data, tokenizer, query=query, prepend_bos=model_config["prepend_bos"]
    # )

    # idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
    # idx_map = update_idx_map(idx_map, idx_avg)

    # sentences = [prompt_string]  # * model.config.n_head # batch things by head

    # # Figure out tokens of interest
    # target = [query_target_pair["output"]]
    # token_id_of_interest = get_answer_id(sentences[0], target[0], tokenizer)
    # if isinstance(token_id_of_interest, list):
    #     token_id_of_interest = token_id_of_interest[:1]

    inputs = tokenizer(metadata.sentences, return_tensors="pt").to(device)
    token_classes, token_classes_regex = _get_token_classes(last_token_only)

    indirect_effect_storage = torch.zeros(model_config["n_layers"], model_config["n_heads"], len(token_classes))

    # Clean Run of Baseline:
    clean_output = model(**inputs).logits[:, -1, :]
    clean_probs = torch.softmax(clean_output[0], dim=-1)

    intervention_probs_storage = torch.zeros(
        model_config["n_layers"], model_config["n_heads"], len(token_classes), clean_probs.shape[-1]
    )

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    for layer in range(model_config["n_layers"]):
        head_hook_layer = [model_config["attn_hook_names"][layer]]

        for head_n in range(model_config["n_heads"]):
            for token_idx, (_, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
                reg_class_match = re.compile(f"^{class_regex}$")
                class_token_inds = [x[0] for x in metadata.token_labels[0] if reg_class_match.match(x[2])]

                intervention_locations = [(layer, head_n, token_n) for token_n in class_token_inds]

                intervention_fn = replace_activation_w_avg(
                    layer_head_token_pairs=intervention_locations,
                    avg_activations=avg_activations,
                    model=model,
                    model_config=model_config,
                    batched_input=False,
                    idx_map=metadata.idx_map[0],
                    last_token_only=last_token_only,
                )
                with torch.no_grad(), TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:
                    output = model(**inputs).logits[
                        :, -1, :
                    ]  # batch_size x n_tokens x vocab_size, only want last token prediction

                # print(f"{layer}/{head_n}/{token_idx} output sum: {output.sum()}")
                # TRACK probs of tokens of interest
                intervention_probs = torch.softmax(output, dim=-1).detach()
                intervention_probs_storage[layer, head_n, token_idx] = intervention_probs

                # convert to probability distribution
                indirect_effect_storage[layer, head_n, token_idx] = (
                    (intervention_probs - clean_probs)
                    .index_select(1, torch.LongTensor(metadata.token_id_of_interest[0]).to(device).squeeze())
                    .squeeze()
                    .cpu()
                )

        _debug_cuda_memory(f"After layer {layer}", device)

    if debug:
        # print("=" * 64)
        return indirect_effect_storage, intervention_probs_storage

    return indirect_effect_storage


@tokenizer_padding_side_token
def batch_activation_replacement_per_class_intervention(
    batch_prompt_data,
    avg_activations,
    dummy_labels,
    *,
    model,
    model_config,
    tokenizer,
    last_token_only=True,
    debug=False,
):
    """
    Experiment to determine top intervention locations through avg activation replacement.
    Performs a systematic sweep over attention heads (layer, head) to track their causal influence on probs of key tokens.

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    avg_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    last_token_only: If True, only computes indirect effect for heads at the final token position. If False, computes indirect_effect for heads for all token classes

    Returns:
    indirect_effect_storage: torch tensor containing the indirect_effect of each head for each token class.
    """
    device = model.device
    # print(batch_prompt_data)
    batch_metadata = prompt_data_to_metadata(batch_prompt_data, tokenizer, model_config, dummy_labels)
    # print(batch_metadata)
    inputs = tokenizer(batch_metadata.sentences, return_tensors="pt", padding=True).to(device)
    token_classes, token_classes_regex = _get_token_classes(last_token_only)

    indirect_effect_storage = torch.zeros(
        len(batch_metadata), model_config["n_layers"], model_config["n_heads"], len(token_classes)
    )

    pad_positions = torch.argmax((inputs.input_ids == tokenizer.pad_token_id).to(int), dim=1)
    output_positions = torch.tensor(
        [(inputs.input_ids.shape[1] if pos.item() == 0 else pos.item()) - 1 for pos in pad_positions], dtype=torch.long
    ).squeeze()

    # Clean Run of Baseline:
    batch_indices = torch.arange(len(batch_metadata))
    clean_output = model(**inputs).logits[batch_indices, output_positions, :]
    clean_probs = torch.softmax(clean_output, dim=-1)
    prints_by_example = {
        i: [f"Clean output sum: {clean_output[i].sum()} for example {i}"] for i in range(clean_output.shape[0])
    }

    intervention_probs_storage = torch.zeros(
        len(batch_metadata),
        model_config["n_layers"],
        model_config["n_heads"],
        len(token_classes),
        clean_probs.shape[-1],
    )

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    for layer in range(model_config["n_layers"]):
        head_hook_layer = [model_config["attn_hook_names"][layer]]

        for head_n in range(model_config["n_heads"]):
            for token_idx, (_, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
                reg_class_match = re.compile(f"^{class_regex}$")
                class_token_inds = [
                    [x[0] for x in token_labels if reg_class_match.match(x[2])]
                    for token_labels in batch_metadata.token_labels
                ]

                intervention_locations = [
                    [(layer, head_n, token_n) for token_n in class_token_inds] for class_token_inds in class_token_inds
                ]

                # if layer == 0 and head_n == 0:
                #     print(intervention_locations)

                intervention_fn = batched_replace_activation_w_avg(
                    batch_layer_head_token_pairs=intervention_locations,
                    output_positions=output_positions,
                    avg_activations=avg_activations,
                    model=model,
                    model_config=model_config,
                    batch_idx_map=batch_metadata.idx_map,
                    last_token_only=last_token_only,
                )
                with torch.no_grad(), TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:
                    # batch_size x n_tokens x vocab_size, only want last token prediction
                    output = model(**inputs).logits[batch_indices, output_positions, :]

                for i in range(output.shape[0]):
                    prints_by_example[i].append(
                        f"{layer}/{head_n}/{token_idx} output sum: {output[i].sum()} for example {i}"
                    )

                # TRACK probs of tokens of interest
                # convert to probability distribution
                intervention_probs = torch.softmax(output, dim=-1).detach()
                intervention_probs_storage[:, layer, head_n, token_idx] = intervention_probs

                token_id_of_interest_indices = torch.tensor(
                    batch_metadata.token_id_of_interest, dtype=torch.long
                ).squeeze()
                intervention_probs_of_interest = intervention_probs[batch_indices, token_id_of_interest_indices]
                clean_probs_of_interest = clean_probs[batch_indices, token_id_of_interest_indices]
                indirect_effect_storage[:, layer, head_n, token_idx] = (
                    (intervention_probs_of_interest - clean_probs_of_interest).squeeze().cpu()
                )

        _debug_cuda_memory(f"After layer {layer}", device)

    # for i, prints in prints_by_example.items():
    #     print("\n".join(prints))
    #     print()

    if debug:
        return indirect_effect_storage, intervention_probs_storage

    return indirect_effect_storage


def compute_indirect_effect(
    dataset,
    mean_activations,
    model,
    model_config,
    tokenizer,
    n_shots=10,
    n_trials=25,
    last_token_only=True,
    prefixes=None,
    separators=None,
    filter_set=None,
    partial_path=None,
):
    """
    Computes Indirect Effect of each head in the model

    Parameters:
    dataset: ICL dataset
    mean_activations:
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: Number of shots in each in-context prompt
    n_trials: Number of in-context prompts to average over
    last_token_only: If True, only computes Indirect Effect for heads at the final token position. If False, computes Indirect Effect for heads for all token classes


    Returns:
    indirect_effect: torch tensor of the indirect effect for each attention head in the model, size n_trials * n_layers * n_heads
    """
    n_test_examples = 1

    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(
            n_shots,
            tokenizer=tokenizer,
            prefixes=prefixes,
            separators=separators,
            model_config=model_config,
        )
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, model_config=model_config)

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config["prepend_bos"] else True
    start_index = 0

    if partial_path is not None and os.path.exists(partial_path):
        indirect_effect = torch.load(partial_path)
        start_index = (indirect_effect == 0).all(axis=2).all(axis=1).nonzero().min().item()
    elif last_token_only:
        indirect_effect = torch.zeros(n_trials, model_config["n_layers"], model_config["n_heads"])
    else:
        indirect_effect = torch.zeros(
            n_trials, model_config["n_layers"], model_config["n_heads"], 10
        )  # have 10 classes of tokens

    if filter_set is None:
        filter_set = np.arange(len(dataset["valid"]))

    for i in tqdm(range(start_index, n_trials), total=n_trials):
        word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset["valid"][np.random.choice(filter_set, n_test_examples, replace=False)]
        if prefixes is not None and separators is not None:
            prompt_data_random = word_pairs_to_prompt_data(
                word_pairs,
                query_target_pair=word_pairs_test,
                shuffle_labels=True,
                prepend_bos_token=prepend_bos,
                prefixes=prefixes,
                separators=separators,
                tokenizer=tokenizer,
            )
        else:
            prompt_data_random = word_pairs_to_prompt_data(
                word_pairs,
                query_target_pair=word_pairs_test,
                shuffle_labels=True,
                prepend_bos_token=prepend_bos,
                tokenizer=tokenizer,
            )

        ind_effects = activation_replacement_per_class_intervention(
            prompt_data=prompt_data_random,
            avg_activations=mean_activations,
            dummy_labels=dummy_gt_labels,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            last_token_only=last_token_only,
        )
        indirect_effect[i] = ind_effects.squeeze()
        if partial_path is not None:
            torch.save(indirect_effect, partial_path)

    return indirect_effect


def _to_input_ids(sample_ids, model):
    return torch.tensor(sample_ids, dtype=torch.int64, device=model.device).unsqueeze(0)


class LogprobProportionalTextSampler:
    partial_dfs: typing.Dict[int, pd.DataFrame]
    prompt_cache: typing.Dict[str, typing.Tuple[int, np.ndarray]]

    def __init__(self, partial_dfs: typing.Dict[int, pd.DataFrame], rng):
        self.partial_dfs = partial_dfs
        self.rng = rng
        self.prompt_cache = {}

    def __call__(
        self,
        model,
        tokenizer,
        prompt: str,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        if prompt not in self.prompt_cache:
            tokenized = tokenizer(prompt, return_tensors="pt").to(model.device)
            n_tok = tokenized.input_ids.shape[1]
            if n_tok not in self.partial_dfs:
                dfs = []
                if n_tok - 1 in self.partial_dfs:
                    dfs.append(self.partial_dfs[n_tok - 1])
                if n_tok + 1 in self.partial_dfs:
                    dfs.append(self.partial_dfs[n_tok + 1])

                if dfs:
                    logger.warning(f"Found no matches with {n_tok} tokens, using {n_tok} +/- 1 instead.")
                else:
                    raise ValueError(
                        f"LogprobProportionalTextSampler encountered an unexpected number of tokens in a prompt: '{prompt}' => {n_tok}"
                    )

                n_tok = f"{n_tok}+/-1"
                self.partial_dfs[n_tok] = pd.concat(dfs)

            logits = model(**tokenized.to(model.device)).logits
            logprobs = F.log_softmax(logits.float(), dim=-1)
            lp = logprobs[
                0,
                torch.arange(logits.shape[1] - 1),
                tokenized.input_ids[0, 1:],
            ].sum()
            # sample proportionally to a softmax over minus the absolute logprob deviations
            probs = (
                F.softmax(
                    -torch.abs(torch.tensor(self.partial_dfs[n_tok].logprob.values).to(model.device) - lp)
                    / temperature,
                    dim=-1,
                )
                .cpu()
                .numpy()
            )
            self.prompt_cache[prompt] = (n_tok, probs)

        n_tok, probs = self.prompt_cache[prompt]
        idx = self.rng.choice(len(probs), p=probs)
        return self.partial_dfs[n_tok].text.iloc[idx]


class ClosestLogprobTextSampler:
    partial_dfs: typing.Dict[int, pd.DataFrame]
    prompt_cache: typing.Dict[str, typing.Tuple[int, np.ndarray]]

    def __init__(self, partial_dfs: typing.Dict[int, pd.DataFrame], rng):
        self.partial_dfs = partial_dfs
        self.combined_dfs = {}
        self.rng = rng
        self.prompt_cache = {}

    def __call__(
        self,
        model,
        tokenizer,
        prompt: str,
        min_options: int = 100,
        *args,
        **kwargs,
    ):
        if prompt not in self.prompt_cache:
            tokenized = tokenizer(prompt, return_tensors="pt").to(model.device)
            n_tok = tokenized.input_ids.shape[1]

            if n_tok not in self.combined_dfs:
                n_options = 0
                dfs = []

                if n_tok in self.partial_dfs:
                    df = self.partial_dfs[n_tok]
                    n_options += len(df)
                    dfs.append(df)

                increment = 1
                while n_options <= min_options:
                    if n_tok - increment in self.partial_dfs:
                        df = self.partial_dfs[n_tok - increment]
                        n_options += len(df)
                        dfs.append(df)
                    if n_tok + increment in self.partial_dfs:
                        df = self.partial_dfs[n_tok + increment]
                        n_options += len(df)
                        dfs.append(df)

                    increment += 1

                self.combined_dfs[n_tok] = pd.concat(dfs)

            combined_df = self.combined_dfs[n_tok]

            logits = model(**tokenized.to(model.device)).logits
            logprobs = F.log_softmax(logits.float(), dim=-1)
            lp = logprobs[
                0,
                torch.arange(logits.shape[1] - 1),
                tokenized.input_ids[0, 1:],
            ].sum()

            logprob_abs_diffs = torch.abs(torch.tensor(combined_df.logprob.values).to(model.device) - lp)
            sort_indices = torch.argsort(logprob_abs_diffs).tolist()
            self.prompt_cache[prompt] = (n_tok, sort_indices)

        n_tok, sort_indices = self.prompt_cache[prompt]
        idx = sort_indices.pop(0)
        return self.combined_dfs[n_tok].text.iloc[idx]


class PromptBaseline(StrEnum):
    EQUIPROBABLE_STRING = "equiprobable"
    REAL_TEXT_EQUIPROBABLE = "real_text"
    EMPTY_STRING = "empty_string"
    OTHER_TASK_PROMPT = "other_task_prompt"
    HALF_PROMPT_RESAMPLE = "half_prompt_resample"

    def build_baseline_generator(baseline, model, tokenizer, **kwargs):
        match baseline:
            case PromptBaseline.EMPTY_STRING:

                def bos_baseline(*args, **kwargs):
                    return ""

                return bos_baseline

            case PromptBaseline.EQUIPROBABLE_STRING:

                def sample_matching_logits(
                    model,
                    tokenizer,
                    prompt: str,
                    margin: float = 0.1,
                    margin_inc: float = 0.1,
                    should_prepend_bos: bool = False,
                    *args,
                    **kwargs,
                ):
                    if should_prepend_bos:
                        prompt = tokenizer.bos_token + prompt

                    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
                    logits = model(input_ids=input_ids).logits
                    logprobs = F.log_softmax(logits.float(), dim=-1)
                    sample_logits = None

                    # Mask out added vocab tokens (bos, eos, etc.)
                    n_tokens = logprobs.shape[-1]
                    added_vocab_mask = torch.ones((n_tokens,), dtype=torch.bool, device=logprobs.device)
                    for added_index in tokenizer.get_added_vocab().values():
                        if added_index < n_tokens:
                            added_vocab_mask[added_index] = False

                    sample_ids = [input_ids[0, 0].item()]
                    sample_logits = None
                    sample_logprobs = None
                    for i in range(logits.shape[1] - 1):
                        current_logprob = logprobs[:, i, input_ids[:, i]].squeeze()
                        logprobs_from = logprobs if sample_logprobs is None else sample_logprobs
                        matches = None
                        step_margin = margin
                        while matches is None or matches.numel() == 0:
                            matches = torch.argwhere(
                                ((logprobs_from[0, i, :] - current_logprob).abs() < step_margin) & added_vocab_mask
                            ).squeeze()
                            step_margin += margin_inc

                        new_id = matches if matches.numel() == 1 else matches[torch.randint(matches.shape[0], (1,))]
                        sample_ids.append(new_id.item())
                        sample_logits = model(input_ids=_to_input_ids(sample_ids, model)).logits
                        sample_logprobs = F.log_softmax(sample_logits.float(), dim=-1)

                    # original_logprobs = torch.gather(torch.log_softmax(logits.float(), -1), 2, input_ids.unsqueeze(0))
                    # sample_logprobs = torch.gather(torch.log_softmax(sample_logits.float(), -1), 2, _to_input_ids(sample_ids).unsqueeze(0))

                    # print(
                    #     logprobs[:, torch.arange(logprobs.shape[1]), input_ids].sum(),
                    #     sample_logprobs[:, torch.arange(logprobs.shape[1]), sample_ids].sum()
                    # )

                    # Omit the first token if it's the BOS (which it should be -- the check is a sanity)
                    if sample_ids[0] == tokenizer.bos_token_id:
                        sample_ids = sample_ids[1:]

                    return tokenizer.decode(sample_ids)

                return sample_matching_logits

            case PromptBaseline.OTHER_TASK_PROMPT:
                rng = kwargs.get("rng", None)
                if rng is None:
                    raise ValueError(f"{PromptBaseline.OTHER_TASK_PROMPT} requires specifying 'rng'")

                saved_prompts_root = kwargs.get("saved_prompts_root", None)
                if saved_prompts_root is None:
                    raise ValueError(f"{PromptBaseline.OTHER_TASK_PROMPT} requires specifying 'saved_prompts_root'")

                saved_prompts_file = kwargs.get("saved_prompts_file")
                if saved_prompts_file is None:
                    raise ValueError(f"{PromptBaseline.OTHER_TASK_PROMPT} requires specifying 'saved_prompts_file'")

                saved_prompts_suffix = kwargs.get("saved_prompts_suffix")
                if saved_prompts_suffix is None:
                    raise ValueError(f"{PromptBaseline.OTHER_TASK_PROMPT} requires specifying 'saved_prompts_suffix'")

                prompt_type = kwargs.get("prompt_type", None)
                if "rng" == None:
                    raise ValueError(f"{PromptBaseline.REAL_TEXT_EQUIPROBABLE} requires specifying 'prompt_type'")

                @tokenizer_padding_side_token
                def compute_prompts_by_n_tokens(*, tokenizer):
                    prompts_by_n_tokens = defaultdict(list)
                    n_other_task_prompt_files_parsed = 0

                    # TODO: consider a more clever filter that ignores opposing/related tasks
                    for prompts_file in glob.glob(f"{saved_prompts_root}/*{saved_prompts_suffix}.json"):
                        if os.path.basename(prompts_file) == saved_prompts_file or (
                            prompt_type == SHORT and LONG in prompts_file
                        ):
                            continue

                        else:
                            with open(prompts_file, "r") as f:
                                task_prompts = json.load(f)["prompts"]
                                tokenized_task_prompts = tokenizer(
                                    task_prompts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                ).input_ids
                                tokens_per_partial = (
                                    (tokenized_task_prompts != tokenizer.pad_token_id).sum(dim=-1).tolist()
                                )
                                for task_prompt, n_tokens in zip(task_prompts, tokens_per_partial):
                                    prompts_by_n_tokens[n_tokens].append(task_prompt)

                                n_other_task_prompt_files_parsed += 1

                    logger.info(
                        f"Found a total of {sum(len(tp) for tp in prompts_by_n_tokens.values())} other task prompts across {n_other_task_prompt_files_parsed} other tasks with {len(prompts_by_n_tokens)} token lengths between {min(prompts_by_n_tokens.keys())} and {max(prompts_by_n_tokens.keys())}"
                    )

                    # cache log probs for each token
                    batch_size = kwargs.get("batch_size", MIN_BATCH_SIZE)
                    for n_tokens in list(prompts_by_n_tokens.keys()):
                        n_token_prompts = prompts_by_n_tokens[n_tokens]
                        n_token_logprobs = []
                        n_batches = np.ceil(len(n_token_prompts) / batch_size).astype(int)
                        for b in range(n_batches):
                            tokenized_prompts = tokenizer(
                                n_token_prompts[b * batch_size : min((b + 1) * batch_size, len(n_token_prompts))],
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                            )
                            n_token_logprobs.extend(input_ids_to_logprobs(model, tokenized_prompts.input_ids).tolist())

                        prompts_by_n_tokens[n_tokens] = pd.DataFrame(dict(text=n_token_prompts, logprob=n_token_logprobs))

                    return prompts_by_n_tokens

                prompt_by_tokens_dict = compute_prompts_by_n_tokens(tokenizer=tokenizer)
                return ClosestLogprobTextSampler(prompt_by_tokens_dict, rng)


            case PromptBaseline.REAL_TEXT_EQUIPROBABLE:
                prompt_type = kwargs.get("prompt_type", None)
                if "rng" == None:
                    raise ValueError(f"{PromptBaseline.REAL_TEXT_EQUIPROBABLE} requires specifying 'prompt_type'")

                rng = kwargs.get("rng", None)
                if "rng" == None:
                    raise ValueError(f"{PromptBaseline.REAL_TEXT_EQUIPROBABLE} requires specifying 'rng'")

                model_name = kwargs.get("model_name", None)
                if model_name is None:
                    raise ValueError(f"{PromptBaseline.REAL_TEXT_EQUIPROBABLE} requires specifying 'model_name'")

                cache_path = kwargs.get("cache_path", DEFAULT_TEXTS_CACHE_PATH)
                hf_dataset_name = kwargs.get("hf_dataset_name", WIKITEXT_NAME)
                save_path = f"{cache_path}/{model_name[model_name.rfind('/') + 1 :]}_{hf_dataset_name}.csv.gz".lower()

                if prompt_type != SHORT:
                    save_path = save_path.replace(".csv.gz", "_long.csv.gz")

                if not os.path.exists(save_path):
                    raise ValueError(
                        f"{PromptBaseline.REAL_TEXT_EQUIPROBABLE} expected the following path to exist, but it doeesn't: {save_path}"
                    )

                df = pd.read_csv(save_path).rename(columns=dict(sentence="text"))
                partial_dfs = {n_tokens: df[df.n_tokens == n_tokens].reindex() for n_tokens in df.n_tokens.unique()}

                return ClosestLogprobTextSampler(partial_dfs, rng)

            case _:
                raise NotImplementedError(f"PromptBaseline.build_baseline_generator not implemented yet for {baseline}")


def compute_prompt_based_indirect_effect(
    dataset,
    prompts,
    mean_activations,
    baseline,
    model,
    model_config,
    tokenizer,
    n_trials_per_prompt=5,
    last_token_only=True,
    prefixes=None,
    separators=None,
    filter_set=None,
    partial_path=None,
    n_icl_examples=0,
    shuffle_icl_labels=False,
    query_dataset="train",
    baseline_generator_kwargs=None,
    batch_size: int = 1,
    forced: bool = False,
    # rng: typing.Optional[np.random.Generator] = None,
):
    """
    Computes Indirect Effect of each head in the model

    Parameters:
    dataset: ICL dataset
    mean_activations:
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: Number of shots in each in-context prompt
    n_trials: Number of in-context prompts to average over
    last_token_only: If True, only computes Indirect Effect for heads at the final token position. If False, computes Indirect Effect for heads for all token classes


    Returns:
    indirect_effect: torch tensor of the indirect effect for each attention head in the model, size n_trials * n_layers * n_heads
    """
    if batch_size > n_trials_per_prompt:
        logger.warning(
            f"Batch size {batch_size} is greater than n_trials_per_prompt {n_trials_per_prompt}, setting batch size to {n_trials_per_prompt}"
        )
        batch_size = n_trials_per_prompt

    if baseline_generator_kwargs is None:
        baseline_generator_kwargs = {}

    n_test_examples = 1

    dummy_labels = _build_dummy_labels_with_prompt(
        tokenizer,
        model_config,
        prefixes,
        separators,
        n_icl_examples=n_icl_examples,
    )

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config["prepend_bos"] else True
    start_index = 0

    baselines_by_prompt = {}
    baselines_by_prompt_path = None
    indirect_effect = None

    if partial_path is not None:
        baselines_by_prompt_path = partial_path.replace(".pt.partial", "_baseline_prompts.json")

        if forced:
            logger.info(
                f"Deleting existing partial file {partial_path} and baselines_by_prompt file {baselines_by_prompt_path} as forced indirect effect"
            )

            if os.path.exists(partial_path):
                os.remove(partial_path)
            if os.path.exists(baselines_by_prompt_path):
                os.remove(baselines_by_prompt_path)

        else:
            if os.path.exists(partial_path):
                indirect_effect = torch.load(partial_path)

                missing = (indirect_effect == 0).all(axis=3).all(axis=2).nonzero()
                if missing.numel() == 0:
                    return indirect_effect

                prompt_index, example_index = missing[0]
                prompt_index = prompt_index.item()
                example_index = example_index.item()
                start_index = (prompt_index * n_trials_per_prompt) + example_index
                logger.info(
                    f"Resuming `compute_prompt_based_indirect_effect` from p={prompt_index}, e={example_index} => s={start_index}"
                )

            if os.path.exists(baselines_by_prompt_path):
                with open(baselines_by_prompt_path, "r") as f:
                    baselines_by_prompt = json.load(f)

    if indirect_effect is None:
        if last_token_only:
            indirect_effect = torch.zeros(
                len(prompts),
                n_trials_per_prompt,
                model_config["n_layers"],
                model_config["n_heads"],
            )
        else:
            indirect_effect = torch.zeros(
                len(prompts),
                n_trials_per_prompt,
                model_config["n_layers"],
                model_config["n_heads"],
                10,  # have 10 classes of tokens
            )

    if filter_set is None:
        logger.warning(
            f"`compute_prompt_based_indirect_effect` called with no filter set, sampling from all {query_dataset} examples"
        )
        filter_set = np.arange(len(dataset[query_dataset]))

    prompt_baseline_generator = PromptBaseline.build_baseline_generator(
        baseline, model, tokenizer, **baseline_generator_kwargs
    )
    pbar = tqdm(desc="Indirect effect", total=len(prompts) * n_trials_per_prompt)

    for p, prompt in enumerate(prompts):
        fs = None
        if isinstance(filter_set, dict):
            if prompt not in filter_set:
                error_message = f"`compute_prompt_based_indirect_effect` called with dictionary filter set, but prompt '{prompt}' not in it"
                logger.error(error_message)
                raise ValueError(error_message)

            fs = filter_set[prompt]
        else:
            fs = filter_set

        batch_prompt_data = []

        for trial_idx in range(n_trials_per_prompt):
            pbar.update(1)
            pbar.set_postfix({"Prompt": p + 1, "Trial": trial_idx + 1})

            overall_idx = p * n_trials_per_prompt + trial_idx
            if overall_idx < start_index:
                continue

            prompt_baseline = prompt_baseline_generator(model, tokenizer, prompt, **baseline_generator_kwargs)
            # logger.debug(f"Prompt {p}, trial {trial_idx}, prompt_baseline {prompt_baseline}")
            if prompt not in baselines_by_prompt:
                baselines_by_prompt[prompt] = []

            skip = True
            while skip:
                if n_icl_examples == 0:
                    if shuffle_icl_labels:
                        raise ValueError(
                            "Cannot providee shuffle_icl_labels = True and n_icl_examples = 0 (meaningless)"
                        )
                    word_pairs = {"input": [], "output": []}
                    # if rng is None:
                    #     query_idx = np.random.choice(fs, n_test_examples, replace=False)
                    # else:
                    #     query_idx = rng.choice(fs, n_test_examples, replace=False)
                    query_idx = np.random.choice(fs, n_test_examples, replace=False)
                    # logger.debug(f"Prompt {p}, trial {trial_idx}, query_idx {query_idx}")
                    word_pairs_query = dataset[query_dataset][query_idx]

                else:
                    if query_dataset == "train":
                        raise ValueError("Query dataset cannot be train when providing n_icl_examples != 0")

                    word_pairs = dataset["train"][
                        np.random.choice(len(dataset["train"]), n_icl_examples, replace=False)
                    ]
                    word_pairs_query = dataset["valid"][np.random.choice(fs, n_test_examples, replace=False)]

                wpq = word_pairs_query
                if isinstance(wpq, list):
                    wpq = wpq[0]

                target = wpq["output"]
                if isinstance(target, list):
                    target = target[0]

                skip = should_skip_prompt(target, prompt_baseline)

            baselines_by_prompt[prompt].append((prompt_baseline, word_pairs_query))

            if prefixes is not None and separators is not None:
                prompt_data_baseline = word_pairs_to_prompt_data(
                    word_pairs,
                    query_target_pair=word_pairs_query,
                    prepend_bos_token=prepend_bos,
                    shuffle_labels=shuffle_icl_labels,
                    instructions=prompt_baseline,
                    prefixes=prefixes,
                    separators=separators,
                    tokenizer=tokenizer,
                )
            else:
                prompt_data_baseline = word_pairs_to_prompt_data(
                    word_pairs,
                    query_target_pair=word_pairs_query,
                    prepend_bos_token=prepend_bos,
                    shuffle_labels=shuffle_icl_labels,
                    instructions=prompt_baseline,
                    tokenizer=tokenizer,
                )

            if batch_size > 1:
                batch_prompt_data.append(prompt_data_baseline)
                bs = len(batch_prompt_data)
                if (bs >= batch_size) or (trial_idx == n_trials_per_prompt - 1):
                    batch_ind_effects = batch_activation_replacement_per_class_intervention(
                        batch_prompt_data,
                        avg_activations=mean_activations,
                        dummy_labels=dummy_labels,
                        model=model,
                        model_config=model_config,
                        tokenizer=tokenizer,
                        last_token_only=last_token_only,
                    )
                    batch_start_idx = trial_idx - bs + 1
                    batch_end_idx = trial_idx + 1

                    for prompt_idx in range(len(batch_prompt_data)):
                        if (batch_ind_effects[prompt_idx] == 0).all():
                            weird_pb, weird_wpq = baselines_by_prompt[prompt][batch_start_idx + prompt_idx]
                            logger.warning(
                                f"Indirect effect is all zeros for prompt '{prompt}', baseline '{weird_pb}', query '{weird_wpq}'"
                            )

                        if batch_ind_effects[prompt_idx].isnan().any():
                            nans = batch_ind_effects[prompt_idx].isnan().nonzero()
                            weird_pb, weird_wpq = baselines_by_prompt[prompt][batch_start_idx + prompt_idx]
                            logger.warning(
                                f"Indirect effect has NaNs for prompt '{prompt}', baseline '{weird_pb}', query '{weird_wpq}' at {nans}"
                            )

                    indirect_effect[p, batch_start_idx:batch_end_idx] = batch_ind_effects.squeeze()
                    if partial_path is not None:
                        torch.save(indirect_effect, partial_path)

                    batch_prompt_data = []

            else:
                ind_effects = activation_replacement_per_class_intervention(
                    prompt_data=prompt_data_baseline,
                    avg_activations=mean_activations,
                    dummy_labels=dummy_labels,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    last_token_only=last_token_only,
                )
                indirect_effect[p, trial_idx] = ind_effects.squeeze()
                if partial_path is not None:
                    torch.save(indirect_effect, partial_path)

            if baselines_by_prompt_path is not None:
                with open(baselines_by_prompt_path, "w") as f:
                    json.dump(baselines_by_prompt, f)

        wandb_log(
            "indirect_effect",
            indirect_effect_prompt_index=p,
            indirect_effect_index=(p * n_trials_per_prompt) + trial_idx,
        )

    return indirect_effect


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to be loaded",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name",
        help="Name of model to be loaded",
        type=str,
        required=False,
        default="EleutherAI/gpt-j-6b",
    )
    parser.add_argument(
        "--root_data_dir",
        help="Root directory of data files",
        type=str,
        required=False,
        default="../dataset_files",
    )
    parser.add_argument(
        "--save_path_root",
        help="File path to save indirect effect to",
        type=str,
        required=False,
        default="../results",
    )
    parser.add_argument("--seed", help="Randomized seed", type=int, required=False, default=42)
    parser.add_argument(
        "--n_shots",
        help="Number of shots in each in-context prompt",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--n_trials",
        help="Number of in-context prompts to average over",
        type=int,
        required=False,
        default=25,
    )
    parser.add_argument(
        "--test_split",
        help="Percentage corresponding to test set split size",
        required=False,
        default=0.3,
    )
    parser.add_argument(
        "--device",
        help="Device to run on",
        type=str,
        required=False,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mean_activations_path",
        help="Path to mean activations file used for intervention",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--last_token_only",
        help="Whether to compute indirect effect for heads at only the final token position, or for all token classes",
        required=False,
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--prefixes",
        help="Prompt template prefixes to be used",
        type=json.loads,
        required=False,
        default={"input": "Q:", "output": "A:", "instructions": ""},
    )
    parser.add_argument(
        "--separators",
        help="Prompt template separators to be used",
        type=json.loads,
        required=False,
        default={"input": "\n", "output": "\n\n", "instructions": ""},
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    seed = args.seed
    n_shots = args.n_shots
    n_trials = args.n_trials
    test_split = args.test_split
    device = args.device
    mean_activations_path = args.mean_activations_path
    last_token_only = args.last_token_only
    prefixes = args.prefixes
    separators = args.separators

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    set_seed(seed)

    # Load the dataset
    print("Loading Dataset")
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # Load or Re-Compute Mean Activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f"{save_path_root}/{dataset_name}_mean_head_activations.pt"):
        mean_activations_path = f"{save_path_root}/{dataset_name}_mean_head_activations.pt"
        mean_activations = torch.load(mean_activations_path)
    else:
        print("Computing Mean Activations")
        mean_activations = get_mean_head_activations(
            dataset,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            n_icl_examples=n_shots,
            N_TRIALS=n_trials,
            prefixes=prefixes,
            separators=separators,
        )
        torch.save(
            mean_activations,
            f"{save_path_root}/{dataset_name}_mean_head_activations.pt",
        )

    print("Computing Indirect Effect")
    indirect_effect = compute_indirect_effect(
        dataset,
        mean_activations,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        n_shots=n_shots,
        n_trials=n_trials,
        last_token_only=last_token_only,
        prefixes=prefixes,
        separators=separators,
    )

    # Write args to file
    args.save_path_root = save_path_root
    args.mean_activations_path = mean_activations_path
    with open(f"{save_path_root}/indirect_effect_args.txt", "w") as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    torch.save(indirect_effect, f"{save_path_root}/{dataset_name}_indirect_effect.pt")
