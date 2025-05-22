import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers.trainer_utils import enable_full_determinism

from recipe.function_vectors.utils.prompt_utils import (
    create_prompt,
    filter_prompts_by_max_tokens,
    word_pairs_to_prompt_data,
)


def _get_project_root():
    p = Path(__file__).absolute()
    while not list(p.glob(".git")):
        p = p.parent
        if str(p) == "/":
            raise ValueError("Project root not found in parents")
    return p


PROJECT_ROOT = str(_get_project_root())
DATASETS_PATH = os.path.join(PROJECT_ROOT, "tests", "test_data", "dataset_files")
PROMPTS_PATH = os.path.join(PROJECT_ROOT, "tests", "test_data", "prompts")


def load_prompts_from_file(prompts_file, max_prompts=None, max_tokens=None, tokenizer=None):
    """Load prompts from a JSON file.

    Args:
        prompts_file: Path to the JSON file containing prompts
        max_prompts: Maximum number of prompts to use
        max_tokens: Maximum token length for prompts
        tokenizer: Tokenizer to use for token length calculation

    Returns:
        List of prompts
    """
    with open(prompts_file, "r") as f:
        prompts_data = json.load(f)

    prompts = prompts_data.get("prompts", [])

    if max_tokens and tokenizer:
        keep_indices = filter_prompts_by_max_tokens(prompts, tokenizer=tokenizer, max_length_tokens=max_tokens)
        prompts = [prompts[i] for i in keep_indices]

    if max_prompts and len(prompts) > max_prompts:
        prompts = prompts[:max_prompts]

    return prompts


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # type: ignore
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    enable_full_determinism(seed)


def prepare_data_for_batch(data, batch_size=None, return_texts=False, tokenizer=None):
    """Prepare data for a specific batch size."""
    if batch_size is None:
        batch_size = len(data["dataset"]["train"])

    batch_prompt_data = []
    sentences = []
    targets = []

    prompt_or_prompts = data["prompt_baseline"]

    for i in range(batch_size):
        word_pairs_query = data["dataset"]["train"][i]

        prompt = (
            prompt_or_prompts[i % len(prompt_or_prompts)] if isinstance(prompt_or_prompts, list) else prompt_or_prompts
        )

        prompt_data = word_pairs_to_prompt_data(
            {"input": [], "output": []},
            query_target_pair=word_pairs_query,
            prepend_bos_token=data["prepend_bos"],
            shuffle_labels=False,
            instructions=prompt,
            prefixes=data["prefixes"],
            separators=data["separators"],
            tokenizer=tokenizer,
        )
        batch_prompt_data.append(prompt_data)

        # Extract sentence and target for intervention
        sentence = create_prompt(prompt_data)
        target = prompt_data["query_target"]["output"]
        sentences.append(sentence)
        targets.append(target)

    if return_texts:
        return batch_prompt_data, sentences, targets

    return batch_prompt_data


def compare_rank_lists(ranks1, ranks2, equality_threshold_rank=20, max_rank_diff=2):
    return all(
        r1 == r2 if min(r1, r2) <= equality_threshold_rank else abs(r1 - r2) <= max_rank_diff
        for r1, r2 in zip(ranks1, ranks2)
    )


def summarize_differences(first_tensor, second_tensor, softmax=False):
    first_tensor = first_tensor.squeeze()
    second_tensor = second_tensor.squeeze()

    if softmax:
        first_tensor = torch.softmax(first_tensor, dim=-1)
        second_tensor = torch.softmax(second_tensor, dim=-1)
    either_nonzero = (first_tensor != 0) | (second_tensor != 0)
    diffs_where_nonzero = torch.abs(first_tensor - second_tensor)[either_nonzero].ravel()
    max_diff = torch.max(diffs_where_nonzero).item()

    return f"max abs diff={max_diff:.4e}, mean nonzero diff={torch.mean(diffs_where_nonzero):.4e}, nonzero diff std={torch.std(diffs_where_nonzero):.4e}, {len(diffs_where_nonzero)} nonzero elements"
