import argparse
import os
import re
import typing
from collections import defaultdict
from pathlib import Path

import datasets
import fireducks.pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from recipe.function_vectors.utils.model_utils import load_gpt_model_and_tokenizer

STORAGE_ROOT = os.environ.get("STORAGE_ROOT")
DEFAULT_TEXTS_CACHE_PATH = f"{STORAGE_ROOT}/function_vectors/short_real_text_caches"
DEFAULT_RANDOM_SEED = 33
MIN_LENGTH_TOKENS = 4
MAX_LENGTH_TOKENS = 64
MAX_SPACE_INDEX = 32
N_TOTAL_SAMPLES = 2**16
MIN_BATCH_SIZE = 8
WIKITEXT_PATH = "Salesforce/wikitext"
WIKITEXT_NAME = "wikitext-103-v1"
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def input_ids_to_logprobs(model, input_ids):
    logits = model(input_ids.to(model.device)).logits.cpu()
    logprobs = F.log_softmax(logits.float(), dim=-1)
    batch_size = logprobs.shape[0]
    n_tokens = logprobs.shape[1] - 1  # ignore the bos token
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_tokens)
    token_idx = torch.arange(n_tokens).repeat(batch_size).reshape(batch_size, -1)
    return logprobs[batch_idx, token_idx, input_ids[:, 1:]].sum(-1)


def main(args: typing.Optional[typing.List[str]] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        help="Name of model to be loaded",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_path_root",
        help="File path to save to",
        type=str,
        required=False,
        default=DEFAULT_TEXTS_CACHE_PATH,
    )
    parser.add_argument(
        "--save_path_suffix",
        help="Subdirectory to save results into within the results directory",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--random_seed",
        help="Randomized seed",
        type=int,
        required=False,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument(
        "--min_batch_size",
        help="Minimal batch size to collate and pass through the model",
        type=int,
        default=MIN_BATCH_SIZE,
    )
    parser.add_argument(
        "--min_length_tokens",
        help="Minimal length (in # of tokens) to track examples for",
        type=int,
        default=MIN_LENGTH_TOKENS,
    )
    parser.add_argument(
        "--max_length_tokens",
        help="Maximal length (in # of tokens) to track examples for",
        type=int,
        default=MAX_LENGTH_TOKENS,
    )
    parser.add_argument(
        "--max_space_index",
        help="How many spaces at most to consider",
        type=int,
        default=MAX_SPACE_INDEX,
    )
    parser.add_argument(
        "--n_samples",
        help="How many samples to stop after",
        type=int,
        default=N_TOTAL_SAMPLES,
    )
    parser.add_argument(
        "--hf_dataset_path",
        help="Path of the huggingface dataset to pull samplse from",
        default=WIKITEXT_PATH,
    )
    parser.add_argument(
        "--hf_dataset_name",
        help="Name of the huggingface dataset to pull samplse from",
        default=WIKITEXT_NAME,
    )
    parser.add_argument(
        "--hf_dataset_split",
        help="Which split of the huggingface dataset to use",
        default="train",
    )
    parser.add_argument("--regenerate", help="Regenerate the file if exists", action="store_true")
    parser.add_argument(
        "--device",
    )

    args = parser.parse_args(args)
    logger.info(f"Parsed arguments:\n{args}")

    save_path = Path(args.save_path_root)
    if args.save_path_suffix:
        save_path /= args.save_path_suffix

    save_name = f"{args.model_name[args.model_name.rfind('/') + 1 :]}_{args.hf_dataset_name}.csv.gz".lower()
    if args.max_length_tokens > 16:
        save_name = save_name.replace(".csv.gz", "_long.csv.gz")
    save_path /= save_name

    if save_path.exists() and not args.regenerate:
        logger.warning(f"Save path {str(save_path)} exists and regenerate is False, aborting...")
        return

    dataset = datasets.load_dataset(args.hf_dataset_path, args.hf_dataset_name, split=args.hf_dataset_split)
    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        device = f"cuda:{torch.cuda.device_count() - 1}"

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer sets the pad token id automatically when the pad token is set
        # tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # right-pad to get correct lengths to first EOS token

    rng = np.random.default_rng(args.random_seed)
    seen_indices = set()
    pbar = tqdm(desc="Samples", total=args.n_samples)

    queue_by_n_tokens = defaultdict(list)
    output_rows = []
    samples = 0

    def process_queue(queue):
        input_ids = torch.stack([t[3] for t in queue])
        logprobs = input_ids_to_logprobs(model, input_ids)
        output_rows.extend([(*tup[:-1], n_tokens, lp.item()) for (tup, lp) in zip(queue, logprobs)])

    while samples < args.n_samples:
        next_text_index = None
        while next_text_index is None or next_text_index in seen_indices:
            next_text_index = int(rng.integers(len(dataset)))

        seen_indices.add(next_text_index)
        next_text = dataset[next_text_index]["text"].strip()
        if not next_text:
            continue

        next_text = next_text.replace("<unk>", "").replace(" @-@ ", "-").strip()
        next_text = MULTIPLE_SPACES_PATTERN.sub(" ", next_text)
        space_indices = find(next_text, " ")[: args.max_space_index]
        partials = [next_text[:si].strip() for si in space_indices]
        if len(partials) == 0:
            continue

        tokens = tokenizer(partials, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        tokens_per_partial = (tokens != tokenizer.pad_token_id).sum(dim=-1)
        # the below can be made more efficient but it's surely not the bottleneck
        relevant_partial_indices = (
            torch.argwhere(
                (args.min_length_tokens <= tokens_per_partial) & (tokens_per_partial <= args.max_length_tokens)
            )
            .squeeze()
            .tolist()
        )

        if isinstance(relevant_partial_indices, int):
            relevant_partial_indices = [relevant_partial_indices]
        elif len(relevant_partial_indices) == 0:
            continue

        for idx in relevant_partial_indices:
            idx_tokens = tokens[idx]
            eos_idx = torch.argwhere(idx_tokens == tokenizer.pad_token_id).squeeze()
            eos_idx = eos_idx if eos_idx.ndim == 0 else (None if len(eos_idx) == 0 else eos_idx[0])
            relevant_tokens = idx_tokens[:eos_idx]
            queue_by_n_tokens[tokens_per_partial[idx].item()].append(
                (
                    next_text_index,
                    space_indices[idx],
                    partials[idx],
                    relevant_tokens,
                )
            )

        new_samples = len(relevant_partial_indices)
        samples += new_samples
        pbar.update(new_samples)
        pbar.set_postfix(
            dict(
                rows=len(output_rows),
                queue_total=sum(len(v) for v in queue_by_n_tokens.values()),
            )
        )

        for n_tokens, queue in queue_by_n_tokens.items():
            if len(queue) >= args.min_batch_size:
                process_queue(queue)
                queue_by_n_tokens[n_tokens] = []

    pbar.close()

    for n_tokens, queue in queue_by_n_tokens.items():
        if len(queue) > 0:
            process_queue(queue)

    df = pd.DataFrame(
        output_rows,
        columns=(
            "text_idx",
            "space_idx",
            "sentence",
            "n_tokens",
            "logprob",
        ),
    )
    logger.info(f"Created final df with shape {df.shape}, about to save to {str(save_path)}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(save_path), compression="gzip", index=False)


if __name__ == "__main__":
    main()
