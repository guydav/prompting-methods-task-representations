import argparse
import gc
import json
import os
import time
import typing
from pathlib import Path

import numpy as np
import torch
from git import Repo
from loguru import logger

from recipe.function_vectors.compute_indirect_effect import (
    PromptBaseline,
)
from recipe.function_vectors.generate_prompts_for_dataset import (
    LONG,
    PROMPT_TYPES,
    SHORT,
)
from recipe.function_vectors.utils.eval_utils import (
    compute_dataset_baseline,
    make_valid_path_name,
    prompt_based_eval_no_intervention,
)
from recipe.function_vectors.utils.extract_utils import get_prompt_based_mean_head_activations
from recipe.function_vectors.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from recipe.function_vectors.utils.prompt_utils import filter_prompts_by_max_tokens, load_dataset

STORAGE_ROOT = os.environ.get("STORAGE_ROOT")

SETTINGS_BY_PROMPT_TYPE = {
    SHORT: {
        "propmt_max_len_tokens": 16,
        "saved_prompts_suffix": "prompts",
    },
    LONG: {
        "propmt_max_len_tokens": 64,
        "saved_prompts_suffix": "long_prompts",
    },
}

IGNORE_OVERLAP_SKIP_CHECK_DATASETS = [
    "commonsense_qa",
]


def _get_git_commit_hex():
    p = Path(__file__).absolute()
    # p = Path(".").absolute()

    while not list(p.glob(".git")):
        p = p.parent
        if str(p) == "/":
            raise ValueError("Git repo not found in parents")

    repo = Repo(p)
    return repo.head.commit.hexsha


def _protected_decorator(func):
    def protected(output_path, output_data, *args, **kwargs):
        lock_path = Path(f"{output_path}.lock")
        if lock_path.exists():
            logger.warning(f"Skipping saving {output_path} as the lock file exists")
            return

        try:
            lock_path.touch()
            func(output_path, output_data, *args, **kwargs)
        finally:
            lock_path.unlink()

    return protected


@_protected_decorator
def _protected_json_dump(output_path, output_data):
    with open(output_path, "w") as results_file:
        json.dump(output_data, results_file, indent=2)


@_protected_decorator
def _protected_torch_save(output_path, output_data):
    torch.save(output_data, output_path)


def _gc_clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def prompt_filter_main(alt_args: typing.Optional[typing.List[str]] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to be loaded",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n_top_heads",
        help="Number of attenion head outputs used to compute function vector",
        required=False,
        type=int,
        default=10,
    )
    parser.add_argument(
        "--edit_layer",
        help="Layer for intervention. If -1, sweep over all layers",
        type=int,
        required=False,
        default=-1,
    )  #
    parser.add_argument(
        "--model_name",
        help="Name of model to be loaded",
        type=str,
        required=False,
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--root_data_dir",
        help="Root directory of data files",
        type=str,
        required=False,
        default=f"{STORAGE_ROOT}/function_vectors/dataset_files",
    )
    parser.add_argument(
        "--save_path_root",
        help="File path to save to",
        type=str,
        required=False,
        default=f"{STORAGE_ROOT}/function_vectors/full_results_prompt_based",
    )
    parser.add_argument(
        "--saved_prompts_root",
        help="File path to save to",
        type=str,
        required=False,
        default=f"{STORAGE_ROOT}/function_vectors/prompts",
    )
    parser.add_argument(
        "--saved_prompts_file",
        help="File name to load prompts from, defaults to [dataset_name]_prompts.json",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--saved_prompts_suffix",
        help="File name to load prompts from, defaults to [dataset_name]_prompts.json",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--save_path_suffix",
        help="Subdirectory to save results into within the results directory",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--ie_path_root",
        help="File path to load indirect effects from",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("--seed", help="Randomized seed", type=int, required=False, default=42)
    parser.add_argument(
        "--device",
        help="Device to run on",
        type=str,
        required=False,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mean_activations_path",
        help="Path to file containing mean_head_activations for the specified task",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--indirect_effect_path",
        help="Path to file containing indirect_effect scores for the specified task X prompt baseline",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_split",
        help="Percentage corresponding to test set split size",
        required=False,
        default=0.3,
    )

    prompt_arg_group = parser.add_argument_group("Propmt-based FV arguments")
    prompt_arg_group.add_argument(
        "--prompt_type",
        help="Which prompts to use",
        type=str,
        required=True,
        choices=PROMPT_TYPES,
    )
    prompt_arg_group.add_argument(
        "--n_best_prompts",
        help="How many prompts to use (default 5)",
        type=int,
        required=False,
        default=5,
    )
    prompt_arg_group.add_argument(
        "--propmt_max_len_tokens",
        help="Max length in tokens to use for a prompt (default 16)",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--force_prompt_evaluation",
        help="Flag for whether to force evaluating prompts even if the files exist",
        action="store_true",
        required=False,
    )
    prompt_arg_group.add_argument(
        "--allow_different_examples_per_prompt",
        help="Whether or not use the same or different examples for each prompt (default same)",
        action="store_true",
    )
    prompt_arg_group.add_argument(
        "--min_passing_examples",
        help="Smallest number of examples per/for all prompts to continue with; if None, defaults to the number of examples required for mean activations",
        type=int,
        required=False,
        default=None,
    )
    prompt_arg_group.add_argument(
        "--total_mean_activation_examples",
        help="Number total examples (split evenly between all prompts) for mean activations",
        type=int,
        required=False,
        default=100,
    )
    prompt_arg_group.add_argument(
        "--total_indirect_effect_examples",
        help="Number total examples (split evenly between all prompts) for mean activations",
        type=int,
        required=False,
        default=25,
    )
    prompt_arg_group.add_argument(
        "--n_icl_examples",
        help="Number of ICL examples to add to each prompt",
        type=int,
        required=False,
        default=0,
    )
    prompt_arg_group.add_argument(
        "--n_indirect_effect_trials",
        help="Number of baseline prompts to average over for indirect effect for each prompt",
        type=int,
        required=False,
        default=5,
    )
    prompt_arg_group.add_argument(
        "--prompt_baseline",
        help="Which prompt baseline to use",
        type=PromptBaseline,
        required=False,
        choices=list(PromptBaseline),  # type: ignore
    )

    class StoreDictKeyPair(argparse.Action):
        def __init__(self, option_strings, dest, *args, **kwargs):
            super().__init__(option_strings, dest, *args, **kwargs)

        def __call__(
            self,
            parser,
            namespace,
            values,
            option_string=None,
        ):
            my_dict = {}
            for kv in values.split(","):  # type: ignore
                k, v = kv.split("=")
                my_dict[k] = v
            setattr(namespace, self.dest, my_dict)

    prompt_arg_group.add_argument(
        "--baseline_generator_kwargs",
        help="Keyword arguments for the prompt baseline generator",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1,KEY2=VAL2...",
    )
    prompt_arg_group.add_argument(
        "--evaluation_prompts",
        help="Which prompt(s) to use when evaluating (default empty string)",
        action="store",
        nargs="+",
        required=False,
        default=[""],
    )
    prompt_arg_group.add_argument(
        "--n_eval_icl_examples",
        help="Number of ICL examples to evaluate with shuffled labels (default 10)",
        type=int,
        required=False,
        default=10,
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
        default={"input": "\n", "output": "\n\n", "instructions": "\n"},
    )
    parser.add_argument(
        "--compute_baseline",
        help="Whether to compute the model baseline 0-shot -> n-shot performance",
        type=bool,
        required=False,
        default=True,
    )
    parser.add_argument(
        "--generate_str",
        help="Whether to generate long-form completions for the task",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--metric",
        help="Metric to use when evaluating generated strings",
        type=str,
        required=False,
        default="f1_score",
    )
    parser.add_argument(
        "--universal_set",
        help="Flag for whether to evaluate using the univeral set of heads",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--force_evaluation",
        help="Flag for whether to force running evaluations even if the files exist",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--force_compute_baseline",
        help="Flag for whether to force computing baselines even if the files exist",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--split_validation",
        help="Whether or not to split a validation set",
        action="store_true",
    )
    parser.add_argument("--wandb_project", help="Which wandb project to log into", default="prompt-fv")
    parser.add_argument(
        "--dont_cache_prompt_prefixes",
        help="Whether or not to cache prompt prefixes",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for evaluation",
        type=int,
        required=False,
        default=0,
    )

    args = parser.parse_args(alt_args)
    prompt_type = args.prompt_type

    if args.propmt_max_len_tokens is None:
        args.propmt_max_len_tokens = SETTINGS_BY_PROMPT_TYPE[prompt_type]["propmt_max_len_tokens"]

    if args.saved_prompts_suffix is None:
        args.saved_prompts_suffix = SETTINGS_BY_PROMPT_TYPE[prompt_type]["saved_prompts_suffix"]

    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir

    short_model_name = args.model_name[args.model_name.rfind("/") + 1 :]
    save_path_suffix = (args.save_path_suffix if args.save_path_suffix is not None else "MODEL").replace(
        "MODEL", short_model_name
    )
    if not args.save_path_root.endswith(prompt_type):
        args.save_path_root = f"{args.save_path_root}_{prompt_type}"

    save_path_root = f"{args.save_path_root}/{save_path_suffix}/{dataset_name}"
    saved_prompts_file = (
        args.saved_prompts_file
        if args.saved_prompts_file is not None
        else f"{dataset_name}_{args.saved_prompts_suffix}.json"
    )
    saved_prompts_full_path = f"{args.saved_prompts_root}/{saved_prompts_file}"

    ie_path_root = f"{args.ie_path_root}/{dataset_name}" if args.ie_path_root else save_path_root
    seed = args.seed
    device = args.device
    mean_activations_path = args.mean_activations_path
    indirect_effect_path = args.indirect_effect_path
    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer

    test_split = float(args.test_split)
    split_validation = args.split_validation
    splits = ("train", "valid", "test") if split_validation else ("train", "test")
    n_best_prompts = args.n_best_prompts

    if args.total_mean_activation_examples % n_best_prompts:
        raise ValueError(
            f"Total mean activation examples ({args.total_mean_activation_examples}) must be divisible by the number of prompts ({n_best_prompts})"
        )
    args.mean_activation_trials_per_prompt = args.total_mean_activation_examples // n_best_prompts
    if args.total_indirect_effect_examples % n_best_prompts:
        raise ValueError(
            f"Total indirect effect examples ({args.total_indirect_effect_examples}) must be divisible by the number of prompts ({n_best_prompts})"
        )
    args.indirect_effect_trials_per_prompt = args.total_indirect_effect_examples // n_best_prompts

    if args.min_passing_examples is None:
        args.min_passing_examples = max(args.mean_activation_trials_per_prompt, args.indirect_effect_trials_per_prompt)

    n_icl_examples = args.n_icl_examples

    # prompt_baseline = args.prompt_baseline
    # baseline_generator_kwargs = args.baseline_generator_kwargs
    # if baseline_generator_kwargs is None:
    #     baseline_generator_kwargs = {}
    # if "prompt_type" not in baseline_generator_kwargs:
    #     baseline_generator_kwargs["prompt_type"] = prompt_type
    # if "rng" not in baseline_generator_kwargs:
    #     baseline_generator_kwargs["rng"] = np.random.default_rng(seed)
    # if "saved_prompts_root" not in baseline_generator_kwargs:
    #     baseline_generator_kwargs["saved_prompts_root"] = args.saved_prompts_root
    # if "saved_prompts_file" not in baseline_generator_kwargs:
    #     baseline_generator_kwargs["saved_prompts_file"] = saved_prompts_file
    # if "propmt_max_len_tokens" not in baseline_generator_kwargs:
    #     baseline_generator_kwargs["propmt_max_len_tokens"] = args.propmt_max_len_tokens
    # if "model_name" not in baseline_generator_kwargs:
    #     baseline_generator_kwargs["model_name"] = model_name
    # if "saved_prompts_suffix" not in baseline_generator_kwargs:
    #     baseline_generator_kwargs["saved_prompts_suffix"] = args.saved_prompts_suffix

    # evaluation_prompts = args.evaluation_prompts
    n_eval_icl_examples = args.n_eval_icl_examples

    prefixes = args.prefixes
    separators = args.separators
    compute_baseline = args.compute_baseline

    generate_str = args.generate_str
    metric = args.metric
    universal_set = args.universal_set

    if universal_set:
        raise NotImplementedError("Universal set of heads not yet computed")

    ignore_overlap_skip_check = dataset_name in IGNORE_OVERLAP_SKIP_CHECK_DATASETS
    if ignore_overlap_skip_check:
        logger.info(f"Ignoring overlap skip check for dataset {dataset_name}")

    args.commit = _get_git_commit_hex()
    args.slurm_job_id = os.environ.get("SLURM_JOB_ID")
    args.slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    args.slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    args.slurm_array_full_id = f"{args.slurm_array_job_id}_{args.slurm_array_task_id}"

    logger.debug(str(args))

    # wandb_tags = [
    #     save_path_suffix,
    #     dataset_name,
    #     prompt_baseline,
    #     f"{prompt_type}_prompts",
    # ]

    # wandb_run = wandb.init(
    #     project=args.wandb_project,
    #     id="|".join(wandb_tags),
    #     tags=wandb_tags,
    #     config=vars(args),
    # )

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    logger.info("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    args.few_shot_batch_size = None
    is_conll = "conll2003" in dataset_name
    if args.batch_size == 0:
        if "Llama-3.2-1B" in model_name:
            args.batch_size = 32 if is_conll else 128
        elif "Llama-3.2-3B" in model_name:
            args.batch_size = 32 if is_conll else 128
        elif ("Llama-2-7b" in model_name) or ("Llama-2-13b" in model_name):
            args.batch_size = 8 if is_conll else 32
        elif any(name in model_name for name in ("Llama-3.1-8B", "OLMo")):
            args.batch_size = 2 if is_conll else 8
        else:
            raise ValueError(f"Unknown model name provided with batch size 0 (auto-infer batch size): {model_name}")

        if args.few_shot_batch_size is None:
            args.few_shot_batch_size = max(1, args.batch_size // 8)

        logger.info(
            f"Batch size for model '{model_name}' and dataaset '{dataset_name}': {args.batch_size} / {args.few_shot_batch_size}"
        )

    if args.few_shot_batch_size is None:
        args.few_shot_batch_size = args.batch_size

    try:
        if args.edit_layer == -1:  # sweep over all layers if edit_layer=-1
            eval_edit_layer = [0, model_config["n_layers"]]

        # Load the dataset
        logger.info("Loading Dataset")
        set_seed(seed)
        if not os.path.exists(root_data_dir):
            raise ValueError(f"Dataset Directory Not Found: {root_data_dir}")

        dataset = load_dataset(
            dataset_name,
            root_data_dir=root_data_dir,
            test_size=test_split,
            seed=seed,
            split_valid=split_validation,
        )

        # if not os.path.exists(save_path_root):
        os.makedirs(save_path_root, exist_ok=True)

        # Load the prompts
        logger.info("Loading Prompts")
        if not os.path.exists(args.saved_prompts_root):
            raise ValueError(f"Prompts Directory Not Found: {args.saved_prompts_root}")

        with open(saved_prompts_full_path, "r") as prompts_file:
            prompts_data = json.load(prompts_file)

        assert prompts_data["dataset_name"] == dataset_name, (
            f"Dataset name mismatch, found {prompts_data['dataset_name']} in file, expected {dataset_name}"
        )
        all_task_prompts = prompts_data["prompts"]
        logger.info(f"Loaded {len(all_task_prompts)} prompts")

        # n_best_prompts = n_best_prompts if n_best_prompts is not None else len(prompts)
        # logger.info(f"Loaded {len(prompts)} prompts, using the first {n_best_prompts}")
        # prompts = prompts[:n_best_prompts]

        if args.propmt_max_len_tokens:
            keep_indices = filter_prompts_by_max_tokens(
                all_task_prompts,
                tokenizer=tokenizer,
                max_length_tokens=args.propmt_max_len_tokens,
            )
            if len(keep_indices) < n_best_prompts:
                logger.error(
                    f"Prompt max length filtering with {args.propmt_max_len_tokens} left only {len(keep_indices)} prompts, with n_best_prompts = {n_best_prompts}. Aborting..."
                )
                return

            logger.info(
                f"Filtering prompts to have at most {args.propmt_max_len_tokens} tokens, went from {len(all_task_prompts)} to {len(keep_indices)} prompts"
            )

            all_task_prompts = [all_task_prompts[i] for i in keep_indices]

        logger.info("Evaluating full prompt set")
        # 1. Compute Model Prompt-based Baseline & 2. Filter test set to cases where model gets it correct
        per_prompt_results_file_name = f"{save_path_root}/per_prompt_results.json"
        prompt_selection_results = None

        if os.path.exists(per_prompt_results_file_name) and not args.force_prompt_evaluation:
            try:
                logger.info("Loading full prompt filter results as they already exist")
                with open(per_prompt_results_file_name, "r") as f:
                    prompt_selection_results = json.load(f)
            except Exception as e:
                logger.exception(e)

        if prompt_selection_results is None:
            prompt_selection_results = {}
            for split in splits:
                split_partial_per_prompt_results_file_name = f"{per_prompt_results_file_name}.{split}.partial"

                if args.force_prompt_evaluation and os.path.exists(split_partial_per_prompt_results_file_name):
                    logger.info(
                        f"Removing partial results file '{split_partial_per_prompt_results_file_name}' since force flag is on"
                    )
                    os.remove(split_partial_per_prompt_results_file_name)

                prompt_selection_results[split] = prompt_based_eval_no_intervention(
                    dataset,
                    prompts=all_task_prompts,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    compute_ppl=not args.generate_str,
                    generate_str=args.generate_str,
                    metric=metric,
                    relevant_split=split,
                    prefixes=prefixes,
                    separators=separators,
                    partial_path=split_partial_per_prompt_results_file_name,
                    ignore_overlap_skip_check=ignore_overlap_skip_check,
                    cache_prompt_prefixes=not args.dont_cache_prompt_prefixes,
                    batch_size=args.batch_size,
                )

            logger.info(f"Saving full prompt filter results to {per_prompt_results_file_name}")
            args.fs_results_file_name = per_prompt_results_file_name
            _protected_json_dump(per_prompt_results_file_name, prompt_selection_results)

        # Filter out the best `n_best_prompts` prompts
        prompts_by_accuracy = [(p, pr[0][1]) for p, pr in prompt_selection_results["train"]["clean_topk"].items()]
        prompts_by_accuracy.sort(key=lambda t: t[1], reverse=True)
        selected_prompts = prompts_by_accuracy[:n_best_prompts]
        mean_accuracy = np.mean([p[1] for p in selected_prompts])
        selected_prompts = [p[0] for p in selected_prompts]
        logger.info(
            f"Using the following {n_best_prompts} propmts, whose mean task accuracy is {mean_accuracy:.4f}:\n{selected_prompts}"
        )

        filter_set_per_split = {}
        # Select relevant examples (per-prompt or global)
        if args.allow_different_examples_per_prompt:
            for split in splits:
                filter_set_per_split[split] = {}

                for p in selected_prompts:
                    prompt_example_rank_list = np.array(prompt_selection_results[split]["clean_rank_list"][p])
                    passing_indices = np.argwhere(prompt_example_rank_list == 0).squeeze()

                prompt_example_rank_list = np.sum(
                    [np.array(prompt_selection_results[split]["clean_rank_list"][p]) for p in selected_prompts],
                    axis=0,
                )
                passing_indices = np.argwhere(prompt_example_rank_list == 0).squeeze()
                n_passing_indices = 0 if passing_indices.ndim == 0 else len(passing_indices)

                if n_passing_indices < args.min_passing_examples:
                    if split == "train":
                        logger.error(
                            f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, min was set to {args.min_passing_examples}, aborting..."
                        )
                        # wandb.log(dict(failure_reason="insufficient train examples for prompt"))
                        return
                    else:
                        if n_passing_indices == 0:
                            logger.error(
                                f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, min was set to {args.min_passing_examples}, aborting..."
                            )
                            # wandb.log(dict(failure_reason="zero test examples for prompt"))
                            return
                        else:
                            logger.warning(
                                f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, min was set to {args.min_passing_examples}, ..."
                            )

                filter_set_per_split[split][p] = passing_indices

        else:
            for split in splits:
                prompt_example_rank_list = np.sum(
                    [np.array(prompt_selection_results[split]["clean_rank_list"][p]) for p in selected_prompts],
                    axis=0,
                )
                passing_indices = np.argwhere(prompt_example_rank_list == 0).squeeze()
                n_passing_indices = 0 if passing_indices.ndim == 0 else len(passing_indices)
                if n_passing_indices < args.min_passing_examples:
                    if split == "train":
                        logger.error(
                            f"Found {n_passing_indices} that pass all prompts in the {split} split, min was set to {args.min_passing_examples}, aborting..."
                        )
                        # wandb.log(dict(failure_reason="insufficient shared train examples for prompts"))
                        return
                    else:
                        if n_passing_indices == 0:
                            logger.error(
                                f"Found {n_passing_indices} that pass all prompts in the {split} split, min was set to {args.min_passing_examples}, aborting..."
                            )
                            # wandb.log(dict(failure_reason="insufficient shared test examples for prompts"))
                            return
                        else:
                            logger.warning(
                                f"Found {n_passing_indices} that pass all prompts in the {split} split, min was set to {args.min_passing_examples}..."
                            )
                filter_set_per_split[split] = passing_indices

        _gc_clear_cache()

        # Load or Re-Compute mean_head_activations
        if mean_activations_path is not None and os.path.exists(mean_activations_path):
            mean_activations = torch.load(mean_activations_path)
        elif mean_activations_path is None and os.path.exists(
            f"{ie_path_root}/{dataset_name}_mean_head_activations.pt"
        ):
            mean_activations_path = f"{ie_path_root}/{dataset_name}_mean_head_activations.pt"
            mean_activations = torch.load(mean_activations_path)
        else:
            logger.info(
                f"Computing Mean Activations with {args.total_mean_activation_examples} = {args.mean_activation_trials_per_prompt} examples * {n_best_prompts} prompts"
            )
            set_seed(seed)

            mean_activations = get_prompt_based_mean_head_activations(
                dataset,
                prompts=selected_prompts,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                n_trials_per_prompt=args.mean_activation_trials_per_prompt,
                prefixes=prefixes,
                separators=separators,
                filter_set=filter_set_per_split["train"],
                n_icl_examples=n_icl_examples,
                query_dataset="train",
                batch_size=args.batch_size,
            )
            args.mean_activations_path = f"{save_path_root}/{dataset_name}_mean_head_activations.pt"

            _protected_torch_save(args.mean_activations_path, mean_activations)

        _gc_clear_cache()

        if compute_baseline:
            baseline_file_name = f"{save_path_root}/model_icl_baseline.json"

            if os.path.exists(baseline_file_name) and not args.force_compute_baseline:
                logger.info("Skipping baseline since file exists and force flag is off")

            else:
                baseline_file_name = make_valid_path_name(baseline_file_name)
                args.baseline_file_name = baseline_file_name
                logger.info(
                    f"Computing model baseline results for {n_eval_icl_examples}-shots",
                    flush=True,
                )
                baseline_results = compute_dataset_baseline(
                    dataset,
                    model,
                    model_config,
                    tokenizer,
                    n_shots=n_eval_icl_examples,
                    seed=seed,
                    prefixes=prefixes,
                    separators=separators,
                    batch_size=args.few_shot_batch_size,
                )

                _protected_json_dump(baseline_file_name, baseline_results)

        logger.info("Prompt filter main done")

    finally:
        # if wandb_run is not None:
        #     wandb_run.finish()
        if model is not None:
            del model
        _gc_clear_cache()
        time.sleep(10)

    return


if __name__ == "__main__":
    prompt_filter_main()
