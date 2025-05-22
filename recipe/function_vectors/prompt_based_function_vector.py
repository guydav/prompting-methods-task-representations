import argparse
import datetime
import gc
import itertools
import json
import os
import platform
import time
import typing
from pathlib import Path

import numpy as np
import torch
import wandb
from git import Repo
from loguru import logger

from recipe.function_vectors.compute_indirect_effect import (
    PromptBaseline,
    compute_prompt_based_indirect_effect,
)
from recipe.function_vectors.generate_prompts_for_dataset import (
    LONG,
    PROMPT_TYPES,
    SHORT,
)
from recipe.function_vectors.utils.eval_utils import (
    compute_dataset_baseline,
    make_valid_path_name,
    prompt_based_eval,
    prompt_based_eval_no_intervention,
)
from recipe.function_vectors.utils.extract_utils import (
    compute_function_vector,
    compute_universal_function_vector_top_heads_from_file,
    get_prompt_based_mean_head_activations,
)
from recipe.function_vectors.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from recipe.function_vectors.utils.prompt_utils import (
    filter_prompts_by_max_tokens,
    load_dataset,
)

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


class FailureException(Exception):
    pass


def _log_failure(output_path, failure_code, failure_reason):
    if not output_path.lower().endswith(".json"):
        output_path += ".json"

    if wandb.run is not None:
        wandb.log(dict(failure_reason=failure_reason))
        logger.warning(f"Logging failure to {output_path}")
    else:
        logger.warning(f"Logging failure to {output_path} (without wandb as it is not initialized)")

    _protected_json_dump(
        output_path,
        {
            "code": failure_code,
            "reason": failure_reason,
            "timestamp": datetime.datetime.now().isoformat(),
        },
    )

    raise FailureException(failure_reason)


def _mean_act_indirect_effect_fv(
    args,
    mean_activations_path,
    top_heads_path,
    fv_path,
    model,
    tokenizer,
    model_config,
    dataset,
    selected_prompts,
    filter_set_per_split,
    save_path_root,
    indirect_effect_path,
    baseline_generator_kwargs,
    multiple_fvs: bool = False,
):
    dataset_name = args.dataset_name
    n_best_prompts = args.n_best_prompts
    n_icl_examples = args.n_icl_examples
    n_top_heads = args.n_top_heads
    prefixes = args.prefixes
    prompt_baseline = args.prompt_baseline
    seed = args.seed
    separators = args.separators
    universal_set = args.universal_set

    # Load or Re-Compute mean_head_activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        logger.info(f"Loading Mean Activations from {mean_activations_path}")
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

    # Compute function vector
    fv = None
    top_heads = None
    if universal_set:
        if os.path.exists(fv_path):
            logger.info(f"Loading universal function vector from {fv_path}")
            fv = torch.load(fv_path)
        else:
            logger.info(
                f"Loading top heads from {top_heads_path} to compute universal function vector and saving to {fv_path}"
            )
            fv, top_heads = compute_universal_function_vector_top_heads_from_file(
                mean_activations,
                model,
                model_config=model_config,
                top_heads_path=top_heads_path,
                n_top_heads=n_top_heads,
            )
            _protected_torch_save(fv_path, fv)

    else:
        if multiple_fvs:
            raise ValueError("multiple_fvs is not supported for non-universal set FV")

        # Load or Re-Compute indirect_effect values -- only necessary in the non-universal case
        if indirect_effect_path is not None and os.path.exists(indirect_effect_path) and not args.force_indirect_effect:
            logger.info(f"Loading Indirect Effects from {indirect_effect_path}")
            indirect_effect = torch.load(indirect_effect_path)
        elif not universal_set:  # Only compute indirect effects if we need to
            logger.info(
                f"Computing Indirect Effects with {args.total_indirect_effect_examples} = {args.indirect_effect_trials_per_prompt} examples * {n_best_prompts} prompts"
            )
            set_seed(seed)
            args.partial_indirect_effect_path = f"{indirect_effect_path}.partial"

            indirect_effect = compute_prompt_based_indirect_effect(
                dataset,
                selected_prompts,
                mean_activations,
                baseline=prompt_baseline,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                n_trials_per_prompt=args.indirect_effect_trials_per_prompt,
                last_token_only=True,
                prefixes=prefixes,
                separators=separators,
                filter_set=filter_set_per_split["train"],
                n_icl_examples=n_icl_examples,
                partial_path=args.partial_indirect_effect_path,
                query_dataset="train",
                baseline_generator_kwargs=baseline_generator_kwargs,
                batch_size=args.batch_size,
                forced=args.force_indirect_effect,
            )

            _protected_torch_save(args.indirect_effect_path, indirect_effect)

        _gc_clear_cache()

        if os.path.exists(fv_path):
            fv = torch.load(fv_path)
        else:
            fv, top_heads = compute_function_vector(
                mean_activations,
                indirect_effect,
                model,
                model_config=model_config,
                n_top_heads=n_top_heads,
                prompt_based=True,
            )
            _protected_torch_save(fv_path, fv)
            _protected_torch_save(top_heads_path, top_heads)

    _gc_clear_cache()
    return fv


def _multiple_mean_act_indirect_effect_fv(
    args,
    mean_activations_paths,
    top_heads_paths,
    fv_paths,
    model,
    tokenizer,
    model_config,
    dataset,
    selected_prompts,
    filter_set_per_split,
    save_path_root,
    indirect_effect_path,
    baseline_generator_kwargs,
):
    if not isinstance(mean_activations_paths, list):
        raise ValueError("mean_activations_paths should be a list of paths")

    if not isinstance(top_heads_paths, list):
        raise ValueError("top_heads_paths should be a list of paths")

    if not isinstance(fv_paths, list):
        raise ValueError("fv_paths should be a list of paths")

    if len(mean_activations_paths) != len(top_heads_paths) or len(mean_activations_paths) != len(fv_paths):
        raise ValueError("mean_activations_paths, top_heads_paths, and fv_paths should have the same length")

    return [
        _mean_act_indirect_effect_fv(
            args,
            mean_activations_path,
            top_heads_path,
            fv_path,
            model,
            tokenizer,
            model_config,
            dataset,
            selected_prompts,
            filter_set_per_split,
            save_path_root,
            indirect_effect_path,
            baseline_generator_kwargs,
            multiple_fvs=True,
        )
        for (mean_activations_path, top_heads_path, fv_path) in zip(mean_activations_paths, top_heads_paths, fv_paths)
    ]


def _inner_run_eval(
    args,
    fv_or_fvs,
    edit_layer_or_layers,
    pred_filepath,
    n_icl_examples,
    shuffle_icl_labels,
    batch_size,
    model,
    tokenizer,
    model_config,
    dataset,
    filter_set_per_split,
):
    evaluation_prompts = args.evaluation_prompts
    generate_str = args.generate_str
    metric = args.metric
    prefixes = args.prefixes
    seed = args.seed
    separators = args.separators

    set_seed(seed)
    if generate_str:
        results = prompt_based_eval(
            dataset=dataset,
            fv_vector_or_vectors=fv_or_fvs,
            edit_layer_or_layers=edit_layer_or_layers,
            prompts=evaluation_prompts,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            filter_set=filter_set_per_split["test"],
            generate_str=generate_str,
            metric=metric,
            pred_filepath=pred_filepath,
            prefixes=prefixes,
            separators=separators,
            query_dataset="test",
            n_icl_examples=n_icl_examples,
            shuffle_icl_labels=shuffle_icl_labels,
            batch_size=batch_size,
        )
    else:
        results = prompt_based_eval(
            dataset=dataset,
            fv_vector_or_vectors=fv_or_fvs,
            edit_layer_or_layers=edit_layer_or_layers,
            prompts=evaluation_prompts,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            filter_set=filter_set_per_split["test"],
            prefixes=prefixes,
            separators=separators,
            query_dataset="test",
            n_icl_examples=n_icl_examples,
            shuffle_icl_labels=shuffle_icl_labels,
            batch_size=batch_size,
        )
    _gc_clear_cache()

    return results


def _run_zs_eval(
    args,
    fv_or_fvs,
    edit_layer_or_layers,
    pred_filepath,
    model,
    tokenizer,
    model_config,
    dataset,
    filter_set_per_split,
):
    return _inner_run_eval(
        args,
        fv_or_fvs,
        edit_layer_or_layers,
        pred_filepath,
        0,
        False,
        args.batch_size,
        model,
        tokenizer,
        model_config,
        dataset,
        filter_set_per_split,
    )


def _run_fss_eval(
    args,
    fv_or_fvs,
    edit_layer_or_layers,
    pred_filepath,
    model,
    tokenizer,
    model_config,
    dataset,
    filter_set_per_split,
):
    return _inner_run_eval(
        args,
        fv_or_fvs,
        edit_layer_or_layers,
        pred_filepath,
        args.n_eval_icl_examples,
        True,
        args.few_shot_batch_size,
        model,
        tokenizer,
        model_config,
        dataset,
        filter_set_per_split,
    )


def prompt_function_vector_main(alt_args: typing.Optional[typing.List[str]] = None):
    wandb_run = None
    model = None

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
        default=20,
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
        "--revision", help="Specify model checkpoints for pythia or olmo models", type=str, required=False, default=None
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
        required=True,
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

    universal_prompt_arg_group = parser.add_argument_group("Propmt-based universal FV arguments")
    universal_prompt_arg_group.add_argument(
        "--universal_set",
        help="Flag for whether to evaluate using the univeral set of heads",
        action="store_true",
        required=False,
    )
    universal_prompt_arg_group.add_argument(
        "--top_heads_dir",
        help="Directory to load top heads from",
        type=str,
        required=False,
        default=f"{STORAGE_ROOT}/function_vectors/full_results_top_heads",
    )
    universal_prompt_arg_group.add_argument(
        "--top_heads_prompt_type",
        help="Load universasl set computed from which propmt types",
        type=str,
        choices=PROMPT_TYPES + ["both"],
        default="both",
    )
    universal_prompt_arg_group.add_argument(
        "--top_heads_baseline",
        help="Load universal set computed from which baseline",
        type=str,
        choices=list(PromptBaseline) + ["all"],
        required=False,
        default="all",
    )
    universal_prompt_arg_group.add_argument(
        "--joint_intervention_icl_root",
        help="Where to load ICL mean activations for joint intervention",
        type=str,
        default=f"{STORAGE_ROOT}/function_vectors/full_icl_results_same_test_sets",
    )
    universal_prompt_arg_group.add_argument(
        "--joint_intervention_min_layer_depth",
        help="Minimum layer depth for joint intervention",
        type=float,
        default=0.25,
    )
    universal_prompt_arg_group.add_argument(
        "--joint_intervention_max_layer_depth",
        help="Maximum layer depth for joint intervention",
        type=float,
        default=0.5,
    )
    universal_experiment_type_group = universal_prompt_arg_group.add_mutually_exclusive_group(required=False)
    universal_experiment_type_group.add_argument(
        "--joint_intervention",
        help="Flag for whether to use joint intervention  (ICL and prompts)",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--use_icl_top_heads",
        help="Flag for whether to use ICL top heads",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--use_icl_mean_activations",
        help="Flag for whether to use ICL mean activations",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--add_prompt_fv_twice",
        help="Flag for whether to add the prompt FV twice",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--add_icl_fv_twice",
        help="Flag for whether to add the ICL FV twice",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--use_min_abs_heads_prompt",
        help="Flag for whether to use the min absolute IE heads with prompt mean activations",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--use_min_abs_heads_icl",
        help="Flag for whether to use the min absolute IE heads with ICL mean activations",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--use_bottom_heads_prompt",
        help="Flag for whether to use the bottom prompt-based heads and activations",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--use_bottom_heads_icl",
        help="Flag for whether to use the bottom ICL heads and activations",
        action="store_true",
    )
    universal_experiment_type_group.add_argument(
        "--use_instruct_model_fv",
        help="Flag for whether to use the instruct model function vector in the base model",
        action="store_true",
    )

    universal_prompt_arg_group.add_argument(
        "--instruct_model_suffix",
        help="Which instruct model to pull a universal fv from if `--use_instruct_model_fv` is set",
        default="-Instruct",
    )
    universal_prompt_arg_group.add_argument(
        "--remove_model_suffix",
        help="Which model suffix to remove before adding `--instruct_model_suffix` if `--use_instruct_model_fv` is set",
        default=None,
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
        "--force_indirect_effect",
        help="Flag for whether to force computing the indirect effect (and rerunning evaluations) even if the files exist",
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
        "--cache_prompt_prefixes",
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
    args.short_model_name = args.model_name[args.model_name.rfind("/") + 1 :]
    root_data_dir = args.root_data_dir
    save_path_suffix = args.save_path_suffix if args.save_path_suffix is not None else args.short_model_name
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
    prompt_baseline = args.prompt_baseline

    indirect_effect_path = args.indirect_effect_path
    if indirect_effect_path is None:
        indirect_effect_path = f"{ie_path_root}/{dataset_name}_{prompt_baseline}_indirect_effect.pt"
        args.indirect_effect_path = indirect_effect_path

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

    if args.force_indirect_effect and not args.force_evaluation:
        logger.warning(
            "Force indirect effect flag was set, but force evaluation was not, setting force evaluation to True"
        )
        args.force_evaluation = True

    baseline_generator_kwargs = args.baseline_generator_kwargs
    if baseline_generator_kwargs is None:
        baseline_generator_kwargs = {}
    if "prompt_type" not in baseline_generator_kwargs:
        baseline_generator_kwargs["prompt_type"] = prompt_type
    if "rng" not in baseline_generator_kwargs:
        baseline_generator_kwargs["rng"] = np.random.default_rng(seed)
    if "saved_prompts_root" not in baseline_generator_kwargs:
        baseline_generator_kwargs["saved_prompts_root"] = args.saved_prompts_root
    if "saved_prompts_file" not in baseline_generator_kwargs:
        baseline_generator_kwargs["saved_prompts_file"] = saved_prompts_file
    if "propmt_max_len_tokens" not in baseline_generator_kwargs:
        baseline_generator_kwargs["propmt_max_len_tokens"] = args.propmt_max_len_tokens
    if "model_name" not in baseline_generator_kwargs:
        baseline_generator_kwargs["model_name"] = model_name
    if "saved_prompts_suffix" not in baseline_generator_kwargs:
        baseline_generator_kwargs["saved_prompts_suffix"] = args.saved_prompts_suffix

    # evaluation_prompts = args.evaluation_prompts
    n_eval_icl_examples = args.n_eval_icl_examples

    prefixes = args.prefixes
    separators = args.separators
    compute_baseline = args.compute_baseline

    # generate_str = args.generate_str
    metric = args.metric
    universal_set = args.universal_set
    joint_intervention = args.joint_intervention
    use_icl_mean_activations = args.use_icl_mean_activations
    use_icl_top_heads = args.use_icl_top_heads
    add_prompt_fv_twice = args.add_prompt_fv_twice
    add_icl_fv_twice = args.add_icl_fv_twice
    use_min_abs_heads_prompt = args.use_min_abs_heads_prompt
    use_min_abs_heads_icl = args.use_min_abs_heads_icl
    use_bottom_heads_prompt = args.use_bottom_heads_prompt
    use_bottom_heads_icl = args.use_bottom_heads_icl
    use_instruct_model_fv = args.use_instruct_model_fv

    # Baseline key setting values
    if args.mean_activations_path is None:
        args.mean_activations_path = f"{ie_path_root}/{dataset_name}_mean_head_activations.pt"
    args.fv_path = f"{save_path_root}/{dataset_name}_{prompt_baseline}_fv.pt"
    args.top_heads_path = f"{save_path_root}/{dataset_name}_{prompt_baseline}_top_heads.pt"

    n_universal_only_flags = (
        int(joint_intervention)
        + int(use_icl_mean_activations)
        + int(use_icl_top_heads)
        + int(add_prompt_fv_twice)
        + int(add_icl_fv_twice)
        + int(use_min_abs_heads_prompt)
        + int(use_min_abs_heads_icl)
        + int(use_bottom_heads_prompt)
        + int(use_bottom_heads_icl)
        + int(use_instruct_model_fv)
    )

    if universal_set:
        if n_universal_only_flags > 1:
            raise ValueError(
                f"Cannot use joint intervention with more than one universal-only flag. Please choose one. Received joint_intervention={joint_intervention}, use_icl_mean_activations={use_icl_mean_activations}, use_icl_top_heads={use_icl_top_heads}, add_prompt_fv_twice={add_prompt_fv_twice}, add_icl_fv_twice={add_icl_fv_twice}, use_min_abs_heads_prompt={use_min_abs_heads_prompt}, use_min_abs_heads_icl={use_min_abs_heads_icl}, use_bottom_heads_prompt={use_bottom_heads_prompt}, use_bottom_heads_icl={use_bottom_heads_icl}, use_instruct_model_fv={use_instruct_model_fv}"
            )

        icl_top_heads_path = f"{args.top_heads_dir}/{args.short_model_name}_icl_same_test_sets_top_heads.json"
        icl_mean_activations_path = f"{args.joint_intervention_icl_root}/{save_path_suffix}/{dataset_name}/{dataset_name}_mean_head_activations.pt"
        icl_fv_path = f"{save_path_root}/{dataset_name}_{n_top_heads}_universal_fv.pt"

        args.universal_fv_type = f"{args.top_heads_prompt_type}_{args.top_heads_baseline}"
        args.top_heads_path = f"{args.top_heads_dir}/{args.short_model_name}_{args.universal_fv_type}_top_heads.json"

        if use_icl_top_heads:
            args.top_heads_path = icl_top_heads_path
            logger.info(f"Using ICL top heads from {args.top_heads_path}")
            args.universal_fv_type = "icl_top_heads"

        if use_icl_mean_activations:
            args.mean_activations_path = icl_mean_activations_path
            logger.info(f"Using ICL mean activations from {args.mean_activations_path}")
            args.universal_fv_type = "icl_mean_activations"

        if add_icl_fv_twice:
            logger.info("Adding ICL FV twice")
            args.top_heads_path = icl_top_heads_path
            args.mean_activations_path = icl_mean_activations_path
            args.universal_fv_type = "icl_fv_twice"

        if add_prompt_fv_twice:
            logger.info("Adding prompt FV twice")
            args.universal_fv_type = "prompt_fv_twice"

        if use_min_abs_heads_prompt or use_min_abs_heads_icl:
            args.top_heads_path = f"{args.top_heads_dir}/{args.short_model_name}_universal_all_min_abs_heads.json"
            logger.info(f"Using min absolute top heads from {args.top_heads_path}")
            if use_min_abs_heads_icl:
                logger.info("Using min absolute heads with ICL activations")
                args.mean_activations_path = icl_mean_activations_path
                args.universal_fv_type = "min_abs_heads_icl"
            else:
                logger.info("Using min absolute heads with prompt activations")
                args.universal_fv_type = "min_abs_heads_prompt"

        if use_bottom_heads_prompt:
            args.top_heads_path = args.top_heads_path.replace("top_heads.json", "bottom_heads.json")
            logger.info(f"Using bottom prompt-based heads from {args.top_heads_path}")
            args.universal_fv_type = "bottom_prompt_heads"

        if use_bottom_heads_icl:
            args.top_heads_path = icl_top_heads_path.replace("top_heads.json", "bottom_heads.json")
            logger.info(f"Using bottom prompt-based heads from {args.top_heads_path}")
            args.mean_activations_path = icl_mean_activations_path
            logger.info(f"Using ICL mean activations from {icl_mean_activations_path}")
            args.universal_fv_type = "bottom_icl_heads"

        if use_instruct_model_fv:
            instruct_model_short_name = args.short_model_name
            if args.remove_model_suffix is not None:
                args.remove_model_suffix = args.remove_model_suffix.replace('"', "").replace("'", "")
                instruct_model_short_name = instruct_model_short_name.replace(args.remove_model_suffix, "")

            args.instruct_model_suffix = args.instruct_model_suffix.replace('"', "").replace("'", "")
            instruct_model_short_name = f"{instruct_model_short_name}{args.instruct_model_suffix}"
            logger.info(f"Using instruct model {instruct_model_short_name} for universal FV")
            args.top_heads_path = args.top_heads_path.replace(args.short_model_name, instruct_model_short_name)
            logger.info(f"Using instruct model top heads from {args.top_heads_path}")
            args.mean_activations_path = args.mean_activations_path.replace(
                args.short_model_name, instruct_model_short_name
            )
            logger.info(f"Using instruct model mean activations from {args.mean_activations_path}")
            args.universal_fv_type = "instruct_model"

        args.fv_path = f"{save_path_root}/{dataset_name}_{args.universal_fv_type}_{n_top_heads}_heads_universal_fv.pt"

    elif n_universal_only_flags != 0:
        raise ValueError("Joint intervention/ICL mean acts/top heads are not supported for non-universal set FV")

    # In the universal set case, the mean activations and top head paths must exist already
    for path, name, must_exist in zip(
        (args.mean_activations_path, args.top_heads_path, args.fv_path),
        ("mean_activations", "top_heads", "function_vector"),
        (universal_set, universal_set, False),
    ):
        if path is None:
            raise ValueError(f"args.{name} path is None. Please check the arguments.")
        if must_exist and not os.path.exists(path):
            if name == "mean_activations":
                logged_failures = list(Path(path).parent.glob("*failure.json"))
                if len(logged_failures) > 0:
                    failure_path = logged_failures[0]
                    with open(failure_path, "r") as f:
                        failure_data = json.load(f)
                    failure_code = failure_data.get("code")
                    failure_reason = failure_data.get("reason")
                    logger.error(
                        f"Mean activations file not found. Found previous failure code: {failure_code}, reason: {failure_reason}"
                    )

            raise ValueError(f"args.{name} file not found: {path}. Please generate the {name} file first.")

    args.few_shot_batch_size = None
    # model_size = extract_model_size(model_name)
    is_conll = "conll2003" in dataset_name
    if args.batch_size == 0:
        if any(name in model_name for name in ("Llama-3.1-8B", "OLMo-2-1124-7B")):
            args.batch_size = 2 if is_conll else 3
            args.few_shot_batch_size = 1
        else:
            # We can probably afford more; but since mean activations runs with N = 20/prompt
            # and indirect effect with 5/prompt, this only really slows us down in evals, which is fine for now
            args.batch_size = 5

        if any(name in model_name.lower() for name in ("llama-2", "olmo")) and is_conll:
            args.few_shot_batch_size = 1

    if args.few_shot_batch_size is None:
        args.few_shot_batch_size = args.batch_size

    args.commit = _get_git_commit_hex()
    args.slurm_job_id = os.environ.get("SLURM_JOB_ID")
    args.slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    args.slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    args.slurm_array_full_id = f"{args.slurm_array_job_id}_{args.slurm_array_task_id}"
    args.hostname = platform.node()

    args.failure_path = f"{save_path_root}/{prompt_baseline}_failure.json"

    logger.debug(str(args))

    wandb_tags = [
        save_path_suffix,
        dataset_name,
        prompt_baseline,
        f"{prompt_type}_prompts",
    ]

    wandb_run = wandb.init(
        project=args.wandb_project,
        id="|".join(wandb_tags),
        tags=wandb_tags,
        config=vars(args),
    )

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    logger.info("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, revision=args.revision)

    if "should_prepend_bos" not in baseline_generator_kwargs:
        baseline_generator_kwargs["should_prepend_bos"] = not model_config["prepend_bos"]

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
        logger.debug(
            f"Loaded dataset {dataset_name} with the following sizes: { {k: len(d) for k, d in dataset.items()} }"
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
                _log_failure(args.failure_path, "prompt_length_filter", "Insufficient prompts after length filter")

            logger.info(
                f"Filtering prompts to have at most {args.propmt_max_len_tokens} tokens, went from {len(all_task_prompts)} to {len(keep_indices)} prompts"
            )

            all_task_prompts = [all_task_prompts[i] for i in keep_indices]

        # 1. Compute Model Prompt-based Baseline & 2. Filter test set to cases where model gets it correct
        per_prompt_results_file_name = f"{save_path_root}/per_prompt_results.json"
        prompt_selection_results = None

        if os.path.exists(per_prompt_results_file_name) and not args.force_prompt_evaluation:
            try:
                with open(per_prompt_results_file_name, "r") as f:
                    prompt_selection_results = json.load(f)
            except Exception as e:
                logger.exception(e)

        if prompt_selection_results is None:
            logger.info("Evaluating full prompt set")

            prompt_selection_results = {}
            for split in splits:
                split_partial_per_prompt_results_file_name = f"{per_prompt_results_file_name}.{split}.partial"

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
                    cache_prompt_prefixes=args.cache_prompt_prefixes,
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
                    n_passing_indices = 0 if passing_indices.ndim == 0 else len(passing_indices)

                    if n_passing_indices < args.min_passing_examples:
                        if split == "train":
                            logger.error(
                                f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, min was set to {args.min_passing_examples}, aborting..."
                            )
                            _log_failure(
                                args.failure_path, "prompt_train_examples", "Insufficient train examples for prompt"
                            )
                        else:
                            if n_passing_indices == 0:
                                logger.error(
                                    f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, musst have at least one test example, aborting..."
                                )
                                _log_failure(args.failure_path, "prompt_test_examples", "Zero test examples for prompt")
                            else:
                                logger.warning(
                                    f"Found a prompt that only {n_passing_indices} examples pass in the {split} split, min was set to {args.min_passing_examples}, ..."
                                )

                    filter_set_per_split[split][p] = passing_indices

        else:  # same filter set across all prompts
            for split in splits:
                selected_prompt_rank_lists = [
                    np.array(prompt_selection_results[split]["clean_rank_list"][p]) for p in selected_prompts
                ]
                prompt_example_rank_list = np.sum(
                    selected_prompt_rank_lists,
                    axis=0,
                )
                passing_indices = np.argwhere(prompt_example_rank_list == 0).squeeze()
                n_passing_indices = 0 if passing_indices.ndim == 0 else len(passing_indices)
                n_passing_per_prompt = [np.sum(p == 0) for p in selected_prompt_rank_lists]

                if n_passing_indices < args.min_passing_examples:
                    if split == "train":
                        logger.error(
                            f"Found {n_passing_indices} that pass all prompts in the {split} split (with {n_passing_per_prompt} for each prompt), min was set to {args.min_passing_examples}, aborting..."
                        )
                        _log_failure(
                            args.failure_path,
                            "shared_train_examples",
                            "Insufficient shared train examples for top prompts",
                        )

                    else:
                        if n_passing_indices == 0:
                            logger.error(
                                f"Found {n_passing_indices} that pass all prompts in the {split} split, this split requires at least one example, aborting..."
                            )
                            _log_failure(
                                args.failure_path, "shared_test_examples", "Zero shared test examples for top prompts"
                            )
                        else:
                            logger.warning(
                                f"Found {n_passing_indices} that pass all prompts in the {split} split, min was set to {args.min_passing_examples}..."
                            )
                filter_set_per_split[split] = passing_indices

        _gc_clear_cache()

        # Load or Re-Compute mean_head_activations, indirect_effect, and function vector
        fv = _mean_act_indirect_effect_fv(
            args,
            args.mean_activations_path,
            args.top_heads_path,
            args.fv_path,
            model,
            tokenizer,
            model_config,
            dataset,
            selected_prompts,
            filter_set_per_split,
            save_path_root,
            indirect_effect_path,
            baseline_generator_kwargs,
        )

        joint_fvs = None
        if joint_intervention:
            logger.info(f"Loading ICL fv for joint intervention from {icl_fv_path}")
            icl_fv = _mean_act_indirect_effect_fv(
                args,
                icl_mean_activations_path,
                icl_top_heads_path,
                icl_fv_path,
                model,
                tokenizer,
                model_config,
                dataset,
                selected_prompts,
                filter_set_per_split,
                save_path_root,
                indirect_effect_path,
                baseline_generator_kwargs,
            )
            joint_fvs = [fv, icl_fv]
            args.universal_fv_type = "joint_intervention"

        elif add_icl_fv_twice or add_prompt_fv_twice:
            joint_fvs = [fv, fv]

        # Run evaluation
        eval_identifer = f"universal_{args.universal_fv_type}_{n_top_heads}_heads" if universal_set else prompt_baseline

        if n_universal_only_flags > 0:
            results_file_suffix = "mini_sweep.json"
        elif isinstance(eval_edit_layer, int):
            results_file_suffix = f"editlayer_{eval_edit_layer}.json"
        else:
            results_file_suffix = "layer_sweep.json"

        zs_results_file_name = make_valid_path_name(
            f"{save_path_root}/zs_results_{eval_identifer}_{results_file_suffix}"
        )
        args.zs_results_file_name = zs_results_file_name

        fs_shuffled_results_file_name = make_valid_path_name(
            f"{save_path_root}/fs_shuffled_results_{eval_identifer}_{results_file_suffix}"
        )
        args.fs_shuffled_results_file_name = fs_shuffled_results_file_name

        if (
            os.path.exists(zs_results_file_name)
            and os.path.exists(fs_shuffled_results_file_name)
            and not args.force_evaluation
        ):
            logger.info("Skipping evaluations since both files exist and flag was not set")

        else:
            # Run a two-argument sweep
            if joint_intervention or add_icl_fv_twice or add_prompt_fv_twice:
                min_depth = args.joint_intervention_min_layer_depth
                min_edit_layer = (
                    int(min_depth) if min_depth > 1 else int(np.floor(min_depth * model_config["n_layers"]))
                )
                max_depth = args.joint_intervention_max_layer_depth
                max_edit_layer = int(max_depth) if max_depth > 1 else int(np.ceil(max_depth * model_config["n_layers"]))
                args.min_edit_layer = min_edit_layer
                args.max_edit_layer = max_edit_layer
                if min_edit_layer >= max_edit_layer:
                    raise ValueError(
                        f"Minimum edit layer ({min_edit_layer}) must be less than maximum edit layer ({max_edit_layer})"
                    )
                logger.info(
                    f"Running `{args.universal_fv_type}` evaluation with edit_layer=[{min_edit_layer}, {max_edit_layer}] "
                )

                zs_results = {}
                fs_shuffled_results = {}
                layer_range = list(range(min_edit_layer, max_edit_layer + 1))
                for layers in itertools.product(layer_range, layer_range):
                    layers_str = "_".join([str(layer) for layer in layers])
                    zs_pred_filepath = f"{save_path_root}/preds/{args.short_model_name}_zs_{eval_identifer}_intervention_layer_{layers_str}.txt"
                    zs_results[layers_str] = _run_zs_eval(
                        args,
                        joint_fvs,
                        layers,
                        zs_pred_filepath,
                        model,
                        tokenizer,
                        model_config,
                        dataset,
                        filter_set_per_split,
                    )

                    fs_pred_filepath = f"{save_path_root}/preds/{args.short_model_name}_{eval_identifer}_shots_shuffled_{prompt_baseline}_intervention_layer_{layers_str}.txt"
                    fs_shuffled_results[layers_str] = _run_fss_eval(
                        args,
                        joint_fvs,
                        layers,
                        fs_pred_filepath,
                        model,
                        tokenizer,
                        model_config,
                        dataset,
                        filter_set_per_split,
                    )

            # Evaluate single layer
            elif isinstance(eval_edit_layer, int):
                logger.info(f"Running ZS Eval with edit_layer={eval_edit_layer}")
                zs_pred_filepath = f"{save_path_root}/preds/{args.short_model_name}_zs_{eval_identifer}_intervention_layer_{eval_edit_layer}.txt"
                zs_results = _run_zs_eval(
                    args,
                    fv,
                    eval_edit_layer,
                    zs_pred_filepath,
                    model,
                    tokenizer,
                    model_config,
                    dataset,
                    filter_set_per_split,
                )

                logger.info(f"Running {n_eval_icl_examples}-Shot Shuffled Eval")
                fs_pred_filepath = f"{save_path_root}/preds/{args.short_model_name}_{n_eval_icl_examples}_shots_shuffled_{eval_identifer}_intervention_layer_{eval_edit_layer}.txt"
                fs_shuffled_results = _run_fss_eval(
                    args,
                    fv,
                    eval_edit_layer,
                    fs_pred_filepath,
                    model,
                    tokenizer,
                    model_config,
                    dataset,
                    filter_set_per_split,
                )

            # Sweep over layers
            else:
                logger.info(f"Running sweep over layers {eval_edit_layer}")
                zs_results = {}
                fs_shuffled_results = {}
                for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
                    zs_pred_filepath = f"{save_path_root}/preds/{args.short_model_name}_zs_{eval_identifer}_intervention_layer_{edit_layer}.txt"
                    zs_results[edit_layer] = _run_zs_eval(
                        args,
                        fv,
                        edit_layer,
                        zs_pred_filepath,
                        model,
                        tokenizer,
                        model_config,
                        dataset,
                        filter_set_per_split,
                    )

                    fs_pred_filepath = f"{save_path_root}/preds/{args.short_model_name}_{n_eval_icl_examples}_shots_shuffled_{eval_identifer}_intervention_layer_{edit_layer}.txt"
                    fs_shuffled_results[edit_layer] = _run_fss_eval(
                        args,
                        fv,
                        edit_layer,
                        fs_pred_filepath,
                        model,
                        tokenizer,
                        model_config,
                        dataset,
                        filter_set_per_split,
                    )

            # Save results to files
            _protected_json_dump(zs_results_file_name, zs_results)
            _protected_json_dump(fs_shuffled_results_file_name, fs_shuffled_results)

        if compute_baseline:
            baseline_file_name = f"{save_path_root}/model_icl_baseline.json"

            if os.path.exists(baseline_file_name) and not args.force_compute_baseline:
                logger.info("Skipping baseline since file exists and force flag is off")

            else:
                baseline_file_name = make_valid_path_name(baseline_file_name)
                args.baseline_file_name = baseline_file_name
                logger.info(f"Computing model baseline results for {n_eval_icl_examples}-shots")
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

        logger.debug(f"Results saved to '{save_path_root}', saving arguments and terminating")

        # write args to file
        args_file_name = make_valid_path_name(f"{save_path_root}/{eval_identifer}_prompt_fv_eval_args.txt")
        _protected_json_dump(args_file_name, vars(args))

    except FailureException:
        pass

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if model is not None:
            del model
        _gc_clear_cache()
        time.sleep(10)

    return


if __name__ == "__main__":
    prompt_function_vector_main()
