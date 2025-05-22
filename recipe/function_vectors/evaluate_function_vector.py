import argparse
import json
import os
import platform
import typing
from pathlib import Path

import numpy as np
import torch
from git import Repo
from loguru import logger

from recipe.function_vectors.utils.eval_utils import (
    compute_dataset_baseline,
    make_valid_path_name,
    n_shot_eval,
    n_shot_eval_no_intervention,
)
from recipe.function_vectors.utils.extract_utils import (
    compute_function_vector,
    compute_universal_function_vector_top_heads_from_file,
    get_mean_head_activations,
)
from recipe.function_vectors.utils.model_utils import (
    load_gpt_model_and_tokenizer,
    set_seed,
)
from recipe.function_vectors.utils.prompt_utils import (
    load_dataset,
)

from .compute_indirect_effect import compute_indirect_effect

STORAGE_ROOT = os.environ.get("STORAGE_ROOT")


def _get_git_commit_hex():
    p = Path(__file__).absolute()
    # p = Path(".").absolute()

    while not list(p.glob(".git")):
        p = p.parent
        if str(p) == "/":
            raise ValueError("Git repo not found in parents")

    repo = Repo(p)
    return repo.head.commit.hexsha


def evaluate_fv_main(extra_args: typing.Optional[typing.List[str]] = None):
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
        default=f"{STORAGE_ROOT}/function_vectors/full_icl_results_same_test_sets",
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
        help="Path to file containing indirect_effect scores for the specified task",
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
    parser.add_argument(
        "--n_shots",
        help="Number of shots in each in-context prompt",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--total_mean_activation_examples",
        help="Number total examples (split evenly between all prompts) for mean activations",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--total_indirect_effect_examples",
        help="Number of baseline prompts to average over for indirect effect for each prompt",
        type=int,
        required=False,
        default=25,
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
        "--top_heads_dir",
        help="Directory to load top heads from",
        type=str,
        required=False,
        default=f"{STORAGE_ROOT}/function_vectors/full_results_top_heads",
    )
    parser.add_argument(
        "--force_fs_layer_sweep",
        help="Flag for whether to force running evaluations even if the files exist",
        action="store_true",
    )
    parser.add_argument(
        "--fs_layer_sweep_only",
        help="Flag for whether to only run the layer sweep evaluations",
        action="store_true",
    )
    parser.add_argument(
        "--force_evaluation",
        help="Flag for whether to force running evaluations even if the files exist",
        action="store_true",
    )
    parser.add_argument(
        "--force_compute_baseline",
        help="Flag for whether to force computing baselines even if the files exist",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for evaluation",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--use_old_paths",
        help="Flag for whether to use the older paths",
        action="store_true",
    )

    args = parser.parse_args(extra_args)

    dataset_name = args.dataset_name
    model_name = args.model_name
    args.short_model_name = args.model_name[args.model_name.rfind("/") + 1 :]
    root_data_dir = args.root_data_dir
    save_path_suffix = args.save_path_suffix if args.save_path_suffix is not None else args.short_model_name
    save_path_root = f"{args.save_path_root}/{save_path_suffix}/{dataset_name}"

    use_old_paths = args.use_old_paths
    if use_old_paths:
        save_path_root = save_path_root.replace("full_icl_results_same_test_sets", "full_icl_results")

    ie_path_root = f"{args.ie_path_root}/{dataset_name}" if args.ie_path_root else save_path_root
    seed = args.seed
    device = args.device
    mean_activations_path = args.mean_activations_path
    indirect_effect_path = args.indirect_effect_path
    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer

    test_split = float(args.test_split)
    n_shots = args.n_shots
    total_mean_activation_examples = args.total_mean_activation_examples
    total_indirect_effect_examples = args.total_indirect_effect_examples

    prefixes = args.prefixes
    separators = args.separators
    compute_baseline = args.compute_baseline

    generate_str = args.generate_str
    metric = args.metric
    universal_set = args.universal_set

    is_conll = "conll2003" in dataset_name
    if args.batch_size == 0:
        if any(name in model_name for name in ("Llama-3.1-8B", "OLMo-2-1124-7B")):
            args.batch_size = 1
        elif "Llama-2" in model_name and is_conll:
            args.batch_size = 1
        else:
            # we can probably afford more, but it's not crucial
            args.batch_size = 5

    args.commit = _get_git_commit_hex()
    args.slurm_job_id = os.environ.get("SLURM_JOB_ID")
    args.slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    args.slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    args.hostname = platform.node()
    logger.debug(str(args))

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    logger.info("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, revision=args.revision)

    if args.edit_layer == -1:  # sweep over all layers if edit_layer=-1
        eval_edit_layer = [0, model_config["n_layers"]]

    # Load the dataset
    logger.info("Loading Dataset")
    set_seed(seed)
    if not os.path.exists(root_data_dir):
        raise ValueError(f"Dataset Directory Not Found: {root_data_dir}")

    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)
    logger.debug(f"Loaded dataset {dataset_name} with the following sizes: { {k: len(d) for k, d in dataset.items()} }")

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    logger.info(f"Filtering Dataset via {n_shots}-shot Eval")
    # 1. Compute Model 10-shot Baseline & 2. Filter test set to cases where model gets it correct

    force_fs_layer_sweep = args.force_fs_layer_sweep
    fs_layer_sweep_only = args.fs_layer_sweep_only

    fs_results_file_name = f"{save_path_root}/fs_results_layer_sweep.json"
    logger.info(fs_results_file_name)
    if os.path.exists(fs_results_file_name) and not force_fs_layer_sweep:
        with open(fs_results_file_name, "r") as indata:
            fs_results = json.load(indata)
        key = "score" if generate_str else "clean_rank_list"
        target_val = 1 if generate_str else 0

        if "test" in fs_results:
            filter_set_test = np.where(np.array(fs_results["test"][key]) == target_val)[0]
            filter_set_valid = np.where(np.array(fs_results["valid"][key]) == target_val)[0]
        else:
            filter_set_test = np.where(np.array(fs_results[key]) == target_val)[0]
            filter_set_valid = None

    else:
        if generate_str:
            set_seed(seed + 42)
            fs_results_validation = n_shot_eval_no_intervention(
                dataset=dataset,
                n_shots=n_shots,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                compute_ppl=False,
                generate_str=True,
                metric=metric,
                test_split="valid",
                prefixes=prefixes,
                separators=separators,
                batch_size=args.batch_size,
            )
            filter_set_valid = np.where(np.array(fs_results_validation["score"]) == 1)[0]
            set_seed(seed)
            fs_results = n_shot_eval_no_intervention(
                dataset=dataset,
                n_shots=n_shots,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                compute_ppl=False,
                generate_str=True,
                metric=metric,
                prefixes=prefixes,
                separators=separators,
                batch_size=args.batch_size,
            )
            filter_set_test = np.where(np.array(fs_results["score"]) == 1)[0]
        else:
            set_seed(seed + 42)
            fs_results_validation = n_shot_eval_no_intervention(
                dataset=dataset,
                n_shots=n_shots,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                compute_ppl=True,
                test_split="valid",
                prefixes=prefixes,
                separators=separators,
                batch_size=args.batch_size,
            )
            filter_set_valid = np.where(np.array(fs_results_validation["clean_rank_list"]) == 0)[0]
            set_seed(seed)
            fs_results = n_shot_eval_no_intervention(
                dataset=dataset,
                n_shots=n_shots,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                compute_ppl=True,
                prefixes=prefixes,
                separators=separators,
                batch_size=args.batch_size,
            )
            filter_set_test = np.where(np.array(fs_results["clean_rank_list"]) == 0)[0]

        args.fs_results_file_name = fs_results_file_name
        with open(fs_results_file_name, "w") as results_file:
            json.dump(dict(test=fs_results, valid=fs_results_validation), results_file, indent=2)

    if fs_layer_sweep_only:
        logger.info("Skipping evaluation since --fs_layer_sweep_only flag was set")
        return args

    set_seed(seed)
    # Load or Re-Compute mean_head_activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f"{ie_path_root}/{dataset_name}_mean_head_activations.pt"):
        mean_activations_path = f"{ie_path_root}/{dataset_name}_mean_head_activations.pt"
        mean_activations = torch.load(mean_activations_path)
    else:
        if filter_set_valid is None or len(filter_set_valid) == 0:
            logger.error("Found 0 passing validation split examples, we require at least one example, aborting...")
            return args

        logger.info(f"Computing Mean Activations using {total_mean_activation_examples} examples")
        set_seed(seed)
        mean_activations = get_mean_head_activations(
            dataset,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            n_icl_examples=n_shots,
            N_TRIALS=total_mean_activation_examples,
            prefixes=prefixes,
            separators=separators,
            filter_set=filter_set_valid,
        )
        args.mean_activations_path = f"{save_path_root}/{dataset_name}_mean_head_activations.pt"
        torch.save(mean_activations, args.mean_activations_path)

    # Load or Re-Compute indirect_effect values
    if indirect_effect_path is not None and os.path.exists(indirect_effect_path):
        indirect_effect = torch.load(indirect_effect_path)
    elif indirect_effect_path is None and os.path.exists(f"{ie_path_root}/{dataset_name}_indirect_effect.pt"):
        indirect_effect_path = f"{ie_path_root}/{dataset_name}_indirect_effect.pt"
        indirect_effect = torch.load(indirect_effect_path)
    elif not universal_set:  # Only compute indirect effects if we need to
        if filter_set_valid is None or len(filter_set_valid) == 0:
            logger.error("Found 0 passing validation split examples, we require at least one example, aborting...")
            return args

        logger.info(f"Computing Indirect Effects using {total_indirect_effect_examples} examples")
        set_seed(seed)
        args.partial_indirect_effect_path = f"{save_path_root}/{dataset_name}_indirect_effect.pt.partial"
        indirect_effect = compute_indirect_effect(
            dataset,
            mean_activations,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            n_shots=n_shots,
            n_trials=total_indirect_effect_examples,
            last_token_only=True,
            prefixes=prefixes,
            separators=separators,
            filter_set=filter_set_valid,
            partial_path=args.partial_indirect_effect_path,
        )
        args.indirect_effect_path = f"{save_path_root}/{dataset_name}_indirect_effect.pt"
        torch.save(indirect_effect, args.indirect_effect_path)

    # Compute Function Vector
    if universal_set:
        args.fv_path = f"{save_path_root}/{dataset_name}_{n_top_heads}_universal_fv.pt"
        if os.path.exists(args.fv_path):
            logger.info(f"Loading universal function vector from {args.fv_path}")
            fv = torch.load(args.fv_path)
        else:
            if use_old_paths:
                top_heads_suffix = "icl_top_heads"
            else:
                top_heads_suffix = "icl_same_test_sets_top_heads"

            args.top_heads_path = f"{args.top_heads_dir}/{args.short_model_name}_{top_heads_suffix}.json"

            logger.info(
                f"Loading top heads from {args.top_heads_path} to compute universal function vector and saving to {args.fv_path}"
            )
            fv, top_heads = compute_universal_function_vector_top_heads_from_file(
                mean_activations,
                model,
                model_config=model_config,
                top_heads_path=args.top_heads_path,
                n_top_heads=n_top_heads,
            )
            torch.save(fv, args.fv_path)
    else:
        args.fv_path = f"{save_path_root}/{dataset_name}_fv.pt"
        args.top_heads_path = f"{save_path_root}/{dataset_name}_top_heads.pt"
        if os.path.exists(args.fv_path):
            fv = torch.load(args.fv_path)
        else:
            fv, top_heads = compute_function_vector(
                mean_activations,
                indirect_effect,
                model,
                model_config=model_config,
                n_top_heads=n_top_heads,
            )
            torch.save(fv, args.fv_path)
            torch.save(top_heads, args.top_heads_path)

    # Run Evaluation

    zs_results_file_suffix = (
        f"_editlayer_{eval_edit_layer}.json" if isinstance(eval_edit_layer, int) else "_layer_sweep.json"
    )
    if universal_set:
        zs_results_file_suffix = f"_universal_{n_top_heads}_heads" + zs_results_file_suffix
    zs_results_file_name = make_valid_path_name(f"{save_path_root}/zs_results" + zs_results_file_suffix)
    args.zs_results_file_name = zs_results_file_name

    fs_shuffled_results_file_suffix = (
        f"_editlayer_{eval_edit_layer}.json" if isinstance(eval_edit_layer, int) else "_layer_sweep.json"
    )
    if universal_set:
        fs_shuffled_results_file_suffix = f"_universal_{n_top_heads}_heads" + fs_shuffled_results_file_suffix
    fs_shuffled_results_file_name = make_valid_path_name(
        f"{save_path_root}/fs_shuffled_results" + fs_shuffled_results_file_suffix
    )
    args.fs_shuffled_results_file_name = fs_shuffled_results_file_name

    if (
        os.path.exists(zs_results_file_name)
        and os.path.exists(fs_shuffled_results_file_name)
        and not args.force_evaluation
    ):
        logger.warning("Skipping evaluations since both files exist and flag was not set")

    else:
        if isinstance(eval_edit_layer, int):
            logger.info(f"Running ZS Eval with edit_layer={eval_edit_layer}")
            set_seed(seed)
            if generate_str:
                pred_filepath = f"{save_path_root}/preds/{model_config['name_or_path'].replace('/', '_')}_ZS_intervention_layer{eval_edit_layer}.txt"
                zs_results = n_shot_eval(
                    dataset=dataset,
                    fv_vector=fv,
                    edit_layer=eval_edit_layer,
                    n_shots=0,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    filter_set=filter_set_test,
                    generate_str=generate_str,
                    metric=metric,
                    pred_filepath=pred_filepath,
                    prefixes=prefixes,
                    separators=separators,
                )
            else:
                zs_results = n_shot_eval(
                    dataset=dataset,
                    fv_vector=fv,
                    edit_layer=eval_edit_layer,
                    n_shots=0,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    filter_set=filter_set_test,
                    prefixes=prefixes,
                    separators=separators,
                )

            logger.info(f"Running {n_shots}-Shot Shuffled Eval")
            set_seed(seed)
            if generate_str:
                pred_filepath = f"{save_path_root}/preds/{model_config['name_or_path'].replace('/', '_')}_{n_shots}shots_shuffled_intervention_layer{eval_edit_layer}.txt"
                fs_shuffled_results = n_shot_eval(
                    dataset=dataset,
                    fv_vector=fv,
                    edit_layer=eval_edit_layer,
                    n_shots=n_shots,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    filter_set=filter_set_test,
                    shuffle_labels=True,
                    generate_str=generate_str,
                    metric=metric,
                    pred_filepath=pred_filepath,
                    prefixes=prefixes,
                    separators=separators,
                )
            else:
                fs_shuffled_results = n_shot_eval(
                    dataset=dataset,
                    fv_vector=fv,
                    edit_layer=eval_edit_layer,
                    n_shots=n_shots,
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    filter_set=filter_set_test,
                    shuffle_labels=True,
                    prefixes=prefixes,
                    separators=separators,
                )

        else:
            logger.info(f"Running sweep over layers {eval_edit_layer}")
            zs_results = {}
            fs_shuffled_results = {}
            for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
                set_seed(seed)
                if generate_str:
                    zs_results[edit_layer] = n_shot_eval(
                        dataset=dataset,
                        fv_vector=fv,
                        edit_layer=edit_layer,
                        n_shots=0,
                        model=model,
                        model_config=model_config,
                        tokenizer=tokenizer,
                        filter_set=filter_set_test,
                        generate_str=generate_str,
                        metric=metric,
                        prefixes=prefixes,
                        separators=separators,
                    )
                else:
                    zs_results[edit_layer] = n_shot_eval(
                        dataset=dataset,
                        fv_vector=fv,
                        edit_layer=edit_layer,
                        n_shots=0,
                        prefixes=prefixes,
                        separators=separators,
                        model=model,
                        model_config=model_config,
                        tokenizer=tokenizer,
                        filter_set=filter_set_test,
                    )
                set_seed(seed)
                if generate_str:
                    fs_shuffled_results[edit_layer] = n_shot_eval(
                        dataset=dataset,
                        fv_vector=fv,
                        edit_layer=edit_layer,
                        n_shots=n_shots,
                        model=model,
                        model_config=model_config,
                        tokenizer=tokenizer,
                        filter_set=filter_set_test,
                        generate_str=generate_str,
                        metric=metric,
                        shuffle_labels=True,
                        prefixes=prefixes,
                        separators=separators,
                    )
                else:
                    fs_shuffled_results[edit_layer] = n_shot_eval(
                        dataset=dataset,
                        fv_vector=fv,
                        edit_layer=edit_layer,
                        n_shots=n_shots,
                        model=model,
                        model_config=model_config,
                        tokenizer=tokenizer,
                        filter_set=filter_set_test,
                        shuffle_labels=True,
                        prefixes=prefixes,
                        separators=separators,
                    )

        # Save results to files
        with open(zs_results_file_name, "w") as results_file:
            json.dump(zs_results, results_file, indent=2)

        with open(fs_shuffled_results_file_name, "w") as results_file:
            json.dump(fs_shuffled_results, results_file, indent=2)

    if compute_baseline:
        baseline_file_name = f"{save_path_root}/model_baseline.json"
        args.baseline_file_name = baseline_file_name

        if os.path.exists(baseline_file_name) and not args.force_compute_baseline:
            logger.warning("Skipping baseline since file exists and force flag is off")

        else:
            logger.info(f"Computing model baseline results for {n_shots}-shots")
            baseline_file_name = make_valid_path_name(f"{save_path_root}/model_baseline.json")
            args.baseline_file_name = baseline_file_name
            baseline_results = compute_dataset_baseline(
                dataset,
                model,
                model_config,
                tokenizer,
                n_shots=n_shots,
                seed=seed,
                prefixes=prefixes,
                separators=separators,
            )

            with open(baseline_file_name, "w") as results_file:
                json.dump(baseline_results, results_file, indent=2)

    logger.debug(f"Results saved to '{save_path_root}', saving arguments and terminating")

    # Write args to file
    args_file_name = make_valid_path_name(f"{save_path_root}/fv_eval_args.txt")
    with open(args_file_name, "w") as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    return args


if __name__ == "__main__":
    evaluate_fv_main()
