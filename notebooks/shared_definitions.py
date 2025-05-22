import gzip
import json
import os
import pickle
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats.mstats import gmean

SHORT = "short"
LONG = "long"
ICL = "icl"
BOTH = "both"
ALL = "all"

RESULT_TYPES = [SHORT, LONG, ICL]
N_TOP_HEADS_VALUES = (10, 20, 40)

STORAGE_ROOT = os.environ.get("STORAGE_ROOT")
SHORT_PROMPTS_RESULTS_ROOT = f"{STORAGE_ROOT}/function_vectors/full_results_prompt_based_short"
LONG_PROMPTS_RESULTS_ROOT = f"{STORAGE_ROOT}/function_vectors/full_results_prompt_based_long"
ICL_RESULTS_ROOT = f"{STORAGE_ROOT}/function_vectors/full_icl_results"
ICL_SAME_TEST_SETS_RESULTS_ROOT = f"{STORAGE_ROOT}/function_vectors/full_icl_results_same_test_sets"

RESULT_ROOTS = {
    SHORT: SHORT_PROMPTS_RESULTS_ROOT,
    LONG: LONG_PROMPTS_RESULTS_ROOT,
    ICL: ICL_RESULTS_ROOT,
}

EQUIPROBABLE = "equiprobable"
REAL_TEXT = "real_text"
OTHER_TASK_PROMPT = "other_task_prompt"

BASELINES = [
    EQUIPROBABLE,
    REAL_TEXT,
    OTHER_TASK_PROMPT,
]
RELEVANT_BASELINES = BASELINES
ORDERED_MODELS = [
    "Llama-3.2-1B",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B",
    "Llama-3.1-8B-Instruct",
    "Llama-2-7b-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-hf",
    "Llama-2-13b-chat-hf",
    "OLMo-2-1124-7B",
    "OLMo-2-1124-7B-SFT",
    "OLMo-2-1124-7B-DPO",
    "OLMo-2-1124-7B-Instruct",
]
SCATTER_ORDERED_MODELS = [
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
]
MAIN_PAPER_PLOT_MODELS = ["Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct"]
APPENDIX_MODELS = [model for model in ORDERED_MODELS if model not in MAIN_PAPER_PLOT_MODELS and "13b" not in model]
MODEL_TO_N_LAYERS = {
    "Llama-3.2-1B": 16,
    "Llama-3.2-1B-Instruct": 16,
    "Llama-3.2-3B": 28,
    "Llama-3.2-3B-Instruct": 28,
    "Llama-3.1-8B": 32,
    "Llama-3.1-8B-Instruct": 32,
    "Llama-2-7b-chat-hf": 32,
    "Llama-2-7b-hf": 32,
    "Llama-2-13b-chat-hf": 40,
    "Llama-2-13b-hf": 40,
    "OLMo-2-1124-7B": 32,
    "OLMo-2-1124-7B-SFT": 32,
    "OLMo-2-1124-7B-DPO": 32,
    "OLMo-2-1124-7B-Instruct": 32,
}
MODEL_TO_N_LAYERS_HEADS = {
    "Llama-3.2-1B": (16, 32),
    "Llama-3.2-1B-Instruct": (16, 32),
    "Llama-3.2-3B": (28, 24),
    "Llama-3.2-3B-Instruct": (28, 24),
    "Llama-3.1-8B": (32, 32),
    "Llama-3.1-8B-Instruct": (32, 32),
    "Llama-2-7b-chat-hf": (32, 32),
    "Llama-2-7b-hf": (32, 32),
    "Llama-2-13b-chat-hf": (40, 40),
    "Llama-2-13b-hf": (40, 40),
    "OLMo-2-1124-7B": (32, 32),
    "OLMo-2-1124-7B-SFT": (32, 32),
    "OLMo-2-1124-7B-DPO": (32, 32),
    "OLMo-2-1124-7B-Instruct": (32, 32),
}
DATASET_CHANCE_ACCURACIES = {
    "adjective_v_verb_3": 1 / 3,
    "adjective_v_verb_5": 1 / 5,
    "alphabetically_first_3": 1 / 3,
    "alphabetically_first_5": 1 / 5,
    "alphabetically_last_3": 1 / 3,
    "alphabetically_last_5": 1 / 5,
    "animal_v_object_3": 1 / 3,
    "animal_v_object_5": 1 / 5,
    "antonym": 0,
    "capitalize": 1 / 26,
    "capitalize_first_letter": 1 / 26,
    "capitalize_last_letter": 1 / 26,
    "capitalize_second_letter": 1 / 26,
    "choose_first_of_3": 1 / 3,
    "choose_first_of_5": 1 / 5,
    "choose_last_of_3": 1 / 3,
    "choose_last_of_5": 1 / 5,
    "choose_middle_of_3": 1 / 3,
    "choose_middle_of_5": 1 / 5,
    "color_v_animal_3": 1 / 3,
    "color_v_animal_5": 1 / 5,
    "concept_v_object_3": 1 / 3,
    "concept_v_object_5": 1 / 5,
    "conll2003_location": 0.075,  # approximated as 1 / mean number of words (space splits) in training set
    "conll2003_organization": 0.069,  # approximated as above
    "conll2003_person": 0.059,  # approximated as above
    "country-capital": 1 / 195,  # approximated as 1 / number of countries in the world
    "country-currency": 1 / 195,
    "english-french": 0,
    "english-german": 0,
    "english-spanish": 0,
    "fruit_v_animal_3": 1 / 3,
    "fruit_v_animal_5": 1 / 5,
    "landmark-country": 1 / 195,
    "lowercase_first_letter": 1 / 26,
    "lowercase_last_letter": 1 / 26,
    "national_parks": 1 / 50,  # approximated as 1 / number of states
    "next_capital_letter": 1 / 26,
    "next_item": 0,
    "object_v_concept_3": 1 / 3,
    "object_v_concept_5": 1 / 5,
    "park-country": 1 / 195,
    "present-past": 0,
    "prev_item": 0,
    "product-company": 1 / 27,  # approximated as 1 / unique answers in the training set
    "singular-plural": 0,
    "synonym": 0,
    "verb_v_adjective_3": 1 / 3,
    "verb_v_adjective_5": 1 / 5,
    "word_length": 1 / 13,  # approximated as 1 / unique answers in the training set
}


def error_catching_decorator(f):
    def wrapped_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except json.JSONDecodeError:
            return {}

    return wrapped_f


@error_catching_decorator
def parse_per_prompt_results(md_path: Path, top_n: int = 10):
    result_path = str(md_path / "per_prompt_results.json")
    if not os.path.exists(result_path):
        if ("ignore" not in str(md_path)) and ("icl" not in str(md_path)):
            logger.warning(f"{result_path} does not exist, skipping...")
        return {}

    with open(result_path, "r") as f:
        per_prompt_results = json.load(f)

    prompts_by_accuracy = [(p, pr[0][1]) for p, pr in per_prompt_results["train"]["clean_topk"].items()]
    prompts_by_accuracy.sort(key=lambda t: t[1], reverse=True)
    selected_prompts = prompts_by_accuracy[:top_n]
    top_accuracy = selected_prompts[0][1]
    mean_accuracy = np.mean([p[1] for p in selected_prompts])
    selected_prompts = [p[0] for p in selected_prompts]

    return dict(
        top_n_prompt_acc=mean_accuracy,
        top_1_prompt_acc=top_accuracy,
        top_n_prompts=selected_prompts,
    )


@error_catching_decorator
def parse_icl_baseline_results(md_path: Path, top_n: int = 10):
    icl_baseline_names = ["model_icl_baseline.json", "model_baseline.json"]

    result_path = None
    for name in icl_baseline_names:
        rp = md_path / name
        if rp.exists():
            result_path = rp
            break

    if result_path is None:
        if not all((md_path / f"{baseline}_failure.json").exists() for baseline in BASELINES):
            if "ignore" not in str(md_path):
                logger.warning(f"{result_path} does not exist and at least one baseline did not fail, skipping...")

        return {}

    with open(result_path, "r") as f:
        baseline_results = json.load(f)

    return {f"{k}_shot_acc": kr["clean_topk"][0][1] for k, kr in baseline_results.items()}


@error_catching_decorator
def parse_prompt_based_results(md_path: Path, glob_prefix: str, output_prefix: str | None = None):
    if output_prefix is None:
        output_prefix = glob_prefix.replace("_results", "")

    if not glob_prefix.endswith("_"):
        glob_prefix += "_"

    if not output_prefix.endswith("_"):
        output_prefix += "_"

    glob_str = f"{glob_prefix}*_sweep*.json"
    result_paths = list(md_path.glob(glob_str))
    result_path_baselines = set(_get_baseline_name(p) for p in result_paths)
    if (
        (len(result_path_baselines) != len(BASELINES))
        and ("icl" not in str(md_path))
        and ("ignore" not in str(md_path))
    ):
        for missing_baseline in set(BASELINES) - result_path_baselines:
            if not os.path.exists(str(md_path / f"{missing_baseline}_failure.json")):
                logger.warning(f"Missing {missing_baseline} in {md_path} without failure file, skipping...")
                continue

    if not result_paths:
        return {}

    name_to_paths = {
        p.name.replace(f"{glob_prefix}", "")
        .replace("_layer_sweep", "")
        .replace("_mini_sweep", "")
        .replace(".json", ""): p
        for p in result_paths
    }

    latest_by_propmt_baseline = {}
    for name, path in name_to_paths.items():
        index = 0
        if name[-1].isdigit():
            name, index = name.rsplit("_", 1)
            index = int(index)

        if name not in latest_by_propmt_baseline or latest_by_propmt_baseline[name][0] < index:
            latest_by_propmt_baseline[name] = (index, path)

    output = {}
    for prompt_baseline in latest_by_propmt_baseline:
        results_path = latest_by_propmt_baseline[prompt_baseline][1]
        with open(str(results_path), "r") as f:
            res = json.load(f)

        result_by_layer = []
        for k, rk in res.items():
            rk = rk["intervention_topk"]
            if isinstance(rk, dict):
                rk = rk[""]

            if "_" in k:
                k = tuple(int(x) for x in k.split("_"))
            else:
                k = int(k)

            result_by_layer.append((k, rk[0][1]))

        # result_by_layer = [(k, rk["intervention_topk"][""][0][1]) for k, rk in res.items()]
        output[f"{output_prefix}{prompt_baseline}_by_layer_acc"] = result_by_layer
        # result_by_layer.sort(key=lambda t: t[1], reverse=True)
        output[f"{output_prefix}{prompt_baseline}_max_acc"] = result_by_layer[0][1]
        output[f"{output_prefix}{prompt_baseline}_max_acc_layer"] = result_by_layer[0][0]

    accs = [v for k, v in output.items() if k.endswith("_max_acc")]
    output[f"{output_prefix}mean_max_acc"] = np.mean(accs)
    output[f"{output_prefix}max_max_acc"] = max(accs)

    return output


def parse_fs_shuffled_results(md_path: Path):
    return parse_prompt_based_results(md_path, "fs_shuffled_results")


def parse_zs_results(md_path):
    return parse_prompt_based_results(md_path, "zs_results")


def _get_baseline_name(p: Path):
    found_baselines = [baseline_name for baseline_name in BASELINES if baseline_name in str(p)]
    if len(found_baselines) > 1:
        logger.error(f"Found multiple baselines {found_baselines} in path: {str(p)}")
        raise ValueError()

    return found_baselines[0] if len(found_baselines) else "icl"


def parse_top_heads(md_path):
    top_heads_paths = md_path.glob("*_top_heads.pt")
    results = {}
    for p in top_heads_paths:
        top_heads = torch.load(p)
        name = _get_baseline_name(p)
        results[name] = top_heads

    return results


def parse_indirect_effects(md_path):
    indirect_effect_paths = md_path.glob("*_indirect_effect.pt")
    results = {}
    for p in indirect_effect_paths:
        indirect_effect = torch.load(p)
        missing = (indirect_effect == 0).all(axis=-1).all(axis=-1).nonzero()
        if missing.numel() > 0:
            logger.warning(f"{p} has indirect effect of zeros for layers/heads for the following examples:\n{missing}")

        nans = indirect_effect.isnan().any(axis=-1).any(axis=-1).nonzero()
        if nans.numel() > 0:
            logger.warning(f"{p} contains {nans.numel()} NaN values at:\n{nans}")

        name = _get_baseline_name(p)
        mean_dims = tuple(range(indirect_effect.ndim - 2))
        results[name] = indirect_effect.mean(dim=mean_dims)
        # results[name] = indirect_effect

    return results


SKIP_MODELS = ["ignore", "gemma-3-1b-it", "gemma-3-1b-pt", "gemma-3-4b-it", "gemma-3-4b-pt", "Test-3.1-8B-Instruct"]
SKIP_DATASETS = [
    "ag_news",
    "commonsense_qa",
    "person-occupation",
    "person-instrument",
    "person-sport",
    "sentiment",
    "squad_val",
]


def parse_all_results(root_dir: str, skip_models=SKIP_MODELS, skip_datasets=SKIP_DATASETS):
    if skip_models is None:
        skip_models = []

    if skip_datasets is None:
        skip_datasets = []

    result_rows = []
    indirect_effects_by_model_and_dataset = defaultdict(dict)
    top_heads_by_model_and_dataset = defaultdict(dict)

    results_path = Path(root_dir)
    for model_path in results_path.iterdir():
        model_name = model_path.name
        if model_name in skip_models:
            logger.info(f"Skipping {model_name} as it appears in skip_models")
            continue

        for model_dataset_path in model_path.iterdir():
            dataset_name = model_dataset_path.name
            if dataset_name in skip_datasets:
                # logger.info(f"Skipping {dataset_name} as it appears in skip_datasets")
                continue

            row = dict(model=model_name, dataset=dataset_name)
            row.update(parse_per_prompt_results(model_dataset_path))
            row.update(parse_icl_baseline_results(model_dataset_path))
            row.update(parse_fs_shuffled_results(model_dataset_path))
            row.update(parse_zs_results(model_dataset_path))
            result_rows.append(row)

            indirect_effects_by_model_and_dataset[model_name][dataset_name] = parse_indirect_effects(model_dataset_path)
            top_heads_by_model_and_dataset[model_name][dataset_name] = parse_top_heads(model_dataset_path)

    result_df = pd.DataFrame(result_rows)

    return (
        result_df,
        indirect_effects_by_model_and_dataset,
        top_heads_by_model_and_dataset,
    )


RESULT_DF_CACHE_PATH = Path("./data/full_results.pkl.gz")


def load_and_combine_raw_results(
    alternative_root_dirs: typing.Dict[str, str] | None = None,
    regenerate: bool = False,
    cache_path: str | Path = RESULT_DF_CACHE_PATH,
    update_cached_result: bool = True,
):
    if alternative_root_dirs is None:
        alternative_root_dirs = {}

    if isinstance(cache_path, str):
        cache_path = Path(cache_path)

    if not regenerate and cache_path.exists():
        logger.info(f"Loading cached results from {str(cache_path)}")
        with gzip.open(cache_path, "r") as f:
            return pickle.load(f)

    dfs_by_type = {}
    indirect_effects_by_model_and_dataset = {}
    top_heads_by_model_and_dataset = {}

    for key, root_dir in RESULT_ROOTS.items():
        root_dir = alternative_root_dirs.get(key, root_dir)

        dfs_by_type[key], indirect_effects_by_model_and_dataset[key], top_heads_by_model_and_dataset[key] = (
            parse_all_results(root_dir, skip_datasets=SKIP_DATASETS)
        )

    for key, df in dfs_by_type.items():
        df["prompt_type"] = key

    result_df = pd.concat(dfs_by_type.values(), ignore_index=True)
    out = result_df, indirect_effects_by_model_and_dataset, top_heads_by_model_and_dataset

    if update_cached_result:
        logger.info(f"Saving cached results to {str(cache_path)}")
        with gzip.open(cache_path, "w") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    return out


def top_heads_from_indirect_effects_with_values(ie_data, n_top_heads=10, negative=False, largest=True):
    if negative:
        ie_data = -ie_data
    top_heads = torch.topk(ie_data.flatten(), n_top_heads, largest=largest)
    top_head_indices = top_heads.indices
    top_head_rows = top_head_indices // ie_data.shape[1]
    top_head_cols = top_head_indices % ie_data.shape[1]

    vals = top_heads.values
    if negative:
        vals = -vals

    return torch.stack((top_head_rows, top_head_cols)).T.tolist(), vals.tolist()


def compute_top_heads(
    results_df,
    ie_data,
    model: str,
    prompt_types: str | typing.List[str],
    baselines: str | typing.List[str] = RELEVANT_BASELINES,
    n_top_heads: int = 10,
    omit_below_chance_acc: bool = True,
    datasets_below_chance_acc: typing.Dict[str, typing.Set[str]] | None = None,
    negative: bool = False,
    return_mean: bool = False,
):
    if isinstance(prompt_types, str):
        prompt_types = [prompt_types]

    if prompt_types[0] == ICL:
        baselines = ICL

    if isinstance(baselines, str):
        baselines = [baselines]

    all_effects = []
    for pt in prompt_types:
        pt_data = ie_data[pt][model]
        for dataset, dataset_effects in pt_data.items():
            if omit_below_chance_acc:
                chance_acc_column = "10_shot_acc" if pt == ICL else "top_n_prompt_acc"
                if dataset not in DATASET_CHANCE_ACCURACIES:
                    logger.error(f"Dataset {dataset} not in chance accuracies")
                    raise ValueError(f"Dataset {dataset} not in chance accuracies")

                chance_acc = results_df[
                    (results_df["model"] == model)
                    & (results_df["dataset"] == dataset)
                    & (results_df["prompt_type"] == pt)
                ][chance_acc_column]

                if chance_acc.empty:
                    logger.warning(f"Model {model} dataset {dataset} prompt type {pt} not found in results_df")
                    continue

                chance_acc = chance_acc.values[0]
                if np.isnan(chance_acc):
                    logger.warning(f"Model {model} dataset {dataset} prompt type {pt} chance accuracy NaN")
                    continue

                if chance_acc < DATASET_CHANCE_ACCURACIES[dataset]:
                    if datasets_below_chance_acc is not None:
                        datasets_below_chance_acc[(model, pt)].add(dataset)
                    continue

            for baseline in baselines:
                if baseline not in dataset_effects:
                    continue

                all_effects.append(dataset_effects[baseline])
                if dataset_effects[baseline].isnan().any():
                    logger.warning(f"Found NaN in {model} {pt} {dataset} {baseline}")

    if len(all_effects) == 0:
        logger.warning(f"No effects found for {model} {prompt_types} {baselines}")
        return [], []

    mean_ie = torch.stack(all_effects).nanmean(dim=0)
    top_heads, top_effects = top_heads_from_indirect_effects_with_values(mean_ie, n_top_heads, negative)
    if return_mean:
        return top_heads, top_effects, mean_ie

    return top_heads, top_effects


def get_zs_prefix(zs: bool = True):
    return "zs" if zs else "fs_shuffled"


def _get_by_layer_col(zs: bool, n_top_heads: int, prompt_type: str, special_type: str = None):
    if special_type is None:
        special_type = "_both_all" if prompt_type != ICL else ""
    else:
        special_type = f"_{special_type}"
    return f"{get_zs_prefix(zs)}_universal{special_type}_{n_top_heads}_heads_by_layer_acc"


MAX_ACC_LAYER_DEFAULT_TUPLE = (None,) * 4


def find_max_acc_layer(
    df: pd.DataFrame,
    n_top_heads: int,
    model: str,
    prompt_type: str,
    *,
    zs: bool | None = True,
    special_type: str = None,
    geometric_mean: bool = False,
    geometric_mean_epsilon: float = 1e-5,
):
    filtered_df = df
    if zs is not None:
        relevant_cols = [_get_by_layer_col(zs, n_top_heads, prompt_type, special_type)]
        relevant_col_filter = ~filtered_df[relevant_cols[0]].isna()
    else:
        relevant_cols = [_get_by_layer_col(zs, n_top_heads, prompt_type, special_type) for zs in (False, True)]
        relevant_col_filter = ~filtered_df[relevant_cols].isna().all(axis=1)

    row_filter = relevant_col_filter & (filtered_df.model == model)
    if prompt_type == BOTH:
        row_filter &= filtered_df.prompt_type != ICL
    else:
        row_filter &= filtered_df.prompt_type == prompt_type

    filtered_df = filtered_df[row_filter].copy(deep=True)
    if len(filtered_df) == 0:
        logger.warning(
            f"No data for {model} | {prompt_type} | {relevant_cols[0] if len(relevant_cols) == 1 else relevant_cols}"
        )
        return MAX_ACC_LAYER_DEFAULT_TUPLE

    all_layers = set()
    for col in relevant_cols:
        accs = filtered_df[col]
        col_layers = accs.map(lambda layer_accs: [t[0] for t in layer_accs]).values
        col_layers = np.concatenate(col_layers)
        col_layers = list(col_layers)
        if isinstance(col_layers[0], np.ndarray):
            col_layers = [tuple(layer) for layer in col_layers]
        all_layers.update(col_layers)

    all_layers = sorted(all_layers)

    stacked_accs = np.concatenate(
        [
            np.stack(
                filtered_df[col]
                .map(lambda layer_accs: [t[1] for t in sorted(layer_accs, key=lambda x: all_layers.index(x[0]))])
                .values
            )
            for col in relevant_cols
        ]
    )
    if geometric_mean:
        stacked_accs = np.where(stacked_accs < 1e-10, geometric_mean_epsilon, stacked_accs)
        mean_accs = gmean(stacked_accs, axis=0)
    else:
        mean_accs = stacked_accs.mean(axis=0)

    # print(stacked_accs.shape, stacked_accs[:10, :10], mean_accs)
    # raise ValueError("Debugging")
    max_acc_index = mean_accs.argmax()
    max_acc_layer = all_layers[max_acc_index]
    n_layers = MODEL_TO_N_LAYERS[model]
    max_acc_layer_depth = (
        tuple(layer / n_layers for layer in max_acc_layer)
        if isinstance(max_acc_layer, (list, tuple))
        else (max_acc_layer / n_layers)
    )
    return max_acc_layer, n_layers, max_acc_layer_depth, mean_accs[max_acc_index]


def row_to_same_layer_acc(row: pd.Series, max_acc_layer_key: str, by_layer_col: str, prompt_type: str):
    max_acc_layer = row[max_acc_layer_key]
    if max_acc_layer is None:
        return None

    valid_row = ((prompt_type == BOTH) and row.prompt_type != ICL) or (row.prompt_type == prompt_type)
    if not valid_row:
        return None

    by_layer_accs = row[by_layer_col]
    if (isinstance(by_layer_accs, float) and np.isnan(by_layer_accs)) or len(by_layer_accs) == 0:
        logger.warning(f"No data for {row.model} | {row.dataset}  | {row.prompt_type} | {by_layer_col}")
        return None

    layer_acc = [t[1] for t in by_layer_accs if t[0] == max_acc_layer]
    if len(layer_acc) != 1:
        logger.warning(
            f"Unexpected data for {row.model} | {row.prompt_type} | {by_layer_col} | layer {max_acc_layer}. Expected length 1, found: {layer_acc}"
        )
        return None

    return layer_acc[0]


def row_to_universal_acc(row: pd.Series, n_top_heads: int, zs: bool, special_type: str = None):
    prompt_type = ICL if row.prompt_type == ICL else BOTH
    zs_prefix = get_zs_prefix(zs)
    pt_key = f"{prompt_type}{'' if special_type is None else '_' + special_type}"
    universal_layer_acc_key = f"{zs_prefix}_universal_{pt_key}_all_{n_top_heads}_heads_same_layer_acc"
    return row[universal_layer_acc_key] if row[universal_layer_acc_key] is not None else np.nan


def one_third_layer_rule(model: str, **kwargs):
    return round(MODEL_TO_N_LAYERS[model] / 3)


def joint_one_third_layer_rule(model: str, **kwargs):
    return (one_third_layer_rule(model), one_third_layer_rule(model))


SPECIAL_RESULT_TYPE_SHORT_NAMES = {
    "joint_intervention": "Joint",
    "icl_mean_activations": "ICL activations",
    "icl_top_heads": "ICL top heads",
    "prompt_fv_twice": "Prompt FV twice",
    "icl_fv_twice": "ICL FV twice",
    "min_abs_heads_prompt": "Prompt least imp heads",
    "min_abs_heads_icl": "ICL least imp heads",
    "bottom_prompt_heads": "Prompt bottom heads",
    "bottom_icl_heads": "ICL bottom heads",
    "instruct_model": "Instruct model",
}

SPECIAL_RESULT_TYPES = list(SPECIAL_RESULT_TYPE_SHORT_NAMES.keys())

SPECIAL_RESULT_TYPE_TO_ONE_THIRD_LAYER_RULE = {
    special_type: joint_one_third_layer_rule
    if (("joint" in special_type) or ("twice" in special_type))
    else one_third_layer_rule
    for special_type in SPECIAL_RESULT_TYPES
}
SPECIAL_RESULT_TYPE_TO_ONE_THIRD_LAYER_RULE = defaultdict(
    lambda: one_third_layer_rule, SPECIAL_RESULT_TYPE_TO_ONE_THIRD_LAYER_RULE
)


def add_same_layer_results(
    df: pd.DataFrame,
    *,
    special_type: str = None,
    geometric_mean: bool = False,
    use_both_zs_and_fs_for_top_layer: bool = False,
    include_icl: bool = True,
    n_top_head_values: typing.Sequence[int] = N_TOP_HEADS_VALUES,
    use_one_third_layer_rule: bool = False,
):
    out_df = df.copy(deep=True)
    prompt_types = [SHORT, LONG, BOTH]
    if include_icl:
        prompt_types.append(ICL)

    max_acc_layer_by_model_prompt_type_n_top_heads = defaultdict(dict)
    for model in out_df.model.unique():
        for prompt_type in prompt_types:
            for n_top_heads in n_top_head_values:
                if use_one_third_layer_rule:
                    layer = SPECIAL_RESULT_TYPE_TO_ONE_THIRD_LAYER_RULE[special_type](
                        model=model, prompt_type=prompt_type, n_top_heads=n_top_heads, special_type=special_type
                    )
                    max_acc_layer_output = (
                        layer,
                        MODEL_TO_N_LAYERS[model],
                        tuple(lay / MODEL_TO_N_LAYERS[model] for lay in layer)
                        if isinstance(layer, (list, tuple))
                        else layer / MODEL_TO_N_LAYERS[model],
                        None,
                    )

                else:
                    max_acc_layer_output = find_max_acc_layer(
                        out_df,
                        n_top_heads,
                        model,
                        prompt_type=prompt_type,
                        zs=None if use_both_zs_and_fs_for_top_layer else True,
                        special_type=special_type,
                        geometric_mean=geometric_mean,
                    )

                max_acc_layer_by_model_prompt_type_n_top_heads[model][(prompt_type, n_top_heads)] = max_acc_layer_output

    for model, layer_info in max_acc_layer_by_model_prompt_type_n_top_heads.items():
        for k, v in layer_info.items():
            if len(v) < 3:
                print(model, k, v)

    # print(max_acc_layer_by_model_prompt_type_n_top_heads)

    for prompt_type in prompt_types:
        pt_key = f"{prompt_type}{'' if special_type is None else '_' + special_type}"
        for n_top_heads in n_top_head_values:
            for zs in (False, True):
                by_layer_col = _get_by_layer_col(zs, n_top_heads, prompt_type=prompt_type, special_type=special_type)
                zs_prefix = get_zs_prefix(zs)
                max_acc_layer_key = f"{zs_prefix}_{pt_key}_{n_top_heads}_max_acc_layer"
                max_acc_layer_depth_key = f"{zs_prefix}_{pt_key}_{n_top_heads}_max_acc_layer_depth"
                out_df = out_df.assign(
                    **{
                        max_acc_layer_key: out_df.model.map(
                            lambda model: max_acc_layer_by_model_prompt_type_n_top_heads[model].get(
                                (prompt_type, n_top_heads), MAX_ACC_LAYER_DEFAULT_TUPLE
                            )[0]
                        ),
                        "n_layers": out_df.model.map(
                            lambda model: max_acc_layer_by_model_prompt_type_n_top_heads[model].get(
                                (prompt_type, n_top_heads), MAX_ACC_LAYER_DEFAULT_TUPLE
                            )[1]
                        ),
                        max_acc_layer_depth_key: out_df.model.map(
                            lambda model: max_acc_layer_by_model_prompt_type_n_top_heads[model].get(
                                (prompt_type, n_top_heads), MAX_ACC_LAYER_DEFAULT_TUPLE
                            )[2]
                        ),
                    }
                )

                universal_layer_acc_key = f"{zs_prefix}_universal_{pt_key}_all_{n_top_heads}_heads_same_layer_acc"

                out_df = out_df.assign(
                    **{
                        universal_layer_acc_key: out_df.apply(
                            row_to_same_layer_acc, axis=1, args=(max_acc_layer_key, by_layer_col, prompt_type)
                        )
                    }
                )

    for n_top_heads in n_top_head_values:
        for zs in (False, True):
            zs_prefix = get_zs_prefix(zs)
            univ = f"universal{'' if special_type is None else '_' + special_type}"
            universal_layer_acc_key = f"{zs_prefix}_{n_top_heads}_heads_{univ}_FV_acc"
            out_df = out_df.assign(
                **{universal_layer_acc_key: out_df.apply(row_to_universal_acc, axis=1, args=(n_top_heads, zs))}
            )

    return out_df
