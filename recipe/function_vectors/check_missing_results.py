import argparse
import os
import re
import typing
from collections import defaultdict
from pathlib import Path

from loguru import logger

from recipe.function_vectors.utils.model_utils import extract_model_size

CURRENT_DATASETS = [
    "adjective_v_verb_3",
    "adjective_v_verb_5",
    "alphabetically_first_3",
    "alphabetically_first_5",
    "alphabetically_last_3",
    "alphabetically_last_5",
    "animal_v_object_3",
    "animal_v_object_5",
    "antonym",
    "capitalize",
    "capitalize_first_letter",
    "capitalize_last_letter",
    "capitalize_second_letter",
    "choose_first_of_3",
    "choose_first_of_5",
    "choose_last_of_3",
    "choose_last_of_5",
    "choose_middle_of_3",
    "choose_middle_of_5",
    "color_v_animal_3",
    "color_v_animal_5",
    "concept_v_object_3",
    "concept_v_object_5",
    "conll2003_location",
    "conll2003_organization",
    "conll2003_person",
    "country-capital",
    "country-currency",
    "english-german",
    "english-french",
    "english-spanish",
    "fruit_v_animal_3",
    "fruit_v_animal_5",
    "landmark-country",
    "lowercase_first_letter",
    "lowercase_last_letter",
    "national_parks",
    "next_capital_letter",
    "next_item",
    "object_v_concept_3",
    "object_v_concept_5",
    "park-country",
    "present-past",
    "prev_item",
    "product-company",
    "singular-plural",
    "synonym",
    "verb_v_adjective_3",
    "verb_v_adjective_5",
    "word_length",
]

SKIP_DATASETS = [
    "ag_news",
    "commonsense_qa",
    "person-instrument",
    "person-occupation",
    "person-sport",
    "sentiment",
    "squad_val",
]


ALL_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-SFT",
    "allenai/OLMo-2-1124-7B-DPO",
    "allenai/OLMo-2-1124-7B-Instruct",
    # "google/gemma-3-1b-pt",
    # "google/gemma-3-1b-it",
    # "google/gemma-3-4b-pt",
    # "google/gemma-3-4b-it",
]

ALL_MODELS = [m_[m_.rfind("/") + 1 :] for m_ in ALL_MODELS]


IN_PROGRESS_MODELS = [m_ for m_ in ALL_MODELS if "OLMo" in m_]

UNIVERSAL_ABLATION_MODELS = [m_ for m_ in ALL_MODELS if any(part in m_ for part in ("3.2-1B", "3.2-3B"))]

EXPECTED_N_HEADS_BY_MODEL = {}
for m_ in ALL_MODELS:
    model_size = extract_model_size(m_)
    expected_heads = [10, 20]
    if model_size >= 7:
        expected_heads.append(40)

    EXPECTED_N_HEADS_BY_MODEL[m_] = expected_heads


SHORT = "short"
LONG = "long"
ICL = "icl"
ALL_PROMPT_TYPES = [SHORT, LONG, ICL]
BOTH = "both"
ALL = "all"

STORAGE_ROOT = os.environ.get("STORAGE_ROOT")
SHORT_PROMPTS_RESULTS_ROOT = f"{STORAGE_ROOT}/function_vectors/full_results_prompt_based_short"
LONG_PROMPTS_RESULTS_ROOT = f"{STORAGE_ROOT}/function_vectors/full_results_prompt_based_long"
ICL_RESULTS_ROOT = f"{STORAGE_ROOT}/function_vectors/full_icl_results"

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

UNIVERSAL_FV = "universal"
JOINT_INTERVENTION = "joint_intervention"
ICL_HEADS = "icl_heads"
ICL_ACTIVATIONS = "icl_mean_activations"

UNIVERSAL_NAME_TO_RESULT_MARKER = {
    "both_all": UNIVERSAL_FV,
    JOINT_INTERVENTION: JOINT_INTERVENTION,
    ICL_HEADS: ICL_HEADS,
    ICL_ACTIVATIONS: ICL_ACTIVATIONS,
}
ALL_UNIVERSAL_NAMES = set(UNIVERSAL_NAME_TO_RESULT_MARKER.values())


def _get_baseline_name(p: Path):
    found_baselines = [baseline_name for baseline_name in BASELINES if baseline_name in str(p)]
    if len(found_baselines) > 1:
        logger.error(f"Found multiple baselines {found_baselines} in path: {str(p)}")
        raise ValueError()

    return found_baselines[0] if len(found_baselines) else "icl"


def _check_prompt_based_results(md_path: Path, glob_prefix: str, output_prefix: str | None = None):
    if output_prefix is None:
        output_prefix = glob_prefix.replace("_results", "")

    if not glob_prefix.endswith("_"):
        glob_prefix += "_"

    if not output_prefix.endswith("_"):
        output_prefix += "_"

    glob_str = f"{glob_prefix}*_layer_sweep*.json"
    result_paths = list(md_path.glob(glob_str))
    result_path_baselines = set(_get_baseline_name(p) for p in result_paths)
    missing_baselines = set(BASELINES) - result_path_baselines
    if (
        (len(result_path_baselines) != len(BASELINES))
        and ("icl" not in str(md_path))
        and ("ignore" not in str(md_path))
    ):
        for missing_baseline in missing_baselines:
            if not os.path.exists(str(md_path / f"{missing_baseline}_failure.json")):
                logger.warning(f"Missing {missing_baseline} in {md_path} without failure file, skipping...")
                continue

    return result_path_baselines - missing_baselines


def check_fs_shuffled_results(md_path: Path):
    return _check_prompt_based_results(md_path, "fs_shuffled_results")


def check_zs_results(md_path: Path):
    return _check_prompt_based_results(md_path, "zs_results")


def _find_n_heads(s: str) -> int | None:
    match = re.search(r"_(\d+)_", s)
    if match:
        return int(match.group(1))

    return None


def _check_universal_prompt_based_results(
    model_name: str, md_path: Path, glob_prefix: str, output_prefix: str | None = None
):
    icl_mode = "icl" in str(md_path)
    
    if output_prefix is None:
        output_prefix = glob_prefix.replace("_results", "")

    if not glob_prefix.endswith("_"):
        glob_prefix += "_"

    if not output_prefix.endswith("_"):
        output_prefix += "_"

    glob_str = f"{glob_prefix}universal_*_sweep*.json"
    result_paths = list(md_path.glob(glob_str))

    results_by_n_heads = defaultdict(set)

    for result_path in result_paths:
        name = result_path.name
        n_heads = _find_n_heads(name)
        if n_heads is None:
            logger.warning(f"Could not find n_heads in {md_path / name}, skipping...")

        if icl_mode:
            if UNIVERSAL_FV in name:
                results_by_n_heads[n_heads].add(UNIVERSAL_FV)

        else:
            for key, value in UNIVERSAL_NAME_TO_RESULT_MARKER.items():
                if key in name:
                    results_by_n_heads[n_heads].add(value)
                    break

    expected_n_heads = EXPECTED_N_HEADS_BY_MODEL.get(model_name)
    if expected_n_heads is None:
        logger.warning(f"Could not find expected n_heads for {model_name}")
        return

    missing_by_n_heads = {}

    for n_heads in expected_n_heads:
        if n_heads not in results_by_n_heads:
            logger.warning(f"Missing {n_heads} in {md_path}")
            missing_by_n_heads[n_heads] = "everything"
            continue

        res = results_by_n_heads[n_heads]
        if UNIVERSAL_FV not in res:
            logger.warning(f"Missing {UNIVERSAL_FV} in {md_path}, skipping")
            missing_by_n_heads[n_heads] = UNIVERSAL_FV
            continue

        if (not icl_mode) and (model_name in UNIVERSAL_ABLATION_MODELS):
            if len(res) != len(ALL_UNIVERSAL_NAMES):
                missing_universals = ALL_UNIVERSAL_NAMES - res
                logger.warning(f"Missing {missing_universals} in {md_path}")
                missing_by_n_heads[n_heads] = missing_universals
            # else:
            # logger.debug(f"Found all {ALL_UNIVERSAL_NAMES} in {md_path}")

    return missing_by_n_heads


def check_universal_fs_shuffled_results(model_name: str, md_path: Path):
    return _check_universal_prompt_based_results(model_name, md_path, "fs_shuffled_results")


def check_universal_zs_results(model_name: str, md_path: Path):
    return _check_universal_prompt_based_results(model_name, md_path, "zs_results")


def main(args: typing.Optional[typing.List[str]] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to be checked",
        nargs="+",
        type=str,
        action="extend",
    )
    parser.add_argument(
        "--model",
        help="Name of the model to be checked",
        nargs="+",
        type=str,
        action="extend",
    )
    parser.add_argument(
        "--prompt_type",
        help="Which prompts to use",
        type=str,
        default=None,
        choices=ALL_PROMPT_TYPES,
    )
    parser.add_argument(
        "--ignore_default_datasets",
        help="Ignore the pre-defined set of datasets",
        action="store_true",
    )
    parser.add_argument(
        "--ignore_default_models",
        help="Ignore the pre-defined set of models",
        action="store_true",
    )

    args = parser.parse_args(args)
    logger.info(str(args))

    datasets = []
    if not args.ignore_default_datasets:
        datasets.extend(CURRENT_DATASETS)

    if args.dataset_name:
        datasets.extend(args.dataset_name)

    datasets = list(set(datasets))

    models = []
    if not args.ignore_default_models:
        models.extend(ALL_MODELS)

    if args.model:
        models.extend(args.model)

    models = list(set(models))

    prompt_types = [args.prompt_type] if args.prompt_type is not None else list(ALL_PROMPT_TYPES)

    missing_universal_by_model_dataset = defaultdict(lambda: defaultdict(dict))

    for prompt_type in prompt_types:
        logger.info(f"Processing prompt type: {prompt_type}")
        results_path = Path(RESULT_ROOTS[prompt_type])

        for model_path in results_path.iterdir():
            model_name = model_path.name
            if model_name in IN_PROGRESS_MODELS:
                logger.info(f"Skipping {model_name} as it is in progress")
                continue

            if model_name not in models:
                logger.info(f"Skipping {model_name} as it is not recognized")
                continue

            for model_dataset_path in model_path.iterdir():
                dataset_name = model_dataset_path.name
                if dataset_name in SKIP_DATASETS:
                    logger.info(f"Skipping {dataset_name} as it appears in skip_datasets")
                    continue

                if dataset_name not in datasets:
                    logger.info(f"Skipping {dataset_name} as it is not recognized")
                    continue

                if prompt_type != ICL:
                    found_baselines = check_zs_results(model_dataset_path) | check_fs_shuffled_results(model_dataset_path)

                    if not found_baselines:
                        logger.debug(f"No baselines found for {model_name} on {dataset_name}, skipping...")
                        continue

                else:
                    results_files = list(model_dataset_path.glob("*results_layer_sweep.json"))
                    if not results_files:
                        logger.debug(f"No results found for {model_name} on {dataset_name}, skipping...")
                        continue


                missing_universal_by_model_dataset[model_name][dataset_name]["fs_shuffled"] = (
                    check_universal_fs_shuffled_results(model_name, model_dataset_path)
                )

                missing_universal_by_model_dataset[model_name][dataset_name]["zs"] = check_universal_zs_results(
                    model_name, model_dataset_path
                )

    for model_name, datasets in missing_universal_by_model_dataset.items():
        model_missing_lines = []

        for dataset_name, missing_results in datasets.items():
            dataset_listed = False
            for result_type, missing in missing_results.items():
                if not missing:
                    continue

                if not dataset_listed:
                    model_missing_lines.append(f"  {dataset_name}:")
                    dataset_listed = True

                for n_heads, n_heads_missing in missing.items():
                    model_missing_lines.append(f"    {result_type}, {n_heads}: {n_heads_missing}")

        if len(model_missing_lines):
            model_missing_lines.insert(0, f"\n{model_name}:")
            print("\n".join(model_missing_lines))


if __name__ == "__main__":
    main()
