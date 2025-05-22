import argparse
import os
import typing
from pathlib import Path

from loguru import logger

from recipe.function_vectors.cache_short_texts import WIKITEXT_NAME
from recipe.function_vectors.generate_prompts_for_dataset import LONG, PROMPT_TYPES
from recipe.function_vectors.prompt_based_function_vector import SETTINGS_BY_PROMPT_TYPE
from recipe.function_vectors.utils.prompt_utils import find_dataset_folder

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


DEFAULT_DATASETS = CURRENT_DATASETS + SKIP_DATASETS


DEFAULT_MODELS = [
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
    "google/gemma-3-1b-pt",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-3-4b-it",
]


DEFAULT_CACHED_TEXTS_DATASET = WIKITEXT_NAME


STORAGE_ROOT = os.environ.get("STORAGE_ROOT")


def check_datasets(base_path, datasets):
    any_missing = False
    datasets_path = base_path / "dataset_files"
    for dataset in datasets:
        folder = find_dataset_folder(dataset, datasets_path)
        if folder is None:
            any_missing = True
            logger.warning(f"No dataset file found for {dataset}")

    if not any_missing:
        logger.info("No dataset files missing")


def check_omitted_datasets(base_path, use_datasets, ignore_datasets):
    all_datasets = set(use_datasets) | set(ignore_datasets)
    any_omitted = False

    datasets_path = base_path / "dataset_files"
    for datasets_folder in ["abstractive", "extractive"]:
        for dataset_path in (datasets_path / datasets_folder).glob("*.json"):
            dataset_name = dataset_path.name.replace(".json", "")
            if dataset_name not in all_datasets:
                logger.info(f"Omitted dataset: {dataset_name}")
                any_omitted = True

    if not any_omitted:
        logger.info("No datasets omitted")


def check_generated_prompts(base_path, prompt_types, datasets, skip_datasets):
    any_missing = False
    prompts_path = base_path / "prompts"
    for prompt_type in prompt_types:
        prompt_suffix = SETTINGS_BY_PROMPT_TYPE[prompt_type]["saved_prompts_suffix"]
        for dataset in datasets:
            dataset_path = prompts_path / f"{dataset}_{prompt_suffix}.json"
            if not os.path.exists(dataset_path):
                if dataset in skip_datasets:
                    logger.debug(f"No {prompt_type} prompts found for {dataset}: {str(dataset_path)}")
                else:
                    any_missing = True
                    logger.warning(f"No {prompt_type} prompts found for {dataset}: {str(dataset_path)}")

    if not any_missing:
        logger.info("No prompt files missing")


def check_cached_texts(base_path, prompt_types, models, cached_texts_dataset):
    any_missing = False
    text_caches_path = base_path / "short_real_text_caches"
    for prompt_type in prompt_types:
        prompt_suffix = "_long" if prompt_type == LONG else ""
        for model in models:
            m = model[model.find("/") + 1 :].lower()
            model_text_cache_path = text_caches_path / f"{m}_{cached_texts_dataset}{prompt_suffix}.csv.gz"
            if not os.path.exists(model_text_cache_path):
                any_missing = True
                logger.warning(f"No {prompt_type} cached texts found for {model}: {str(model_text_cache_path)}")

    if not any_missing:
        logger.info("No text cache files missing")


def list_promptless_datasets(base_path, prompt_types):
    datasets_path = base_path / "dataset_files"
    prompts_path = base_path / "prompts"
    data_folders = ["abstractive", "extractive"]

    for data_folder in data_folders:
        dataset_folder = datasets_path / data_folder
        for dataset_path in dataset_folder.glob("*.json"):
            dataset_name = dataset_path.name.replace(".json", "")
            for prompt_type in prompt_types:
                prompt_suffix = SETTINGS_BY_PROMPT_TYPE[prompt_type]["saved_prompts_suffix"]
                dataset_prompts_path = prompts_path / f"{dataset_name}_{prompt_suffix}.json"
                if not dataset_prompts_path.exists():
                    logger.debug(
                        f"No prompts of type '{prompt_type}' for dataset '{dataset_name}': {str(dataset_prompts_path)}"
                    )


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
        choices=PROMPT_TYPES,
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
    parser.add_argument(
        "--cached_texts_dataset",
        help="Which datasets were real texts cached from",
        type=str,
        default=DEFAULT_CACHED_TEXTS_DATASET,
    )

    args = parser.parse_args(args)
    logger.info(str(args))

    datasets = []
    if not args.ignore_default_datasets:
        datasets.extend(DEFAULT_DATASETS)

    if args.dataset_name:
        datasets.extend(args.dataset_name)

    datasets = list(set(datasets))

    models = []
    if not args.ignore_default_models:
        models.extend(DEFAULT_MODELS)

    if args.model:
        models.extend(args.model)

    models = list(set(models))

    prompt_types = [args.prompt_type] if args.prompt_type is not None else list(PROMPT_TYPES)

    base_path = Path(STORAGE_ROOT) / "function_vectors"

    # dataset files
    check_datasets(base_path, datasets)

    check_omitted_datasets(base_path, datasets, SKIP_DATASETS)

    # generated prompts
    check_generated_prompts(base_path, prompt_types, datasets, SKIP_DATASETS)

    # real text caches
    check_cached_texts(base_path, prompt_types, models, args.cached_texts_dataset)

    # list any datasets without generated prompts
    list_promptless_datasets(base_path, prompt_types)


if __name__ == "__main__":
    main()
