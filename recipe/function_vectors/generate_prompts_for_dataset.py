import argparse
import json
import os
import re
import time
import typing

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm

from recipe.function_vectors.utils.eval_utils import make_valid_path_name
from recipe.function_vectors.utils.model_utils import set_seed
from recipe.function_vectors.utils.prompt_utils import load_dataset

LLAMA_31_405B = "Llama-3.1-405B"

MODEL_URLS = {
    LLAMA_31_405B: 'https://REDACTED-FOR-REVIEW'
}


# <|begin_of_text|><|start_header_id|>system<|end_header_id|>Your task is to embody the following persona and answer the user's questions as though you, the assistant, were the person described below:
# {system_persona}
# <|eot_id|>

RESPONSE_START = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
PROMPTS_HEADER = "# Task Prompts"

SHORT_PROMPT_GENERATION_PROMPT_TEMPLATE = """<|start_header_id|>user<|end_header_id|># Instructions
You are powerful model helping write prompts to help smaller models perform tasks better.
Below, you will be given a set of input-output pairs for a particular undescribed task. 
First, please study the examples to deduce what the task is, and describe your thinking under the header "# Task Deduction".
Next, please write 10 prompts that might help a smaller model perform this task. The prompts should be:
1. Short, up to 10 words.
2. Informative about what the task is.
3. Not repetitive with each other.
Please write your prompts under the header "# Task Prompts". 

# Task examples
{task_examples}

Now, think step by step and follow the instructions above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

LONG_PROMPT_GENERATION_PROMPT_TEMPLATE = """<|start_header_id|>user<|end_header_id|># Instructions
You are powerful model helping write prompts to help smaller models perform tasks better.
Below, you will be given a set of input-output pairs for a particular undescribed task. 
First, please study the examples to deduce what the task is, and describe your thinking under the header "# Task Deduction".
Next, please write 10 prompts that might help a smaller model perform this task. The prompts should be:
1. As long as necessary to be helpful for the smaller model.
2. Informative about what the task is.
3. Not repetitive with each other.
4. Not including any examples of the task.
Please write your prompts under the header "# Task Prompts". 

# Task examples
{task_examples}

Now, think step by step and follow the instructions above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SHORT = "short"
LONG = "long"
PROMPT_TEMPLATES = {
    SHORT: SHORT_PROMPT_GENERATION_PROMPT_TEMPLATE,
    LONG: LONG_PROMPT_GENERATION_PROMPT_TEMPLATE,
}
PROMPT_TYPES = list(PROMPT_TEMPLATES.keys())


def _verbose_example_formatter(example):
    return f"Input: {example['input']}\nOutput: {example['output']}\n\n"


EXAMPLE_FORMATTERS = {
    "verbose": _verbose_example_formatter,
}


DEFAULT_RANDOM_SEED = 42


def build_prompt(
    dataset,
    num_examples: int = 32,
    example_formatter: typing.Callable[[typing.Dict], str] = _verbose_example_formatter,
    prompt_template: str = SHORT_PROMPT_GENERATION_PROMPT_TEMPLATE,
    random_examples: bool = True,
    start_index: int = 0,
    seed: int = DEFAULT_RANDOM_SEED,
):
    if random_examples:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(dataset))
        if num_examples >= len(dataset):
            logger.warning(
                f"Asked for {num_examples} but the dataset has only {len(dataset)}, defaulting to, {len(dataset)}"
            )
            num_examples = len(dataset)

        example_indices = perm[:num_examples]

    else:
        example_indices = np.arange(start_index, start_index + num_examples)

    examples = [dataset[int(i)] for i in example_indices]
    task_examples = "".join([example_formatter(example) for example in examples])
    prompt = prompt_template.format(task_examples=task_examples).strip()
    return prompt, example_indices


DEFAULT_QUERY_PARAMETERS = {
    "do_sample": True,
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "watermark": False,
}


class TooManyTokensError(Exception):
    pass


def query_model(
    prompt: str,
    model_url: str,
    query_parameters: typing.Dict[str, typing.Any] | None = None,
    n_attempts: int = 6,
):
    parameters = dict()
    parameters.update(DEFAULT_QUERY_PARAMETERS)
    if query_parameters is not None:
        parameters.update(query_parameters)

    query_dict = {"inputs": prompt, "parameters": parameters}

    att = 0
    success = False
    resp_json = []

    while att < n_attempts and not success:
        try:
            response = requests.post(model_url, json=query_dict, timeout=600)
            if "json" not in response.headers.get("Content-Type", ""):
                logger.warning(f"Received non-JSON response:\n{response.text}")

                if "`inputs` tokens + `max_new_tokens` must be" in response.text:
                    raise TooManyTokensError()

                continue

            resp_json = response.json()
            if isinstance(resp_json, dict) and resp_json["message"] == "Endpoint request timed out":
                logger.warning(f"Timed out on attempt {att}, retrying")
            else:
                success = True

        except requests.exceptions.JSONDecodeError as e:
            logger.exception(e)
            logger.debug(f"Response text:\n{response.text}")

        finally:
            att += 1
            if att < n_attempts:
                sleep_time = 2**att
                if att > 1:
                    logger.info(f"Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)

    if len(resp_json) == 0 or (isinstance(resp_json, dict) and 0 not in resp_json):
        logger.error(f"Unexpected JSON:\n{resp_json}")
        text = ""
    else:
        text = resp_json[0]["generated_text"]

    return text, response


LINE_START_PATTERN = re.compile(r"^\d+\.")


def parse_prompts_from_response(response_text: str):
    model_response = response_text[response_text.find(RESPONSE_START) :]
    if PROMPTS_HEADER in model_response:
        model_response = model_response[model_response.find(PROMPTS_HEADER) + len(PROMPTS_HEADER) :]

    lines = model_response.splitlines()
    parsed_prompts = []

    for line in lines:
        line = line.strip()
        m = LINE_START_PATTERN.match(line)
        if m:
            p = LINE_START_PATTERN.sub("", line).strip()
            if p.endswith("."):
                p = p[:-1]

            parsed_prompts.append(p)

    return parsed_prompts


def generate_prompt_set_from_multiple_queries(
    dataset,
    n_queries: int = 10,
    model_url: str = MODEL_URLS[LLAMA_31_405B],
    build_prompt_parameters: typing.Dict[str, typing.Any] | None = None,
    query_parameters: typing.Dict[str, typing.Any] | None = None,
):
    prompt_set = set()
    raw_prompts = []
    response_jsons = []
    example_index_sets = []

    for i in tqdm(range(n_queries), desc="Queries", total=n_queries):
        build_prompt_parameters = {} if build_prompt_parameters is None else build_prompt_parameters
        model_prompt, example_indices = build_prompt(dataset, **build_prompt_parameters)
        raw_prompts.append(model_prompt)
        example_index_sets.append(example_indices)
        response_text, response = query_model(model_prompt, model_url, query_parameters)
        response_jsons.append(response.json())
        prompt_set.update(parse_prompts_from_response(response_text))

    return prompt_set, raw_prompts, response_jsons


STORAGE_ROOT = os.environ.get("STORAGE_ROOT")


def main(args: typing.Optional[typing.List[str]] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to be loaded",
        nargs="+",
        type=str,
        required=True,
        action="extend",
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
        default=f"{STORAGE_ROOT}/function_vectors/prompts",
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
        "--num_queries",
        help="Number of times to query the prompt-generating model",
        type=int,
        required=False,
        default=20,
    )
    parser.add_argument(
        "--prompt_model",
        help="Which model to prompt",
        default=LLAMA_31_405B,
        choices=list(MODEL_URLS.keys()),
    )
    parser.add_argument(
        "-force_regenerate",
        help="Regenerate even if the file exists",
        action="store_true",
    )

    prompt_group = parser.add_argument_group("build_prompt", "Prompt building parameters")
    prompt_group.add_argument(
        "--prompt_template_key",
        help="Generate short or long prompts",
        required=True,
        choices=PROMPT_TYPES,
    )
    prompt_group.add_argument("--num_examples", help="How many task examples to give the model", default=32, type=int)
    prompt_group.add_argument(
        "--formatter_key",
        help="Which formatter to use to format task examples",
        default="verbose",
        choices=list(EXAMPLE_FORMATTERS.keys()),
    )
    prompt_group.add_argument(
        "--dont_randomize_examples",
        help="Disable example randomization",
        action="store_true",
    )
    prompt_group.add_argument(
        "--example_start_index",
        help="If not randomizing, which index to start from",
        default=0,
        type=int,
    )

    query_group = parser.add_argument_group("query", "Query parameters")
    query_group.add_argument("--max_new_tokens", help="How many tokens to generate", default=2048, type=int)
    query_group.add_argument("--temperature", help="Generation temperature", default=0.7, type=float)

    args = parser.parse_args(args)

    root_data_dir = args.root_data_dir
    save_path_suffix = args.save_path_suffix
    save_path_root = (
        f"{args.save_path_root}/{save_path_suffix}" if save_path_suffix is not None else args.save_path_root
    )

    seed = args.random_seed
    prompt_template_key = args.prompt_template_key
    logger.info(str(args))

    dataset_names = args.dataset_name
    for i, dataset_name in enumerate(dataset_names):
        logger.info(f"On dataset {i + 1}/{len(dataset_names)}: {dataset_name}")

        set_seed(seed)
        if not os.path.exists(root_data_dir):
            raise ValueError(f"Dataset Directory Not Found: {root_data_dir}")

        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=0.01, seed=seed)

        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        output_name = dataset_name
        if prompt_template_key != SHORT:
            output_name = f"{output_name}_{prompt_template_key}"

        # output_path = make_valid_path_name(
        #     f"{save_path_root}/{output_name}_prompts.json"
        # )
        output_path = f"{save_path_root}/{output_name}_prompts.json"

        if not args.force_regenerate and os.path.exists(output_path):
            logger.info(
                f"Skipping '{dataset_name}' for prompt template '{prompt_template_key}' as the following file exists: {output_path} "
            )
            continue

        build_prompt_parameters = dict(
            prompt_template=PROMPT_TEMPLATES[prompt_template_key],
            num_examples=args.num_examples,
            example_formatter=EXAMPLE_FORMATTERS[args.formatter_key],
            random_examples=not args.dont_randomize_examples,
            start_index=args.example_start_index,
        )

        query_parameters = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature)

        prompt_set, raw_prompts, response_jsons = generate_prompt_set_from_multiple_queries(
            dataset["train"],
            args.num_queries,
            MODEL_URLS[args.prompt_model],
            build_prompt_parameters,
            query_parameters,
        )

        output_dict = dict(
            dataset_name=dataset_name,
            prompts=list(prompt_set),
        )

        with open(output_path, "w") as output_file:
            json.dump(output_dict, output_file)
            logger.info(f"Saved prompts to {output_path}")

        debug_info_dict = dict(
            args=vars(args),
            raw_prompts=raw_prompts,
            response_jsons=response_jsons,
        )
        debug_info_path = make_valid_path_name(f"{save_path_root}/{output_name}_debug_info.json")
        with open(debug_info_path, "w") as debug_info_file:
            json.dump(debug_info_dict, debug_info_file)
            logger.info(f"Saved debug info to {debug_info_path}")


if __name__ == "__main__":
    main()
