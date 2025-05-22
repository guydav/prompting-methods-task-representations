import inspect
import os
import re
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split

from recipe.function_vectors.utils.shared_utils import tokenizer_padding_side_token


def create_fewshot_primer(prompt_data) -> str:
    """Creates the primer string for GPT in-context learning

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information

    Returns:
    prompt: the constructed ICL prompt primer as a string
    """
    prompt = ""
    prompt += (
        prompt_data["prefixes"]["instructions"]
        + prompt_data["instructions"]
        + prompt_data["separators"]["instructions"]
    )

    for example in prompt_data["examples"]:
        prompt += prompt_data["prefixes"]["input"] + example["input"] + prompt_data["separators"]["input"]
        prompt += prompt_data["prefixes"]["output"] + example["output"] + prompt_data["separators"]["output"]

    return prompt


def create_prompt(prompt_data, sentence=None) -> str:
    """Creates a prompt using the specified sentence for GPT in-context learning

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence/word) to include in the ICL prompt

    Returns:
    prompt: the constructed ICL prompt as a string
    """
    if sentence is None and prompt_data["query_target"] is not None:
        sentence = prompt_data["query_target"]["input"]

    if isinstance(sentence, list):
        sentence = sentence[0]

    prompt_init = create_fewshot_primer(prompt_data)
    prompt = prompt_init + prompt_data["prefixes"]["input"] + sentence + prompt_data["separators"]["input"]
    prompt += prompt_data["prefixes"]["output"]

    return prompt


# Partial primer & prompt functions
def create_partial_fewshot_primer(prompt_data, include=np.arange(8)) -> str:
    """Creates the primer string for GPT in-context learning, filtering to include a subset of specified priming strings

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    include: an iterable of ints indicating which examples to include in the ICL prompt

    Returns:
    prompt: the constructed ICL prompt primer as a string
    """
    prompt = ""
    prompt += (
        prompt_data["prefixes"]["instructions"]
        + prompt_data["instructions"]
        + prompt_data["separators"]["instructions"]
    )

    # Grab each priming example in the specified order.
    for i in include:
        example = prompt_data["examples"][i]
        prompt += prompt_data["prefixes"]["input"] + example["input"] + prompt_data["separators"]["input"]
        prompt += prompt_data["prefixes"]["output"] + example["output"] + prompt_data["separators"]["output"]

    return prompt


def create_partial_prompt(prompt_data, sentence=None, include=np.arange(8)) -> str:
    """Creates a prompt using the specified sentence and partial list of in-context primer sentences

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence /word) to include in the ICl prompt
    include: an iterable of ints indicating which examples to include in the ICL prompt

    Returns:
    prompt: the prompt as a string
    """
    if sentence is None and prompt_data["query_target"] is not None:
        sentence = prompt_data["query_target"]["input"]
    if isinstance(sentence, list):
        sentence = sentence[0]

    prompt_init = create_partial_fewshot_primer(prompt_data, include)

    prompt = prompt_init + prompt_data["prefixes"]["input"] + sentence + prompt_data["separators"]["input"]
    prompt += prompt_data["prefixes"]["output"]

    return prompt


# UTILS FOR GENERATING PROMPT META LABELS
def get_prompt_parts_and_labels(prompt_data, query_sentence=None):
    """
    Generates high-level labels for ICL prompts according to its ICL role, such as demonstration, label, separator, structural, etc.
    The JSON prompt format should include 'instructions', 'examples' with ('input', 'output') pairs,
    'prefixes', and 'separators' for 'input', 'output', and 'instructions'.
    Used in conjunction with tokenize_labels

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    query_sentence: optional (if contained in prompt_data) str containing a query for an ICL prompt

    Returns:
    prompt_parts: structured list of words to be flattened and tokenized
    prompt_part_labels: structured list of labels to be flattened & extended over tokenization
    """
    if query_sentence is None and prompt_data["query_target"] is not None:
        query_sentence = prompt_data["query_target"]["input"]
    if isinstance(query_sentence, list):
        query_sentence = query_sentence[0]
    n_examples = len(prompt_data["examples"])
    assemble_icl_example = lambda example, prompt_data: [
        prompt_data["prefixes"]["input"],
        example["input"],
        prompt_data["separators"]["input"],
        prompt_data["prefixes"]["output"],
        example["output"],
        prompt_data["separators"]["output"],
    ]
    assemble_icl_query = lambda query, prompt_data: [
        prompt_data["prefixes"]["input"],
        query,
        prompt_data["separators"]["input"],
        prompt_data["prefixes"]["output"],
    ]

    prompt_instructions = [
        prompt_data["prefixes"]["instructions"],
        prompt_data["instructions"],
        prompt_data["separators"]["instructions"],
    ]
    prompt_icl_examples = [assemble_icl_example(prompt_data["examples"][i], prompt_data) for i in range(n_examples)]
    prompt_icl_query = [assemble_icl_query(query_sentence, prompt_data)]

    prompt_instructions_labels = ["bos_token", "instructions_token", "separator_token"]
    prompt_icl_examples_labels = [
        [
            "structural_token",
            f"demonstration_{i + 1}_token",
            "separator_token",
            "structural_token",
            f"demonstration_{i + 1}_label_token",
            "separator_token",
        ]
        for i in range(n_examples)
    ]
    prompt_icl_query_labels = [
        [
            "query_structural_token",
            "query_demonstration_token",
            "query_separator_token",
            "query_structural_token",
        ]
    ]

    prompt_parts = prompt_instructions + prompt_icl_examples + prompt_icl_query

    prompt_part_labels = prompt_instructions_labels + prompt_icl_examples_labels + prompt_icl_query_labels

    return prompt_parts, prompt_part_labels


def extend_labels(sentence_parts, text_labels, tokenizer, tokenizer_prepends_bos=False, tokenizer_kwargs=None):
    """
    Extends ICL component labels across words that are tokenized into multiple tokens

    Parameters:
    sentence_parts: list, where each element is either a token (str), phrase (str), or list of tokens/phrases
    text_labels: list with the same structure as 'sentence_parts', with a corresponding label for that level of the input sentence.
    tokenizer: huggingface tokenizer

    Returns:
    final_labels: flattened/extended list of token labels for an ICL prompt (split into parts, contained in sentence_parts and text_labels)
    """
    final_labels = []
    if tokenizer_prepends_bos:
        final_labels = ["bos_token"]

    zipped_up = [
        list(zip(x, y)) if isinstance(x, list) else [(x, y)] for x, y in list(zip(sentence_parts, text_labels))
    ]

    prompt_builder = ""

    for element in zipped_up:
        for j, (word, label) in enumerate(element):
            if len(word) == 0:
                continue
            pre_prompt = prompt_builder
            pre_tokens = tokenizer(prompt_builder, return_tensors="pt").input_ids
            n_pre = pre_tokens.numel()
            prompt_builder += word
            post_tokens = tokenizer(prompt_builder, return_tensors="pt").input_ids
            n_post = post_tokens.numel()

            n_new_tokens = n_post - n_pre

            if n_new_tokens <= 0:
                # if tokenization gobbles up a previous label, then we overwrite the last previous label w/ label that should've been added
                # in rare edge cases, we can even end up with fewer tokens than we started with, so we look at <= rather than ==
                while n_new_tokens < 0:
                    final_labels.pop()
                    n_new_tokens += 1

                final_labels[-1] = label

            else:
                start_index = 1 if tokenizer_prepends_bos else 0
                first_mismatch_index = (
                    pre_tokens[0, start_index:] != post_tokens[0, start_index : pre_tokens.numel()]
                ).nonzero(as_tuple=True)[0]

                if first_mismatch_index.numel() > 0:
                    first_mismatch_index = first_mismatch_index.item() + start_index
                    # Check starting at the first mismatch
                    for end_index in range(first_mismatch_index, post_tokens.numel()):
                        # While we match the previous string (without the new word)
                        if pre_prompt.startswith(tokenizer.decode(post_tokens[0, start_index:end_index])):
                            # continue the previous label
                            final_labels.append(final_labels[-1])
                        else:
                            break

                        # For any remaining tokens (starting at the first mismatch)
                        # Write the new label, using the fallback case below
                        n_new_tokens = post_tokens.numel() - end_index

                # if there are no mismatches, we can just add the label for the new tokens
                final_labels.extend([label] * (n_new_tokens))

            if j == 3 or (j == 2 and len(element[3]) == 0):
                final_labels[-1] = (
                    final_labels[-1].replace("structural", "predictive").replace("separator", "predictive")
                )
            if j == 5:
                final_labels[-n_new_tokens] = final_labels[-n_new_tokens].replace("separator", "end_of_example")

    return final_labels


def _to_predictive(label):
    return label.replace("structural", "predictive").replace("separator", "predictive")


def _to_end_of_example(label):
    return label.replace("separator", "end_of_example")


def extend_labels_tokenize_combined(
    sentence_parts, text_labels, tokenizer, tokenizer_prepends_bos=False, tokenizer_kwargs=None
):
    flat_parts = []
    flat_labels = []
    label_indices_to_last_predictive = set()

    for part, label in zip(sentence_parts, text_labels):
        if isinstance(part, (list, tuple)):
            flat_parts.extend(part)
            flat_labels.extend(label)

            if len(label) > 3:
                index = len(flat_labels) - len(label) + 3
                if len(part[3]) == 0:
                    index -= 1
                label_indices_to_last_predictive.add(index)

            if len(label) == 6:
                flat_labels[-1] = _to_end_of_example(flat_labels[-1])

        else:
            flat_parts.append(part)
            flat_labels.append(label)

    label_end_indices = np.cumsum([len(p) for p in flat_parts])

    full_string = "".join(flat_parts)
    full_string_tokens = tokenizer(full_string).input_ids
    full_string_labels = []
    start_index = 0
    current_label_index = 0

    if (full_string_tokens[0] == tokenizer.bos_token_id) and (flat_labels[0] == "bos_token"):
        full_string_labels.append("bos_token")
        start_index = 1
        current_label_index = 1
        if full_string.startswith(tokenizer.bos_token):
            full_string = full_string[len(tokenizer.bos_token) :]
            label_end_indices -= len(tokenizer.bos_token)

    # check that we can recover the full string
    clean_up_tokenization_spaces = True
    full_decoded_string = tokenizer.decode(
        full_string_tokens[start_index:], clean_up_tokenization_spaces=clean_up_tokenization_spaces
    )
    if not full_decoded_string == full_string:
        clean_up_tokenization_spaces = False
        full_decoded_string = tokenizer.decode(
            full_string_tokens[start_index:], clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        if not full_decoded_string == full_string:
            raise ValueError(
                f"Tokenization failed to recover the original string witht either setting of `clean_up_tokenization_spaces`. Original: '{full_string}', Tokens: {full_string_tokens}"
            )

    for last_token_index in range(start_index, len(full_string_tokens)):
        substring = tokenizer.decode(
            full_string_tokens[start_index : last_token_index + 1],
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

        # This check can be too aggressive in some unicode edge cases, so removing it for now and relying on the above one
        # if not full_string.startswith(substring):
        #     raise ValueError(
        #         f"Tokenization created a non-substring fragment. Original: '{full_string}', Substring: '{substring}', Tokens: {full_string_tokens[start_index : last_token_index + 1]}, clean_up_tokenization_spaces={clean_up_tokenization_spaces}"
        #     )

        while (len(substring) > label_end_indices[current_label_index]) and (
            (current_label_index + 1) < len(label_end_indices)
        ):
            if current_label_index in label_indices_to_last_predictive:
                full_string_labels[-1] = _to_predictive(full_string_labels[-1])

            current_label_index += 1

        full_string_labels.append(flat_labels[current_label_index])

    # check this again for the last label index
    if current_label_index in label_indices_to_last_predictive:
        full_string_labels[-1] = _to_predictive(full_string_labels[-1])

    return full_string_labels


def tokenize_labels(sentence_parts, text_labels, tokenizer, tokenizer_prepends_bos=False, tokenizer_kwargs=None):
    """
    Extends phrase-level labels across tokenization for in-context learning prompts. Tested with GPT-2's tokenizer from huggingface.
    Parameters:
    sentence_parts: list, where each element is either a token (str), phrase (str), or list of tokens/phrases
    text_labels: list with the same structure as 'sentence_parts', with a corresponding label for that level of the input sentence.
    tokenizer: huggingface tokenizer

    Returns:
    labels: flattened/extended list of token labels for an ICL prompt (split into parts, contained in sentence_parts and text_labels)

    based on the tokenize_and_preserve_labels function from:
    https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
    """

    # If the model typically prepends a bos, we add a bos label to label init

    # labels = extend_labels(
    labels = extend_labels_tokenize_combined(
        sentence_parts,
        text_labels,
        tokenizer,
        tokenizer_prepends_bos=tokenizer_prepends_bos,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    # else:
    #     labels = extend_labels(
    #         sentence_parts,
    #         text_labels,
    #         tokenizer,
    #         label_init=[],
    #         tokenizer_kwargs=tokenizer_kwargs,
    #     )

    return labels


def get_token_meta_labels(prompt_data, tokenizer, query=None, prepend_bos=False, tokenizer_kwargs=None):
    """
    Computes the ICL meta-labels for every token in a prompt.

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    tokenizer: huggingface tokenizer
    query: str of the query input

    Return:
    token_labels: list of tuples (prompt token index, token, label)
    prompt_string: full prompt as a string
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    if query is None and prompt_data["query_target"] is not None:
        query = prompt_data["query_target"]["input"]
    if isinstance(query, list):
        query = query[0]

    prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)
    token_meta_labels = tokenize_labels(prompt_parts, prompt_part_labels, tokenizer, prepend_bos, tokenizer_kwargs)
    prompt_string = create_prompt(prompt_data=prompt_data, sentence=query)
    tokens = [tokenizer.decode(x) for x in tokenizer(prompt_string, **tokenizer_kwargs).input_ids]
    token_labels = list(zip(np.arange(len(tokens)), tokens, token_meta_labels))

    return token_labels, prompt_string


def get_dummy_token_labels(
    n_icl_examples,
    tokenizer,
    model_config,
    instructions="",
    prefixes=None,
    separators=None,
):
    """
    Computes the ground-truth meta labels & indices for an ICL prompt with the specified number of example pairs
    These GT labels assume each word gets a single token

    Parameters:
    n_icl_examples: number of ICL example pairs
    tokenizer: huggingface tokenizer
    prefixes: ICL template prefixes
    separators: ICL template separators

    Return:
    final_token_labels: list of tuples containing a token's index and label name [(int, str), ... ]
    """
    # If the model already prepends a bos token by default, we don't want to add one to our prompts
    prepend_bos = False if model_config["prepend_bos"] else True

    if prefixes is not None and separators is not None:
        dummy_prompt_data = word_pairs_to_prompt_data(
            {"input": ["a"] * n_icl_examples, "output": ["a"] * n_icl_examples},
            query_target_pair={"input": ["a"], "output": ["a"]},
            prepend_bos_token=prepend_bos,
            instructions=instructions,
            prefixes=prefixes,
            separators=separators,
            tokenizer=tokenizer,
        )
    else:
        dummy_prompt_data = word_pairs_to_prompt_data(
            {"input": ["a"] * n_icl_examples, "output": ["a"] * n_icl_examples},
            instructions=instructions,
            query_target_pair={"input": ["a"], "output": ["a"]},
            prepend_bos_token=prepend_bos,
            tokenizer=tokenizer,
        )
    final_token_labels, _ = get_token_meta_labels(dummy_prompt_data, tokenizer, prepend_bos=model_config["prepend_bos"])
    final_token_labels = [(x[0], x[-1]) for x in final_token_labels]
    return final_token_labels


def compute_duplicated_labels(token_labels, gt_labels):
    """
    Computes a map between duplicated labels and ground truth label positions for localized averaging

    Parameters:
    token_labels: token labels of actual prompt being used
    gt_labels: token labels for a "ground truth" prompt that assumes each input & output is a single token

    Returns:
    index_map: a dict mapping prompt label indices to ground truth label indices
    dup_label_ranges: indices where labels should be duplicated
    """
    check_inds = list(filter(lambda x: "demo" in x[2] or "instructions" in x[2], token_labels))
    dup_ranges = pd.DataFrame(check_inds).groupby(2)[0].aggregate(lambda x: (x.min(), x.max()))
    dup_labels = [v for v, x in dup_ranges.items() if (x[1] - x[0]) > 0]

    dup_label_ranges = dup_ranges[dup_labels].to_dict()
    dup_inds = pd.DataFrame(check_inds)[pd.DataFrame(check_inds)[2].duplicated()][0].values

    index_map = {k: v[0] for (k, v) in zip([x[0] for x in token_labels if x[0] not in dup_inds], gt_labels)}

    return index_map, dup_label_ranges


def update_idx_map(idx_map, idx_avg) -> dict:
    """
    Updates the idx_map to map duplicate tokens to its gt token position
    """
    update_map = {}
    for i, j in idx_avg.values():
        for k in range(i, j + 1):
            if k not in idx_map.keys():
                update_map[k] = idx_map[i]

    update_map = {**idx_map, **update_map}
    return update_map


def word_pairs_to_prompt_data(
    word_pairs: dict,
    instructions: str = "",
    prefixes: dict = {"input": "Q:", "output": "A:", "instructions": ""},
    separators: dict = {"input": "\n", "output": "\n\n", "instructions": ""},
    query_target_pair: dict = None,
    prepend_bos_token=False,
    shuffle_labels=False,
    prepend_space=True,
    prepend_space_to_prefix=False,
    tokenizer_bos_token: str | None = None,
    tokenizer=None,
) -> dict:
    """Takes a dataset of word pairs, and constructs a prompt_data dict with additional information to construct an ICL prompt.
    Parameters:
    word_pairs: dict of the form {'word1':['a', 'b', ...], 'word2':['c', 'd', ...]}
    instructions: prefix instructions for an ICL prompt
    prefixes: dict of ICL prefixes that are prepended to inputs, outputs and instructions
    separators: dict of ICL separators that are appended to inputs, outputs and instructions
    query_target_pair: dict with a single input-output pair acting as the query for the prompt
    prepend_bos_token: whether or not to prepend a BOS token to the prompt
    shuffle_labels: whether to shuffle the ICL labels
    prepend_space: whether to prepend a space to every input and output token

    Returns:
    prompt_data: dict containing ICL prompt examples, and template information
    """
    if tokenizer_bos_token is None:
        if tokenizer is not None:
            tokenizer_bos_token = tokenizer.bos_token
        elif prepend_bos_token:
            raise ValueError("'prepend_bos_token' is set to True, but no tokenizer or bos_token was provided.")

    prompt_data = {}
    prompt_data["instructions"] = instructions
    prompt_data["separators"] = separators
    if prepend_bos_token:
        prefixes = {k: (v if k != "instructions" else tokenizer_bos_token + v) for (k, v) in prefixes.items()}
    prompt_data["prefixes"] = prefixes

    if query_target_pair is not None:
        query_target_pair = {k: (v[0] if isinstance(v, list) else v) for k, v in query_target_pair.items()}
    prompt_data["query_target"] = query_target_pair

    pairs = word_pairs.values()
    if shuffle_labels:
        pairs = [
            np.random.permutation(x).tolist() if i == 1 else x for (i, x) in enumerate(list(pairs))
        ]  # shuffle labels only

    # if shuffle_labels:
    #     randomized_pairs = [np.random.permutation(x).tolist() if i==1 else x for (i,x) in enumerate(list(word_pairs.values()))] # shuffle labels only
    #     if prepend_space:
    #         prompt_data['examples'] = [{'input':' ' + w1, 'output':' ' + w2} for (w1,w2) in list(zip(*randomized_pairs))]
    #         prompt_data['query_target'] = {k:' ' + v for k,v in query_target_pair.items()} if query_target_pair is not None else None
    #     else:
    #         prompt_data['examples'] = [{'input':w1, 'output':w2} for (w1,w2) in list(zip(*randomized_pairs))]
    # else:

    if prepend_space_to_prefix:
        prompt_data["prefixes"] = {k: p + " " if len(p) > 0 else p for k, p in prompt_data["prefixes"].items()}
        prepend_space = False

    if prepend_space:
        prompt_data["examples"] = [{"input": " " + str(w1), "output": " " + str(w2)} for (w1, w2) in list(zip(*pairs))]
        prompt_data["query_target"] = (
            {k: " " + str(v) for k, v in query_target_pair.items()} if query_target_pair is not None else None
        )
    else:
        prompt_data["examples"] = [{"input": w1, "output": w2} for (w1, w2) in list(zip(*pairs))]

    return prompt_data


# DATASET UTILS
class ICLDataset:
    """
    A simple dataset class containing input-output pairs, used for ICL prompt construction.
    """

    def __init__(self, dataset):
        if isinstance(dataset, str):
            self.raw_data = pd.read_json(dataset)
        elif isinstance(dataset, dict):
            self.raw_data = pd.DataFrame(dataset)
        elif isinstance(dataset, pd.DataFrame):
            self.raw_data = dataset
        else:
            raise ValueError(f"Unrecognized ICLDataset type: {type(dataset)}")

        self.raw_data = self.raw_data[["input", "output"]]

    def subset(self, start=0, end=None):
        if end is None:
            end = len(self.raw_data)

        return ICLDataset(self.raw_data.iloc[start:end].copy(deep=True).reset_index(drop=True))

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.raw_data.iloc[i].to_dict()
        elif isinstance(i, slice):
            return self.raw_data.iloc[i].to_dict(orient="list")
        elif isinstance(i, list) or isinstance(i, np.ndarray):
            return self.raw_data.iloc[i].to_dict(orient="list")
        elif isinstance(i, str):
            if i not in self.raw_data.columns:
                raise KeyError(
                    f"Column '{i}' not in the dataset. Current columns in the dataset: {self.raw_data.columns.to_list()}"
                )
            else:
                return self.raw_data[i].to_list()
        else:
            raise ValueError(f"{i} is not a valid index type. Expected one of: [int, list, np.ndarray, slice, str]")

    def __len__(self):
        return len(self.raw_data)

    def __repr__(self):
        s = (
            "ICLDataset"
            + "({\n\tfeatures: "
            + f"{self.raw_data.columns.to_list()},\n\tnum_rows: {self.__len__()}"
            + "\n})"
        )
        return s


def split_icl_dataset(
    dataset,
    train_size=None,
    test_size=0.3,
    valid_from_train_size=None,
    seed=42,
    split_valid: bool = True,
) -> Dict[str, ICLDataset]:
    """
    Uses scikit-learn's train_test split to create train, valid, test dataset from provided dataset.

    Parameters:
    dataset: ICL dataset
    train_size: percentage of data (float between 0 and 1) to put in the training data split
    test_size: percentage of data (float between 0 and 1) to put into the test data split
    seed: seed used for splitting the data

    Returns:
    dict containing train, valid, test ICL datasets
    """
    if train_size is None and test_size is None:
        train_size = 0.7
        test_size = 0.3

    elif train_size is not None and test_size is None:
        test_size = 1 - train_size

    elif train_size is None and test_size is not None:
        train_size = 1 - test_size

    elif train_size is not None and test_size is not None:
        assert train_size + test_size == 1

    if valid_from_train_size is None:
        # Recover the previous validation size, when it a `test_size` split from the test set
        valid_from_train_size = (test_size**2) / (1 - test_size)
    else:
        assert (valid_from_train_size > 0) and (valid_from_train_size < 1)

    train, test = train_test_split(dataset.raw_data, test_size=test_size, random_state=seed)
    if split_valid:
        train, valid = train_test_split(train, test_size=valid_from_train_size, random_state=seed)
        valid = ICLDataset(valid.to_dict(orient="list"))

    train = ICLDataset(train.to_dict(orient="list"))
    test = ICLDataset(test.to_dict(orient="list"))

    return {"train": train, "valid": valid, "test": test} if split_valid else {"train": train, "test": test}


def find_dataset_folder(
    task_name: str,
    root_data_dir: str = "../dataset_files",
):
    data_folders = ["abstractive", "extractive"]

    d_group_map = [
        (
            dataset_type,
            os.path.exists(os.path.join(root_data_dir, dataset_type, task_name + ".json")),
        )
        for dataset_type in data_folders
    ]

    d_group = list(filter(lambda x: x[1], d_group_map))

    if len(d_group) != 1:
        return None

    return d_group[0][0]


def load_dataset(
    task_name: str,
    root_data_dir: str = "../dataset_files",
    test_size=0.3,
    seed=32,
    split_valid: bool = True,
) -> Dict[str, ICLDataset]:
    """
    Loads a dataset with input/output pairs

    Parameters:
    task_name: the name of the task dataset
    root_data_dir: the root directory where the data comes from
    test_size: fraction used in train/test split

    Return:
    dataset: the dict contain the train/valid/test dataset splits
    """
    data_folders = ["abstractive", "extractive"]
    assert test_size <= 1.0

    path = Path(root_data_dir)
    dataset_folder = find_dataset_folder(task_name, root_data_dir)

    assert dataset_folder is not None, (
        f"Error! 'task_name'={task_name}.json must be uniquely contained in one of these directories:{data_folders}. Please check the root_data_dir"
    )

    d_path = os.path.join(path, dataset_folder, f"{task_name}.json")

    dataset = ICLDataset(d_path)
    dataset = split_icl_dataset(dataset, test_size=test_size, seed=seed, split_valid=split_valid)

    return dataset


@tokenizer_padding_side_token
def filter_prompts_by_max_tokens(prompts, *, tokenizer, max_length_tokens):
    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    valid_indices = (
        torch.argwhere(
            (
                tokenized_prompts["input_ids"].shape[1]
                - (tokenized_prompts["input_ids"] == tokenizer.pad_token_id).sum(dim=1)
            )
            <= max_length_tokens
        )
        .squeeze()
        .tolist()
    )

    return valid_indices


def should_skip_prompt(target: str, prompt: str):
    should_skip = False
    t = str(target).lower()
    p = prompt.lower()

    try:
        should_skip = next(re.finditer(f"[\\s'\"]({re.escape(t)})[\\s'\".,]", p)) is not None
    except StopIteration:
        should_skip = p.startswith(t + " ")

    if should_skip:
        current_frame = inspect.currentframe()
        parent_frame = current_frame.f_back
        local_logger = logger
        if parent_frame:
            parent_function = parent_frame.f_code.co_name
            local_logger = logger.opt(depth=1).bind(function=parent_function)

        local_logger.info(f"Skipping '{target}' because it appears in the prompt: '{prompt}'")

    return should_skip
