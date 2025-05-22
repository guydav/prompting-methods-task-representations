from collections import defaultdict

import bitsandbytes as bnb
import torch
from baukit import TraceDict, get_module
from loguru import logger
from torch.nn import functional as F

from recipe.function_vectors.utils.shared_utils import (
    EvalDataResults,
    tokenizer_padding_side_token,
)


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def replace_activation_w_avg(
    layer_head_token_pairs,
    avg_activations,
    model,
    model_config,
    idx_map,
    batched_input=False,
    last_token_only=False,
):
    """
    An intervention function for replacing activations with a computed average value.
    This function replaces the output of one (or several) attention head(s) with a pre-computed average value
    (usually taken from another set of runs with a particular property).
    The batched_input flag is used for systematic interventions where we are sweeping over all attention heads for a given (layer,token)
    The last_token_only flag is used for interventions where we only intervene on the last token (such as zero-shot or concept-naming)

    Parameters:
    layer_head_token_pairs: list of tuple triplets each containing a layer index, head index, and token index [(L,H,T), ...]
    avg_activations: torch tensor of the average activations (across ICL prompts) for each attention head of the model.
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    idx_map: dict mapping prompt label indices to ground truth label indices
    batched_input: whether or not to batch the intervention across all heads
    last_token_only: whether our intervention is only at the last token

    Returns:
    rep_act: A function that specifies how to replace activations with an average when given a hooked pytorch module.
    """
    edit_layers = [x[0] for x in layer_head_token_pairs]

    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (
                model_config["n_heads"],
                # model_config["resid_dim"] // model_config["n_heads"],
                model_config["head_dim"],
            )  # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape)  # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Perform Intervention:
            if batched_input:
                # Patch activations from avg activations into baseline sentences (i.e. n_head baseline sentences being modified in this case)
                for i in range(model_config["n_heads"]):
                    layer, head_n, token_n = layer_head_token_pairs[i]
                    inputs[i, token_n, head_n] = avg_activations[layer, head_n, idx_map[token_n]]
            elif last_token_only:
                # Patch activations only at the last token for interventions like
                for layer, head_n, token_n in layer_head_token_pairs:
                    if layer == current_layer:
                        # print(f"Intervening at layer {current_layer}: inputs([{inputs.shape[0] - 1, inputs.shape[1] - 1, head_n}]) = avg_activations[{layer, head_n, idx_map[token_n]}]")
                        inputs[-1, -1, head_n] = avg_activations[layer, head_n, idx_map[token_n]]
            else:
                # Patch activations into baseline sentence found at index, -1 of the batch (targeted & multi-token patching)
                for layer, head_n, token_n in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, token_n, head_n] = avg_activations[layer, head_n, idx_map[token_n]]

            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            if (
                "gpt2-xl" in model_config["name_or_path"]
            ):  # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)

            elif any(name in model_config["name_or_path"].lower() for name in ("olmo", "gemma", "gpt-j")):
                new_output = torch.matmul(inputs, out_proj.T)

            elif "gpt-neox" in model_config["name_or_path"] or "pythia" in model_config["name_or_path"]:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)

            elif "llama" in model_config["name_or_path"]:
                if "70b" in model_config["name_or_path"]:
                    # need to dequantize weights
                    out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)

            return new_output
        else:
            return output

    return rep_act


def batched_replace_activation_w_avg(
    batch_layer_head_token_pairs,
    output_positions,
    avg_activations,
    model,
    model_config,
    batch_idx_map,
    last_token_only=False,
):
    """
    A batched intervention function for replacing activations with a computed average value.
    This function allows applying different interventions to each entry in a batch.
    Each batch entry can have its own set of intervention locations and token mapping.

    Parameters:
    batch_layer_head_token_pairs: list of lists, where each inner list contains triplets [(L,H,T), ...] for one batch entry
    output_positions: tensor of output positions for each batch entry (last token to intervene in)
    avg_activations: torch tensor of the average activations (across ICL prompts) for each attention head of the model
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    batch_idx_map: list of dicts, each mapping prompt label indices to ground truth label indices for one batch entry
    last_token_only: whether our intervention is only at the last token

    Returns:
    rep_act: A function that specifies how to replace activations with an average when given a hooked pytorch module.
    """
    # Validate input formats
    if not isinstance(batch_layer_head_token_pairs, list) or not all(
        isinstance(entry, list) for entry in batch_layer_head_token_pairs
    ):
        raise ValueError("batch_layer_head_token_pairs must be a list of lists")

    if not isinstance(batch_idx_map, list) or not all(isinstance(entry, dict) for entry in batch_idx_map):
        raise ValueError("batch_idx_map must be a list of dictionaries")

    if len(batch_layer_head_token_pairs) != len(batch_idx_map):
        raise ValueError(
            f"Number of intervention specifications ({len(batch_layer_head_token_pairs)}) must match number of index mappings ({len(batch_idx_map)})"
        )

    if len(batch_layer_head_token_pairs) != output_positions.numel():
        raise ValueError(
            f"Number of output positions ({len(output_positions)}) must match number of batch entries ({len(batch_layer_head_token_pairs)})"
        )

    if output_positions.dim() == 0:
        output_positions = output_positions.unsqueeze(0)

    # Collect all layers that need editing across all batch entries
    edit_layers = set([lht[0] for batch_lht_pairs in batch_layer_head_token_pairs for lht in batch_lht_pairs])
    # for batch_lht_pairs in batch_layer_head_token_pairs:
    #     edit_layers.update([x[0] for x in batch_lht_pairs])

    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (
                model_config["n_heads"],
                # model_config["resid_dim"] // model_config["n_heads"],
                model_config["head_dim"],
            )  # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape)  # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Perform Intervention
            batch_size = inputs.size(0)

            # Validate that we have the right number of intervention specs and mappings at runtime
            if len(batch_layer_head_token_pairs) != batch_size:
                raise ValueError(
                    f"Number of intervention specifications ({len(batch_layer_head_token_pairs)}) must match batch size ({batch_size})"
                )
            if len(batch_idx_map) != batch_size:
                raise ValueError(
                    f"Number of index mappings ({len(batch_idx_map)}) must match batch size ({batch_size})"
                )

            # # Apply interventions for each batch entry
            for batch_idx in range(batch_size):
                batch_lht_pairs = batch_layer_head_token_pairs[batch_idx]
                batch_map = batch_idx_map[batch_idx]

                if last_token_only:
                    # Only intervene on the last token for this batch entry
                    # TODO: this could also be batched if we want to
                    for layer, head_n, token_n in batch_lht_pairs:
                        if layer == current_layer:
                            # print(f"Intervening at layer {current_layer}: inputs([{batch_idx, inputs.shape[1] - 1, head_n}]) = avg_activations[{layer, head_n, batch_map[token_n]}]")
                            inputs[batch_idx, output_positions[batch_idx], head_n] = avg_activations[
                                layer, head_n, batch_map[token_n]
                            ]
                else:
                    # Intervene at specific (layer, head, token) locations for this batch entry
                    for layer, head_n, token_n in batch_lht_pairs:
                        if layer == current_layer:
                            inputs[batch_idx, token_n, head_n] = avg_activations[layer, head_n, batch_map[token_n]]

            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            if (
                "gpt2-xl" in model_config["name_or_path"]
            ):  # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)

            elif any(name in model_config["name_or_path"].lower() for name in ("olmo", "gemma", "gpt-j")):
                new_output = torch.matmul(inputs, out_proj.T)

            elif "gpt-neox" in model_config["name_or_path"] or "pythia" in model_config["name_or_path"]:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)

            elif "llama" in model_config["name_or_path"]:
                if "70b" in model_config["name_or_path"]:
                    # need to dequantize weights
                    out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)

            return new_output
        else:
            return output

    return rep_act


def add_function_vector(
    edit_layer_or_layers, fv_vector_or_vectors, device, idx: torch.Tensor | int = -1, same_layer_combination="sum"
):
    """
    Adds a vector to the output of a specified layer in the model

    Parameters:
    edit_layer: the layer to perform the FV intervention
    fv_vector: the function vector to add as an intervention
    device: device of the model (cuda gpu or cpu)
    idx: the token index to add the function vector at

    Returns:
    add_act: a fuction specifying how to add a function vector to a layer's output hidden state
    """
    if isinstance(edit_layer_or_layers, int):
        edit_layer_or_layers = [edit_layer_or_layers]

    edit_layer_or_layers = [int(layer) for layer in edit_layer_or_layers]

    if isinstance(fv_vector_or_vectors, torch.Tensor):
        fv_vector_or_vectors = [fv_vector_or_vectors]

    if len(edit_layer_or_layers) != len(fv_vector_or_vectors):
        raise ValueError("The number of layers must match the number of function vectors.")

    same_layer_combination_func = None
    match same_layer_combination:
        case "sum":
            same_layer_combination_func = torch.sum
        case "mean":
            same_layer_combination_func = torch.mean
        case _:
            raise ValueError(f"Unsupported same_layer_combination: {same_layer_combination}")

    edit_layer_to_fvs = defaultdict(list)
    for edit_layer, fv_vector in zip(edit_layer_or_layers, fv_vector_or_vectors):
        edit_layer_to_fvs[edit_layer].append(fv_vector)

    fv_size = fv_vector_or_vectors[0].size()

    edit_layer_to_final_fv = {
        edit_layer: fvs[0]
        if len(fvs) == 1
        else same_layer_combination_func(torch.stack(fvs, dim=0), dim=0).view(*fv_size)
        for edit_layer, fvs in edit_layer_to_fvs.items()
    }

    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layer_to_final_fv:
            fv_vector = edit_layer_to_final_fv[current_layer]
            if current_layer == edit_layer:
                if isinstance(output, tuple):
                    if isinstance(idx, torch.Tensor):
                        batch_size = output[0].size(0)
                        if idx.size(0) != batch_size:
                            raise ValueError(
                                f"Batch size ({batch_size}) must match index tensor size ({idx.size(0)}): {output[0].shape} vs. {idx.shape}"
                            )
                        output[0][torch.arange(batch_size), idx] += fv_vector.to(device)
                    else:
                        output[0][:, idx] += fv_vector.to(device)
                    return output
                else:
                    return output
            else:
                return output

    return add_act


def original_function_vector_intervention(
    sentence,
    target,
    edit_layer,
    function_vector,
    model,
    model_config,
    tokenizer,
    compute_nll=False,
    generate_str=False,
):
    """
    Runs the model on the sentence and adds the function_vector to the output of edit_layer as a model intervention, predicting a single token.
    Returns the output of the model with and without intervention.

    Parameters:
    sentence: the sentence to be run through the model
    target: expected response of the model (str, or [str])
    edit_layer: layer at which to add the function vector
    function_vector: torch vector that triggers execution of a task
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced completion (used to compute perplexity (PPL))
    generate_str: whether to generate a string of tokens or predict a single token

    Returns:
    fvi_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors="pt").to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze())
        nll_targets[
            :, :-target_len
        ] = -100  # This is the accepted value to skip indices when computing loss (see nn.CrossEntropyLoss default)
        output = model(**nll_inputs, labels=nll_targets)
        clean_nll = output.loss.item()
        clean_output = output.logits[:, original_pred_idx, :]
        intervention_idx = -1 - target_len
    elif generate_str:
        MAX_NEW_TOKENS = 16
        output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1, max_new_tokens=MAX_NEW_TOKENS)
        clean_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        intervention_idx = -1
    else:
        clean_output = model(**inputs).logits[:, -1, :]
        intervention_idx = -1

    # Perform Intervention
    intervention_fn = add_function_vector(
        edit_layer,
        function_vector.reshape(1, model_config["resid_dim"]),
        model.device,
        idx=intervention_idx,
    )
    with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=intervention_fn):
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:, original_pred_idx, :]
        elif generate_str:
            MAX_NEW_TOKENS = 16
            output = model.generate(
                inputs.input_ids,
                top_p=0.9,
                temperature=0.1,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            intervention_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            intervention_output = model(**inputs).logits[
                :, -1, :
            ]  # batch_size x n_tokens x vocab_size, only want last token prediction

    fvi_output = (clean_output, intervention_output)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)

    return fvi_output


@tokenizer_padding_side_token
def function_vector_intervention(
    eval_data: EvalDataResults,
    edit_layer_or_layers,
    function_vector_or_vectors,
    model,
    model_config,
    tokenizer,
    compute_nll=False,
    generate_str=False,
    top_p=0.9,
    temperature=0.1,
    test_deterministic=False,
):
    """
    Runs the model on the sentence and adds the function_vector to the output of edit_layer as a model intervention, predicting a single token.
    Returns the output of the model with and without intervention.

    Parameters:
    sentence: the sentence to be run through the model
    target: expected response of the model (str, or [str])
    edit_layer: layer at which to add the function vector
    function_vector: torch vector that triggers execution of a task
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced completion (used to compute perplexity (PPL))
    generate_str: whether to generate a string of tokens or predict a single token

    Returns:
    fvi_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # Clean Run, No Intervention:
    if len(eval_data) != 1:
        raise ValueError("Expected a single sentence to evaluate.")

    if generate_str and compute_nll:
        raise ValueError("Cannot compute NLL and generate strings simultaneously.")

    sentence = eval_data.sentences
    target = eval_data.targets

    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1
    intervention_idx = -1
    intervention_data = eval_data.clone()
    MAX_NEW_TOKENS = 16

    if generate_str:
        if not test_deterministic:
            output = model.generate(
                **inputs,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        eval_data.strings = [tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])]

    else:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors="pt").to(device)
        nll_input_matches_input = (nll_inputs.input_ids[:, : inputs.input_ids.shape[1]] == inputs.input_ids) | (
            inputs.input_ids == tokenizer.pad_token_id
        )

        if not nll_input_matches_input.all():
            logger.warning(
                "Adding the target completion changes at least one input token. This may lead to incorrect NLL computation."
            )
            logger.debug(f"Sentence: '{sentence}'\nTarget: '{target}'\nTarget Completion: '{target_completion}'")

        if compute_nll:
            nll_targets = nll_inputs.input_ids.clone()
            input_length = len(inputs.input_ids.squeeze())

            while inputs.input_ids[:, input_length - 1] != nll_targets[:, input_length - 1]:
                input_length -= 1

            nll_targets[
                :, :input_length
            ] = -100  # This is the accepted value to skip indices when computing loss in nn.CrossEntropyLoss

            output = model(**nll_inputs, labels=nll_targets)
            eval_data.nlls = [output.loss.item()]
            eval_data.logits = output.logits[:, original_pred_idx, :]
            intervention_idx = original_pred_idx

        else:
            eval_data.logits = model(**inputs).logits[:, -1, :]

    if not isinstance(function_vector_or_vectors, list):
        function_vector_or_vectors = [function_vector_or_vectors]

    function_vector_or_vectors = [fv.reshape(1, model_config["resid_dim"]) for fv in function_vector_or_vectors]

    # Perform Intervention
    intervention_fn = add_function_vector(
        edit_layer_or_layers,
        function_vector_or_vectors,
        model.device,
        idx=intervention_idx,
    )
    with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=intervention_fn):
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_data.nlls = [output.loss.item()]
            intervention_data.logits = output.logits[:, original_pred_idx, :]
        elif generate_str:
            output = model.generate(
                inputs.input_ids,
                top_p=0.9,
                temperature=0.1,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            intervention_data.strings = [tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])]
        else:
            intervention_data.logits = model(**inputs).logits[
                :, -1, :
            ]  # batch_size x n_tokens x vocab_size, only want last token prediction

    return eval_data, intervention_data


@tokenizer_padding_side_token
def batch_function_vector_intervention(
    eval_data: EvalDataResults,
    edit_layer_or_layers,
    function_vector_or_vectors,
    model,
    model_config,
    tokenizer,
    compute_nll=False,
    generate_str=False,
    top_p=0.9,
    temperature=0.1,
    test_deterministic=False,
):
    """
    Runs the model on the sentence and adds the function_vector to the output of edit_layer as a model intervention, predicting a single token.
    Returns the output of the model with and without intervention.

    Parameters:
    sentence: the sentence to be run through the model
    target: expected response of the model (str, or [str])
    edit_layer: layer at which to add the function vector
    function_vector: torch vector that triggers execution of a task
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced completion (used to compute perplexity (PPL))
    generate_str: whether to generate a string of tokens or predict a single token

    Returns:
    fvi_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # Clean Run, No Intervention:
    # if len(eval_data) == 1:
    #     raise ValueError("Expected more than a single sentence to evaluate.")

    if generate_str and compute_nll:
        raise ValueError("Cannot compute NLL and generate strings simultaneously.")

    sentences = eval_data.sentences
    targets = eval_data.targets

    device = model.device
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    # original_pred_idx = len(inputs.input_ids.squeeze()) - 1
    # intervention_idx = -1
    intervention_data = eval_data.clone()
    MAX_NEW_TOKENS = 16

    # Find pad positions in the input tokenized data
    pad_positions = torch.argmax((inputs.input_ids == tokenizer.pad_token_id).to(int), dim=1)

    # Grab the output positions
    output_positions = torch.tensor(
        [(inputs.input_ids.shape[1] if pos.item() == 0 else pos.item()) - 1 for pos in pad_positions],
        dtype=torch.long,
    )
    batch_indices = torch.arange(len(output_positions), dtype=torch.long)

    if generate_str:
        if not test_deterministic:
            outputs = model.generate(
                **inputs,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        eval_data.strings = tokenizer.batch_decode(outputs[:, -MAX_NEW_TOKENS:], skip_special_tokens=True)

    else:
        target_completions = [sentence + target for sentence, target in zip(sentences, targets)]
        nll_inputs = tokenizer(target_completions, return_tensors="pt", padding=True).to(device)

        nll_input_matches_input = (nll_inputs.input_ids[:, : inputs.input_ids.shape[1]] == inputs.input_ids) | (
            inputs.input_ids == tokenizer.pad_token_id
        )

        if not nll_input_matches_input.all():
            logger.warning(
                "Adding the target completion changes at least one input token. This may lead to incorrect NLL computation."
            )
            logger.debug(f"Sentence: '{sentences}'\nTarget: '{targets}'\nTarget Completion: '{target_completions}'")

        # Adjust for no padding case
        seq_lengths = []
        for i, pos in enumerate(pad_positions):
            if pos.item() == 0 and inputs.input_ids[i, 0] != tokenizer.pad_token_id:
                # No padding found, use full length
                seq_lengths.append(inputs.input_ids.shape[1])
            else:
                seq_lengths.append(pos.item())

        # Create labels tensor with -100 for input tokens (to be ignored in loss)
        nll_targets = nll_inputs.input_ids.clone()

        # Mask out the original sentence portions
        for i, input_length in enumerate(seq_lengths):
            valid_length = input_length
            # Ensure we're at the right boundary by checking token IDs
            while valid_length > 0 and nll_targets[i, valid_length - 1] != inputs.input_ids[i, valid_length - 1]:
                valid_length -= 1
            nll_targets[i, :valid_length] = -100

        # Set pad tokens to ignore index
        nll_targets[nll_targets == tokenizer.pad_token_id] = -100

        outputs = model(**nll_inputs, labels=nll_targets)

        # sanity check, but this shouldn't happen:
        if (outputs.logits.size(0) != batch_indices.size(0)) or (
            output_positions.max().item() >= outputs.logits.size(1)
        ):
            raise ValueError(
                f"Encountered mismatched batch size or output positions: {outputs.logits.size()} with batch indices {batch_indices} and output positions {output_positions}"
            )

        # Get predictions at the end of input sequences
        eval_data.logits = outputs.logits[batch_indices, output_positions, :]

        if compute_nll:
            # Calculate NLL for each example
            batch_nlls = []
            for i in range(len(sentences)):
                example_nll = F.cross_entropy(
                    outputs.logits[i, :-1],
                    nll_targets[i, 1:],
                    ignore_index=-100,
                    reduction="mean",
                )
                batch_nlls.append(example_nll.item())

            eval_data.nlls = batch_nlls

    # Perform Intervention
    # print(output_positions)

    if not isinstance(function_vector_or_vectors, list):
        function_vector_or_vectors = [function_vector_or_vectors]

    function_vector_or_vectors = [fv.reshape(1, model_config["resid_dim"]) for fv in function_vector_or_vectors]

    intervention_fn = add_function_vector(
        edit_layer_or_layers,
        function_vector_or_vectors,
        model.device,
        idx=output_positions,
    )
    with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=intervention_fn):
        if generate_str:
            if not test_deterministic:
                outputs = model.generate(
                    **inputs,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            intervention_data.strings = tokenizer.batch_decode(outputs[:, -MAX_NEW_TOKENS:], skip_special_tokens=True)
        else:
            outputs = model(**nll_inputs, labels=nll_targets)

            # Get predictions at the end of input sequences
            intervention_data.logits = outputs.logits[batch_indices, output_positions, :]

            if compute_nll:
                # Calculate NLL for each example
                intervention_nlls = []
                for i in range(len(sentences)):
                    example_nll = F.cross_entropy(
                        outputs.logits[i, :-1],
                        nll_targets[i, 1:],
                        ignore_index=-100,
                        reduction="mean",
                    )
                    intervention_nlls.append(example_nll.item())

                intervention_data.nlls = intervention_nlls

    return eval_data, intervention_data


def fv_intervention_natural_text(
    sentence,
    edit_layer,
    function_vector,
    *,
    model,
    model_config,
    tokenizer,
    max_new_tokens=16,
    num_interv_tokens=None,
    do_sample=False,
):
    """
    Allows for intervention in natural text where we generate and intervene on several tokens in a row.

    Parameters:
    sentence: sentence to intervene on with the FV
    edit_layer: layer at which to add the function vector
    function_vector: vector to add to the model that triggers execution of a task
    model: huggingface model
    model_config: dict with model config parameters (n_layers, n_heads, etc.)
    tokenizer: huggingface tokenizer
    max_new_tokens: number of tokens to generate
    num_interv_tokens: number of tokens to apply the intervention for (defaults to all subsequent generations)
    do_sample: whether to sample from top p tokens (True) or have deterministic greedy decoding (False)

    Returns:
    clean_output: tokens of clean output
    intervention_output: tokens of intervention output

    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    clean_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Perform Intervention
    intervention_fn = add_function_vector(edit_layer, function_vector, model.device)

    if (
        num_interv_tokens is not None and num_interv_tokens < max_new_tokens
    ):  # Intervene only for a certain number of tokens
        num_extra_tokens = max_new_tokens - num_interv_tokens
        with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=intervention_fn):
            intervention_output = model.generate(
                **inputs,
                max_new_tokens=num_interv_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        intervention_output = model.generate(
            intervention_output,
            max_new_tokens=num_extra_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
        )
    else:
        with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=intervention_fn):
            intervention_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

    return clean_output, intervention_output


def add_avg_to_activation(
    layer_head_token_pairs,
    avg_activations,
    model,
    model_config,
    batched_input=False,
    last_token_only=False,
):
    """
    An intervention function for adding a computed average value to activations.
    This function adds a pre-computed average value to the output of one (or several) attention head(s)
    (usually taken from another set of runs with a particular property).
    The batched_input flag is used for systematic interventions where we are sweeping over all attention heads for a given (layer,token)
    The last_token_only flag is used for interventions where we only intervene on the last token (such as zero-shot or concept-naming)

    Parameters:
    layer_head_token_pairs: list of tuple triplets each containing a layer index, head index, and token index [(L,H,T), ...]
    avg_activations: torch tensor of the average activations (across ICL prompts) for each attention head of the model.
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    batched_input: whether or not to batch the intervention across all heads
    last_token_only: whether our intervention is only at the last token

    Returns:
    add_act: A function that specifies how to replace activations with an average when given a hooked pytorch module.
    """
    edit_layers = [x[0] for x in layer_head_token_pairs]
    device = model.device

    def add_act(output, layer_name, inputs):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (
                model_config["n_heads"],
                # model_config["resid_dim"] // model_config["n_heads"],
                model_config["head_dim"],
            )  # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape)  # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Perform Intervention:
            if batched_input:
                # Patch activations from avg activations into baseline sentences (i.e. n_head baseline sentences being modified in this case)
                for i in range(model_config["n_heads"]):
                    layer, head_n, token_n = layer_head_token_pairs[i]
                    inputs[i, token_n, head_n] += avg_activations[layer, head_n, token_n].to(device)
            elif last_token_only:
                # Patch activations only at the last token for interventions like: (zero-shot, concept-naming, etc.)
                for layer, head_n, token_n in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, -1, head_n] += avg_activations[layer, head_n, token_n].to(device)
            else:
                # Patch activations into baseline sentence found at index, -1 of the batch (targeted & multi-token patching)
                for layer, head_n, token_n in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, token_n, head_n] += avg_activations[layer, head_n, token_n].to(device)

            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            if (
                "gpt2-xl" in model_config["name_or_path"]
            ):  # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)

            elif any(name in model_config["name_or_path"].lower() for name in ("olmo", "gemma", "gpt-j")):
                new_output = torch.matmul(inputs, out_proj.T)

            elif "gpt-neox" in model_config["name_or_path"] or "pythia" in model_config["name_or_path"]:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)

            elif "llama" in model_config["name_or_path"]:
                if "70b" in model_config["name_or_path"]:
                    # need to dequantize weights
                    out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)

            return new_output
        else:
            return output

    return add_act
