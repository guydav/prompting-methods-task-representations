import sys
import traceback
from functools import reduce

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from recipe.function_vectors.prompt_based_function_vector import (
    prompt_function_vector_main,
)


def multi_replace(input_string, old_options, new):
    return reduce(lambda s, old: s.replace(old, new), old_options, input_string)


OmegaConf.register_new_resolver("replace", lambda s, old, new: s.replace(old, new))
OmegaConf.register_new_resolver("multi_replace", multi_replace)


EXTRA_CONFIG_ARGS = [
    ("batch_size", True),
    ("device", True),
    ("allow_different_examples_per_prompt", False),
    ("cache_prompt_prefixes", False),
    ("save_path_suffix", True),
    ("force_indirect_effect", False),
    ("universal_set", False),
    ("n_top_heads", True),
    ("force_evaluation", False),
    ("joint_intervention", False),
    ("joint_intervention_icl_root", True),
    ("joint_intervention_min_layer_depth", True),
    ("joint_intervention_max_layer_depth", True),
    ("use_icl_top_heads", False),
    ("use_icl_mean_activations", False),
    ("add_prompt_fv_twice", False),
    ("add_icl_fv_twice", False),
    ("use_min_abs_heads_prompt", False),
    ("use_min_abs_heads_icl", False),
    ("use_bottom_heads_prompt", False),
    ("use_bottom_heads_icl", False),
    ("use_instruct_model_fv", False),
    ("instruct_model_suffix", True),
    ("remove_model_suffix", True),
]


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="prompt_fv.yaml",
)
def main(config: DictConfig) -> None:
    # This main is used to circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664
    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"saving logs, configs, and model checkpoints to {config.logs_dir}")

    args = [
        "--dataset_name",
        config.dataset,
        "--model_name",
        config.model,
        "--prompt_type",
        config.prompt_type,
        "--prompt_baseline",
        config.prompt_baseline,
    ]

    for arg, include_value in EXTRA_CONFIG_ARGS:
        if arg in config:
            args.append("--" + arg)
            if include_value:
                args.append(config[arg])

    # if "device" in config:
    #     args.extend(["--device", config.device])
    # if "allow_different_examples_per_prompt" in config:
    #     args.append("--allow_different_examples_per_prompt")
    # if "cache_prompt_prefixes" in config:
    #     args.append["--cache_prompt_prefixes"]
    # if "save_path_suffix" in config:
    #     args.extend(["--save_path_suffix", config.save_path_suffix])

    args = [str(arg) for arg in args]

    logger.info(args)

    try:
        prompt_function_vector_main(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


# def main() -> None:
#     prompt_function_vector_main()


if __name__ == "__main__":
    main()
