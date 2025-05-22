import sys
import traceback
from functools import reduce

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from recipe.function_vectors.prompt_eval_filter import prompt_filter_main


def multi_replace(input_string, old_options, new):
    return reduce(lambda s, old: s.replace(old, new), old_options, input_string)


OmegaConf.register_new_resolver("replace", lambda s, old, new: s.replace(old, new))
OmegaConf.register_new_resolver("multi_replace", multi_replace)


EXTRA_CONFIG_ARGS = [
    ("batch_size", True),
    ("device", True),
    ("allow_different_examples_per_prompt", False),
    ("cache_prompt_prefixes", False),
    ("dont_cache_prompt_prefixes", False),
    ("save_path_suffix", True),
    ("force_prompt_evaluation", False),
]


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="prompt_filter.yaml",
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
        prompt_filter_main(args)
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
