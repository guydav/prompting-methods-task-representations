import sys
import traceback
from functools import reduce

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from recipe.function_vectors.cache_short_texts import (
    main as cache_short_texts_main,
)


def multi_replace(input_string, old_options, new):
    return reduce(lambda s, old: s.replace(old, new), old_options, input_string)


OmegaConf.register_new_resolver("replace", lambda s, old, new: s.replace(old, new))
OmegaConf.register_new_resolver("multi_replace", multi_replace)


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
        # "--dataset_name",
        # config.dataset,
        "--model_name",
        config.model,
        # "--prompt_baseline",
        # config.prompt_baseline,
        "--min_batch_size",
        config.batch_size,
    ]
    if "max_length_tokens" in config:
        args.extend(["--max_length_tokens", config.max_length_tokens])

    args = [str(arg) for arg in args]

    logger.info(args)

    try:
        cache_short_texts_main(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # flush everything
        sys.stdout.flush()
        sys.stderr.flush()


# def main() -> None:
#     prompt_function_vector_main()


if __name__ == "__main__":
    main()
