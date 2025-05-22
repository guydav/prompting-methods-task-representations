import sys
import traceback
from functools import reduce

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from recipe.function_vectors.evaluate_function_vector import evaluate_fv_main


def multi_replace(input_string, old_options, new):
    return reduce(lambda s, old: s.replace(old, new), old_options, input_string)


OmegaConf.register_new_resolver("replace", lambda s, old, new: s.replace(old, new))
OmegaConf.register_new_resolver("multi_replace", multi_replace)


EXTRA_CONFIG_ARGS = [
    ("device", True),
    ("save_path_suffix", True),
    ("universal_set", False),
    ("n_top_heads", True),
    ("force_evaluation", False),
    ("force_fs_layer_sweep", False),
    ("fs_layer_sweep_only", False),
    ("use_old_paths", False),
]


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="icl_fv.yaml",
)
def main(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"saving logs, configs, and model checkpoints to {config.logs_dir}")

    args = ["--dataset_name", config.dataset, "--model_name", config.model]
    for arg, include_value in EXTRA_CONFIG_ARGS:
        if arg in config:
            args.append("--" + arg)
            if include_value:
                args.append(str(config[arg]))

    logger.info(args)
    try:
        evaluate_fv_main(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
