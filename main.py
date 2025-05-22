import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from loguru import logger


@hydra.main(
    version_base="1.2",
    config_path="configs",
    config_name="train_defaults.yaml",
)
def main(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"saving logs, configs, and model checkpoints to {config.logs_dir}")

    datamodule = instantiate(config.datamodule)
    model = instantiate(config.module)

    trainer = instantiate(
        config.trainer,
        model,
        datamodule.get_trainloader(),
        datamodule.get_testloader(),
        **config.trainer,
    )
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    main()
