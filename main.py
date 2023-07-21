import logging
from typing import Iterable, Union

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from transformers import AutoTokenizer

from xgpt.core.data import TransformerDataModule
from xgpt.core.model import TaskTransformer


def run(
    tokenizer_cfg: DictConfig,
    model_cfg: DictConfig,
    data_module_cfg: DictConfig,
    trainer_cfg: DictConfig,
    logger: Union[Logger, Iterable[Logger], bool] = True,
    run_test_after_fit: bool = True,
    stage: str = "train",
) -> None:
    # instantiate tokenizer and model
    tokenizer: AutoTokenizer = instantiate(tokenizer_cfg)
    model: TaskTransformer = instantiate(
        model_cfg, tokenizer=tokenizer
    )  # load pretrained

    # instantiate data module
    data_module: TransformerDataModule = instantiate(
        data_module_cfg, tokenizer=tokenizer
    )
    data_module.setup("fit")

    # instantiate trainer
    trainer: pl.Trainer = instantiate(trainer_cfg, logger=logger)

    if stage != "train" and trainer.num_devices > 1:
        logging.warn(
            f"Running stage {stage} while using more than 1 devices. "
            "This can lead to unexpected behaviours, such as metrics not correctly "
            "averaged, and only part of the texts are logged. It's recommended to "
            "use only 1 device for inference, although that may lead to long inference time."
        )

    if stage == "train":
        trainer.fit(model, datamodule=data_module)
        if run_test_after_fit:
            trainer.test(model, datamodule=data_module)
    else:  # stage == "valid" or stage == "test"
        if stage == "validate":
            trainer.validate(model, datamodule=data_module)
        elif stage == "test":
            trainer.test(model, datamodule=data_module)
        else:
            raise NotImplementedError(
                f"Only 'train', 'validate' and 'test' stages are supported, but you set stage={stage}."
            )


def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    rank_zero_info(OmegaConf.to_yaml(cfg))
    if cfg.get("logger", None):
        logger = instantiate(cfg.logger)
        logger.log_hyperparams(cfg)
    else:
        logger = True  # use TensorBoardLogger

    if cfg.get("seed", None):
        seed_everything(cfg.seed, workers=True)

    run(
        tokenizer_cfg=cfg.get("tokenizer"),
        model_cfg=cfg.get("model"),
        data_module_cfg=cfg.get("data_module"),
        trainer_cfg=cfg.get("trainer"),
        logger=logger,
        run_test_after_fit=cfg.training.get("run_test_after_fit", False),
        stage=cfg.get("stage"),
    )


@hydra.main(config_path="./xgpt/conf", config_name="rlhf_rm", version_base="1.2.0")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
