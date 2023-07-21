import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from peft import get_peft_model, prepare_model_for_int8_training
from transformers import AutoConfig, PreTrainedTokenizerBase

from xgpt.utils.deepspeed import enable_transformers_pretrained_deepspeed_sharding

if TYPE_CHECKING:
    from transformers import AutoModel
logger = logging.getLogger(__name__)


class TaskTransformer(pl.LightningModule):
    def __init__(
        self,
        downstream_model_type: Type["AutoModel"],
        model_kwargs: DictConfig = None,
        pretrained_model_name_or_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        load_weights: bool = True,
        use_peft: bool = False,
        lora_config: DictConfig = None,
        deepspeed_sharding: bool = False,
        gradient_checkpointing: bool = True,
        optimizer: DictConfig = None,
        scheduler: DictConfig = None,
        scheduler_monitor: str = "val_loss",
        scheduler_mode: str = "min",
        generation_kwargs: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.load_weights = load_weights
        self.use_peft = use_peft
        self.lora_config = lora_config
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.downstream_model_type = downstream_model_type
        self.gradient_checkpointing = gradient_checkpointing
        self.deepspeed_sharding = deepspeed_sharding
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = tokenizer
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_mode = scheduler_mode
        self.generation_kwargs = (
            dict(generation_kwargs) if generation_kwargs is not None else {}
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.deepspeed_sharding and self.trainer.num_devices > 1:
            enable_transformers_pretrained_deepspeed_sharding(self)
        if self.model_kwargs.load_in_8bit or self.model_kwargs.load_in_4bit:
            device_map = {"": self.trainer.local_rank}
        else:
            device_map = None
        self.initialize_model(device_map)

    def initialize_model(self, device_map=None):
        if self.load_weights:
            model = self.downstream_model_type.from_pretrained(
                self.pretrained_model_name_or_path,
                device_map=device_map,
                **self.model_kwargs,
            )
        else:
            config = AutoConfig.from_pretrained(
                self.pretrained_model_name_or_path, **self.model_kwargs
            )
            model = self.downstream_model_type(config)
        # in case word embedding size is different than vocab size
        model.resize_token_embeddings(len(self.tokenizer.vocab))
        if self.use_peft:
            lora_config = instantiate(self.lora_config)
            model = prepare_model_for_int8_training(
                model, use_gradient_checkpointing=self.gradient_checkpointing
            )
            model.enable_input_require_grads()
            self.model = get_peft_model(model, lora_config)
        else:
            self.model = model
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        self.unify_special_token_ids()

    def unify_special_token_ids(self):
        for key in self.tokenizer.special_tokens_map:
            id_name = f"{key}_id"
            id_in_tokenizer = getattr(self.tokenizer, id_name)
            if hasattr(self.model.config, id_name):
                id_in_config = getattr(self.model.config, id_name)
                if id_in_config != id_in_tokenizer:
                    logger.warn(
                        f"{id_name} in model.config is {id_in_config}, but"
                        f" in tokenizer is {id_in_tokenizer}! Setting it to"
                        f" {id_in_tokenizer} for avoiding unpredictable behaviours."
                    )
                    setattr(self.model.config, id_name, id_in_tokenizer)

            if hasattr(self.model.generation_config, id_name):
                id_in_gen_config = getattr(self.model.generation_config, id_name)
                if id_in_gen_config != id_in_tokenizer:
                    logger.warn(
                        f"{id_name} in model.geneartion_config is {id_in_gen_config}, but"
                        f" in tokenizer is {id_in_tokenizer}! Setting it to"
                        f" {id_in_tokenizer} for avoiding unpredictable behaviours."
                    )
                    setattr(self.model.generation_config, id_name, id_in_tokenizer)

    def configure_optimizers(self) -> Dict:
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        optimizer = instantiate(self.optimizer_cfg, trainable_params)
        scheduler = instantiate(self.scheduler_cfg, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": self.scheduler_monitor,
                "mode": self.scheduler_mode,
                "strict": True,
                "name": None,
            },
        }

    def save_hf_checkpoint(self, path: Union[str, Path]) -> None:
        self.model.save_pretrained(path)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer.local_rank == 0:  # main process
            hf_dir = f"{self.trainer.default_root_dir}/huggingface"
            # remove if exist:
            if os.path.isdir(hf_dir):
                shutil.rmtree(hf_dir)
            # save new
            self.model.save_pretrained(hf_dir)
            self.tokenizer.save_pretrained(hf_dir)
        return checkpoint
