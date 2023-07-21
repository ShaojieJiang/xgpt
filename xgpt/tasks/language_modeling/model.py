from typing import Any, Type

import torch
import transformers
from hydra.utils import get_class
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.nn import CrossEntropyLoss
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoModel

from xgpt.core.model import TaskTransformer
from xgpt.utils.prompt_templates import clean_responses, temporary_setattr


class LanguageModelingTransformer(TaskTransformer):
    def __init__(
        self,
        *args,
        downstream_model_type: Type["AutoModel"] = transformers.AutoModelForSeq2SeqLM,
        compute_generate_metrics: bool = True,
        limit_generate_batches: int = None,
        **kwargs,
    ) -> None:
        if type(downstream_model_type) is str:
            downstream_model_type = get_class(downstream_model_type)
        super().__init__(downstream_model_type, *args, **kwargs)
        self.should_compute_generate_metrics = compute_generate_metrics
        self.limit_generate_batches = limit_generate_batches
        self.loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.rouge_metric = ROUGEScore()
        self.bleu_metric = BLEUScore()

    def common_step(self, prefix: str, batch: Any, batch_idx: int) -> torch.Tensor:
        labels = batch["input_ids"]
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )

        logits = outputs.logits.to(torch.float32)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        self.log(f"{prefix}_loss", loss, sync_dist=True)

        ppl = loss.exp()
        self.log(f"{prefix}_ppl", ppl, sync_dist=True)

        if self.should_compute_generate_metrics:
            self.compute_generate_metrics(batch, prefix, batch_idx)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step("train", batch, batch_idx)

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.common_step("val", batch, batch_idx)

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.common_step("test", batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        if hasattr(self.trainer.datamodule, "shuffle_align_train_ds"):
            return self.trainer.datamodule.shuffle_align_train_ds()

    def compute_generate_metrics(self, batch, prefix, batch_idx: int):
        if (
            self.limit_generate_batches is not None
            and batch_idx >= self.limit_generate_batches
        ):
            return

        if "inference_input" in batch and prefix != "train":
            if hasattr(self.generation_kwargs, "max_new_tokens"):
                del self.generation_kwargs.max_new_tokens

            max_new_tokens = batch["inference_target"].size(1)
            with temporary_setattr(self.model.config, "use_cache", True):
                output = self.model.generate(
                    input_ids=batch["inference_input"],
                    max_new_tokens=max_new_tokens,
                    **self.generation_kwargs,
                )
            query = self.tokenizer.batch_decode(
                batch["inference_input"], skip_special_tokens=True
            )
            # decode and clean
            preds = clean_responses(batch["inference_input"], output, self.tokenizer)
            targets = self.tokenizer.batch_decode(
                batch["inference_target"], skip_special_tokens=True
            )

            if isinstance(self.logger, WandbLogger):
                columns = ["query", "response", "ground_truth"]
                data = [
                    row
                    for row in zip(
                        query,
                        preds,
                        targets,
                    )
                ]
                self.logger.log_text(
                    key=f"{prefix}_samples", columns=columns, data=data
                )

            targets = [[target.strip()] for target in targets]
            # compute BLEU, ROUGE, and log
            rouge_scores = self.rouge_metric(preds, targets)
            bleu_score = self.bleu_metric(preds, targets)
            rouge_scores = {
                f"generation/{key}": value for key, value in rouge_scores.items()
            }
            self.log_dict(rouge_scores, sync_dist=True)
            self.log("generation/bleu", bleu_score, sync_dist=True)
