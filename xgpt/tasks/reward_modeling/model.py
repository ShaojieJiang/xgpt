from typing import Any, Type

import torch
import transformers
from hydra.utils import get_class
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy
from transformers import AutoModel

from xgpt.core.model import TaskTransformer


def reward_loss(pooled_logits):
    pos_scores = pooled_logits[::2, :1]  # even entries
    neg_scores = pooled_logits[1::2, :]  # odd entries

    # margin = 16  # when neg_score - pos_score <= -margin, original loss is close enough to zero
    loss = (  # centered and with margin
        torch.log(1 + torch.exp(neg_scores - pos_scores)).mean()
        # + torch.clamp(margin + neg_scores - pos_scores, 0.0).mean()
        # + pooled_logits.mean() ** 2
    )

    return loss


class RewardModelingTransformer(TaskTransformer):
    def __init__(
        self,
        *args,
        downstream_model_type: Type["AutoModel"] = transformers.AutoModelForSeq2SeqLM,
        **kwargs,
    ) -> None:
        if type(downstream_model_type) is str:
            downstream_model_type = get_class(downstream_model_type)
        super().__init__(downstream_model_type, *args, **kwargs)
        self.accuracy = BinaryAccuracy()
        self.mean_score = MeanMetric()
        self.mean_diff_score = MeanMetric()

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        if "labels" not in batch:
            labels = batch["input_ids"]
        else:
            labels = batch["labels"]
            batch.pop("labels")
        valid_entries = labels.int() != -100
        if valid_entries.sum().item() == 0:
            # no labels
            return None

        outputs = self.model(**batch, return_dict=True)
        loss = reward_loss(outputs.logits)

        self.log(f"{prefix}_loss", loss, sync_dist=True)
        self.log_metrics(outputs.logits, prefix)
        return loss

    def reset_metrics(self):
        self.accuracy.reset()
        self.mean_score.reset()
        self.mean_diff_score.reset()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        self.reset_metrics()
        return self.common_step("train", batch)

    def on_validation_epoch_start(
        self,
    ) -> None:
        self.reset_metrics()

    def on_test_epoch_start(self) -> None:
        self.reset_metrics()

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.common_step("test", batch)

    def log_metrics(self, scores, prefix):
        self.mean_score.update(scores)
        pos_scores = scores[::2, :]  # even entries
        neg_scores = scores[1::2, :]  # odd entries

        diff_scores = pos_scores - neg_scores
        self.mean_diff_score.update(diff_scores)

        preds = (pos_scores > neg_scores).squeeze(-1).int()
        target = torch.ones_like(preds)
        self.accuracy.update(preds, target)

        margin = 16
        contrast_loss = torch.log(1 + torch.exp(neg_scores - pos_scores)).mean()
        margin_loss = torch.clamp(margin + neg_scores - pos_scores, 0.0).mean()
        center_loss = scores.mean() ** 2

        on_epoch = False if prefix == "train" else True

        self.log_dict(
            {
                f"{prefix}_score": self.mean_score.compute()
                if self.training
                else self.mean_score,
                f"{prefix}_diff_socre": self.mean_diff_score.compute()
                if self.training
                else self.mean_diff_score,
                f"{prefix}_acc": self.accuracy.compute()
                if self.training
                else self.accuracy,
                f"loss/{prefix}_contrast_loss": contrast_loss,
                f"loss/{prefix}_margin_loss": margin_loss,
                f"loss/{prefix}_center_loss": center_loss,
            },
            sync_dist=True,
            on_epoch=on_epoch,
        )
