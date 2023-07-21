from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from attr import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from xgpt.utils.prompt_templates import temporary_setattr

from ..language_modeling.data import BasicDataModule as LMBasicDataModule


@dataclass
class DataCollatorLeftPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        with temporary_setattr(self.tokenizer, "padding_side", "left"):
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


class BasicDataModule(LMBasicDataModule):
    @property
    def data_module_file_path(self):
        return __file__

    @property
    def collate_fn(self) -> Callable:
        return DataCollatorLeftPadding(self.tokenizer)
