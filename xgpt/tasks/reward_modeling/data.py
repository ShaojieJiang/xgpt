from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from xgpt.core.data import TransformerDataModule


class ComparisonDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        new_features = []
        for feature in features:
            pos_feature = {}
            neg_feature = {}
            pos_feature["input_ids"] = feature["input_ids_pos"]
            pos_feature["attention_mask"] = feature["attention_mask_pos"]
            neg_feature["input_ids"] = feature["input_ids_neg"]
            neg_feature["attention_mask"] = feature["attention_mask_neg"]
            new_features.extend([pos_feature, neg_feature])
        return super().__call__(new_features)


class BasicDataModule(TransformerDataModule):
    @property
    def data_module_file_path(self):
        return __file__

    def process_data(
        self, dataset: Union[Dataset, DatasetDict], stage: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        cols_to_keep = [
            x
            for x in [
                "input_ids_pos",
                "input_ids_neg",
                "attention_mask_pos",
                "attention_mask_neg",
            ]
            if x in dataset["train"].features
        ]
        dataset.set_format(columns=cols_to_keep)

        return dataset

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
        max_length=2048,
    ):
        model_inputs_pos = tokenizer(
            examples["prompt_pos"],
            padding=padding,
            truncation=False,
        )

        model_inputs_neg = tokenizer(
            examples["prompt_neg"],
            padding=padding,
            truncation=False,
        )

        for key in model_inputs_pos:  # throw away the oldest context
            model_inputs_pos[key] = [seq[-max_length:] for seq in model_inputs_pos[key]]

        for key in model_inputs_neg:  # throw away the oldest context
            model_inputs_neg[key] = [seq[-max_length:] for seq in model_inputs_neg[key]]

        # set attention mask for <pad> token to 0
        for i in range(len(model_inputs_pos["attention_mask"])):
            model_inputs_pos["attention_mask"][1][-1] = 0

        for i in range(len(model_inputs_neg["attention_mask"])):
            model_inputs_neg["attention_mask"][1][-1] = 0

        model_inputs = {
            "input_ids_pos": model_inputs_pos["input_ids"],
            "input_ids_neg": model_inputs_neg["input_ids"],
            "attention_mask_pos": model_inputs_pos["attention_mask"],
            "attention_mask_neg": model_inputs_neg["attention_mask"],
        }
        return model_inputs

    @property
    def collate_fn(self) -> Callable:
        return ComparisonDataCollatorWithPadding(self.tokenizer)
