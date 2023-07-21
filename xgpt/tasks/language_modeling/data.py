from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from xgpt.core.data import TransformerDataModule
from xgpt.utils.prompt_templates import temporary_setattr


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
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )
        if self.drop_long_seq:
            dataset = dataset.filter(
                lambda example: len(example["input_ids"]) <= self.max_length
            )

        cols_to_keep = [
            x for x in ["input_ids", "attention_mask"] if x in dataset["train"].features
        ]
        dataset.set_format(columns=cols_to_keep)

        return dataset

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        padding: str,
    ):
        model_inputs = tokenizer(
            examples["text"],
            padding=padding,
            truncation=False,
            add_special_tokens=False,
        )

        return model_inputs

    @property
    def collate_fn(self) -> Callable:
        return DataCollatorWithPadding(self.tokenizer)


class PairedDataModule(BasicDataModule):
    class PairedDataCollatorWithPadding(DataCollatorWithPadding):
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            input_features = [
                {
                    "input_ids": ex["input_ids"],
                    "attention_mask": ex["attention_mask_input"],
                }
                for ex in features
            ]
            output_features = [
                {
                    "input_ids": ex["output_ids"],
                    "attention_mask": ex["attention_mask_output"],
                }
                for ex in features
            ]
            in_out_features = [
                {
                    "input_ids": ex["input_ids"] + ex["output_ids"],
                    "attention_mask": ex["attention_mask_input"]
                    + ex["attention_mask_output"],
                }
                for ex in features
            ]

            with temporary_setattr(self.tokenizer, "padding_side", "left"):
                batch_input = self.tokenizer.pad(
                    input_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )

            with temporary_setattr(self.tokenizer, "padding_side", "right"):
                batch = self.tokenizer.pad(
                    in_out_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                batch_output = self.tokenizer.pad(
                    output_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )

            batch["inference_input"] = batch_input["input_ids"]
            batch["inference_target"] = batch_output["input_ids"]

            return batch

    def process_data(
        self, dataset: Union[Dataset, DatasetDict], stage: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.padding,
        )
        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )
        if self.drop_long_seq:
            dataset = dataset.filter(
                lambda example: len(example["input_ids"] + example["output_ids"])
                <= self.max_length
            )

        cols_to_keep = [
            x
            for x in [
                "input_ids",
                "output_ids",
                "attention_mask_input",
                "attention_mask_output",
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
    ):
        with temporary_setattr(tokenizer, "padding_side", "left"):
            inputs = tokenizer(
                examples["input"],
                padding=padding,
                truncation=False,
                add_special_tokens=False,
            )

        with temporary_setattr(tokenizer, "padding_side", "right"):
            outputs = tokenizer(
                examples["output"],
                padding=padding,
                truncation=False,
                add_special_tokens=False,
            )

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "output_ids": outputs["input_ids"],
            "attention_mask_input": inputs["attention_mask"],
            "attention_mask_output": outputs["attention_mask"],
        }

        return model_inputs

    @property
    def collate_fn(self) -> callable:
        return self.PairedDataCollatorWithPadding(self.tokenizer)
