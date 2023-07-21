from typing import Callable, Optional, Union

import pytorch_lightning as pl
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from xgpt.utils.dataset import local_dataset_module


class TransformerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 32,
        num_workers: int = 0,
        datasets_config: Optional[DictConfig] = None,
        train_val_split: Optional[int] = None,
        padding: Union[str, bool] = "max_length",
        max_length: int = 128,
        drop_long_seq: bool = False,
        preprocessing_num_workers: int = 1,
        load_from_cache_file: bool = True,
        from_hf_hub: bool = False,
        shuffle_training: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets_config = datasets_config
        self.train_val_split = train_val_split
        self.padding = padding
        self.max_length = max_length
        self.drop_long_seq = drop_long_seq
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.from_hf_hub = from_hf_hub
        self.shuffle_training = shuffle_training
        self.infer_dataset_name_and_module()

    def infer_dataset_name_and_module(self):
        if not self.from_hf_hub:  # resolve local path
            self.dataset_module = local_dataset_module(
                self.__module__,
                self.datasets_config.path,
                dataset_type="datasets",
            )

    @property
    def data_module_file_path(self):
        raise NotImplementedError("Subclass should implement this!")

    def setup(self, stage: Optional[str] = None):
        dataset = self.load_dataset()
        dataset = self.split_dataset(dataset)
        dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset

    def process_data(
        self, dataset: Union[Dataset, DatasetDict], stage: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        raise NotImplementedError("Subclass should implement this!")

    def load_dataset(self) -> Dataset:
        if self.from_hf_hub:
            dataset = load_dataset(**self.datasets_config)
        else:
            dataset = self.dataset_module.load_dataset_and_prepare(
                self.datasets_config, self.tokenizer.eos_token
            )
        return dataset

    def split_dataset(
        self, dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        if "validation" not in dataset and self.train_val_split is not None:
            split = dataset["train"].train_test_split(self.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            return DataLoader(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def collate_fn(self) -> Optional[Callable]:
        return None
