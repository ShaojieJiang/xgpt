from functools import partial

from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig


def load_dataset_and_prepare(datasets_config: DictConfig, eos_token):
    def generate_examples(data_split):
        for dialog in data_split:
            chosen = (
                dialog["chosen"]
                .replace("Human: Assistant:", "Assistant:")
                .replace("Assistant: Human:", "Human:")
            )
            rejected = (
                dialog["rejected"]
                .replace("Human: Assistant:", "Assistant:")
                .replace("Assistant: Human:", "Human:")
            )

            yield {
                "prompt_pos": chosen + f" {eos_token}",
                "prompt_neg": rejected + f" {eos_token}",
            }

    # load from HF hub
    datasets_config.path = "Anthropic/hh-rlhf"
    dataset = load_dataset(**datasets_config)

    prepared_dataset = DatasetDict()
    for split in dataset.keys():
        data_split = dataset[split]
        data_generator = partial(generate_examples, data_split=data_split)
        prepared_dataset[split] = Dataset.from_generator(data_generator)

    return prepared_dataset
