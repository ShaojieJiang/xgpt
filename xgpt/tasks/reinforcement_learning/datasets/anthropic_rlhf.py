from functools import partial

from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig


def load_dataset_and_prepare(datasets_config: DictConfig, eos_token):
    def generate_examples(data_split):
        data = data_split["chosen"]
        for dialog in data:
            dialog.replace("Human: Assistant:", "Assistant:")
            dialog.replace("Assistant: Human:", "Human:")
            turns = dialog.split("\n\n")

            role = "Assistant: "
            if role in turns[-1]:
                text = "\n\n".join(turns[:-1] + [role])

                yield {
                    "text": text,
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
