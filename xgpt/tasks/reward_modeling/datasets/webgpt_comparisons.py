from functools import partial

from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig

from xgpt.tasks.language_modeling.datasets.webgpt_comparisons import (
    wrap_text_with_prompts,
)


def load_dataset_and_prepare(datasets_config: DictConfig, eos_token):
    def generate_examples(data_split):
        for ex in data_split:
            prompt_answer1 = wrap_text_with_prompts(ex, 0)
            prompt_answer2 = wrap_text_with_prompts(ex, 1)

            answer1 = ex["answer_0"]
            answer2 = ex["answer_1"]

            score1 = ex["score_0"]
            score2 = ex["score_1"]
            assert score1 + score2 == 0

            if not answer1 or not answer2 or not score1 or answer1 == answer2:
                continue

            if score1 > score2:
                prompt_pos, prompt_neg = prompt_answer1, prompt_answer2
            else:
                prompt_pos, prompt_neg = prompt_answer2, prompt_answer1

            yield {
                "prompt_pos": prompt_pos,
                "prompt_neg": prompt_neg,
            }

    # load from HF hub
    datasets_config.path = "openai/webgpt_comparisons"
    dataset = load_dataset(**datasets_config)

    dataset = dataset["train"]  # only train split exists
    num_exs = len(dataset)
    all_inds = list(range(num_exs))
    # uniformly sample from the data for val and test sets, 5% each
    validation_inds = list(range(0, num_exs, 20))
    test_inds = list(range(1, num_exs, 20))
    train_inds = [
        ind for ind in all_inds if ind not in validation_inds and ind not in test_inds
    ]

    prepared_dataset = DatasetDict()
    for split in ["train", "validation", "test"]:
        data_split = dataset.select(eval(f"{split}_inds"))
        data_generator = partial(generate_examples, data_split=data_split)
        prepared_dataset[split] = Dataset.from_generator(data_generator)

    return prepared_dataset
