from functools import partial

from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig

from xgpt.utils.prompt_templates import (
    knowledge_prompt,
    question_knowledge_prompt,
    wrap_answer,
)


def wrap_knowledge(knowledge_dict):
    result = ""
    for i, (title, extract) in enumerate(
        zip(knowledge_dict["title"], knowledge_dict["extract"])
    ):
        result += knowledge_prompt(i + 1, title, extract)

    return result.strip()


def wrap_text_with_prompts(data_dict, pos_ind):
    question = data_dict["question"]["full_text"]
    knowledge = wrap_knowledge(data_dict[f"quotes_{pos_ind}"])

    return question_knowledge_prompt(question, knowledge)


def load_dataset_and_prepare(datasets_config: DictConfig, eos_token):
    def generate_examples(data_split):
        for ex in data_split:
            input = ""
            output = ""
            pos_ind = None
            if ex["score_0"] > 0 and ex["answer_0"]:  # answer_0 is a positive example
                pos_ind = 0
            elif ex["score_1"] > 0 and ex["answer_1"]:  # answer_1 is a positive example
                pos_ind = 1

            if pos_ind is not None:
                input = wrap_text_with_prompts(ex, pos_ind)
                output = wrap_answer(ex[f"answer_{pos_ind}"]) + f" {eos_token}"

                yield {
                    "input": input,
                    "output": output,
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
