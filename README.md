# XGPT: A toolkit for training GPT-like SFT, RM, and PPO models

Features:
  - One-line of command for running training/evaluation
  - A simple framework for adding essential processing steps, from data preparation to training logit
  - Supports PEFT and [Multi Adapter](https://huggingface.co/docs/trl/multi_adapter_rl), allowing for training large-sized LMs on consumer hardware at _every_ phase
  - Easy to configure
  - Minimalism: friendly for studying, yet powerful enough for dealing with real-world trainings

## Setup

Run `pip install -e .`

Copy `xgpt/conf/local_conf_example.yaml`  to `xgpt/conf/local_conf.yaml`, and change the values accordingly

## Usage

### Run training

For running training at each phase, then simply run the command:

`python main.py --config-name rlhf_[sft | rm | ppo]`

### Run validation/test

For evaluating the trained models on validation/test sets of the supported phases, run
`python main.py --config-name rlhf_[sft | rm | ppo] stage=[validate | test]`


## Acknowledgement

These are the essential components or predecessors of XGPT:
* Pytorch Lightning, Lightning Transformers
* Hugging Face Transformers & Datasets & TRL
