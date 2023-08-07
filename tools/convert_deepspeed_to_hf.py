"""
Checkpoints saved using DeepSpeed need some processing before they can be used as HF models.
"""

import os

import hydra
import torch
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_model_state_file,
    get_optim_files,
)
from hydra.utils import instantiate
from lightning_fabric.utilities.types import _PATH
from omegaconf import DictConfig
from pytorch_lightning.utilities.deepspeed import ds_checkpoint_dir
from transformers import PreTrainedTokenizer

from xgpt.core.model import TaskTransformer

CPU_DEVICE = torch.device("cpu")


def convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir: _PATH, tag: str = None
) -> None:
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    deepspeed_states = [
        "module",
        "optimizer",
        "lr_scheduler",
        "csr_tensor_module_names",
        "skipped_steps",
        "global_steps",
        "dp_world_size",
        "mp_world_size",
    ]
    checkpoint_dir = ds_checkpoint_dir(checkpoint_dir)
    optim_files = get_optim_files(checkpoint_dir)
    optim_state = torch.load(optim_files[0], map_location=CPU_DEVICE)
    zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
    model_file = get_model_state_file(checkpoint_dir, zero_stage)
    client_state = torch.load(model_file, map_location=CPU_DEVICE)
    client_state = {
        key: value for key, value in client_state.items() if key not in deepspeed_states
    }
    # Delete 'module' prefix before saving.
    state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
    client_state["state_dict"] = state_dict

    return client_state


@hydra.main(
    config_path="./conf", config_name="convert_deepspeed_to_hf", version_base="1.2.0"
)
def main(cfg: DictConfig):
    # initialize tokenizer and model
    tokenizer: PreTrainedTokenizer = instantiate(cfg.tokenizer)
    model: TaskTransformer = instantiate(cfg.model, tokenizer=tokenizer)
    model.initialize_model()
    ds_ckpt_path = None

    for name in os.walk(cfg.ckpt_path):
        full_path = name[0]
        if os.path.isdir(full_path) and os.path.isfile(
            os.path.join(full_path, "latest")
        ):
            ds_ckpt_path = full_path
            break

    # load deepspeed ckpt weights and convert names
    states = convert_zero_checkpoint_to_fp32_state_dict(ds_ckpt_path)["state_dict"]

    # override weights and save to HF format
    model.load_state_dict(states)
    target_path = os.path.join(cfg.ckpt_path, "huggingface")
    print(f"Saving model and tokenizer to {target_path}")
    model.save_hf_checkpoint(target_path)
    tokenizer.save_pretrained(target_path)
    print("Done!")


if __name__ == "__main__":
    main()
