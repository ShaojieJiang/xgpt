"""
Merge the trained adapters with base models so that they can be used the same way.
"""

import logging

import hydra
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@hydra.main(config_path="./conf", config_name="merge_adapter", version_base="1.2.0")
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    logging.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_path)

    logging.info("Initialising PEFT model...")
    peft_model: PeftModel = PeftModel.from_pretrained(model, cfg.adapter_path)

    logging.info("Merging the adapter with base model...")
    peft_model.merge_and_unload()

    logging.info("Saving merged weights...")
    peft_model.base_model.model.save_pretrained(cfg.dst_path)
    tokenizer.save_pretrained(cfg.dst_path)

    logging.info("Done!")


if __name__ == "__main__":
    main()
