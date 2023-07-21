import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from peft import (
    PeftModel,
    get_peft_model,
    prepare_model_for_int8_training,
    LoraConfig,
    PeftType,
    PromptLearningConfig,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from huggingface_hub import hf_hub_download
from xgpt.utils.deepspeed import enable_transformers_pretrained_deepspeed_sharding
from xgpt.core.ppo_config import PPOConfig
from xgpt.utils.ppo import (
    AdaptiveKLController,
    FixedKLController,
    clip_by_value,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
)
from xgpt.utils.prompt_templates import process_query_response

logger = logging.getLogger(__name__)


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(
                            module_name, f"{module_name}.modules_to_save.{adapter_name}"
                        )
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(
                        suffix_to_replace, f"{adapter_name}.{suffix_to_replace}"
                    )
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif (
        isinstance(config, PromptLearningConfig)
        or config.peft_type == PeftType.ADAPTION_PROMPT
    ):
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    model.load_state_dict(peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )


class MLPCritic(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "critic_dropout_prob"):
            critic_dropout_prob = kwargs.pop("critic_dropout_prob", 0.1)
        else:
            critic_dropout_prob = config.critic_dropout_prob
        self.dropout = (
            nn.Dropout(critic_dropout_prob) if critic_dropout_prob else nn.Identity()
        )
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size
        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        # Keep the output in fp32 for numerical stability
        if output.dtype != self.linear.weight.dtype:
            output = output.to(self.linear.weight.dtype)
        output = self.linear(output).squeeze(-1)
        return output


class PPOTransformer(pl.LightningModule):
    def __init__(
        self,
        actor: DictConfig,
        reward_model: DictConfig,
        mean_reward: float = 0.0,
        rm_tokenizer: Optional[DictConfig] = None,
        critic: Optional[DictConfig] = None,
        optimizer: DictConfig = None,
        scheduler: DictConfig = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        scheduler_monitor: str = "reward",
        scheduler_mode: str = "max",
        deepspeed_sharding: bool = True,
        gradient_checkpointing: bool = False,
        is_encoder_decoder: bool = False,
        ppo_config: DictConfig = None,
        use_peft: bool = False,
        init_adapter_path: str = None,
        rm_adapter_path: str = None,
        generate_references: bool = False,
        stop_sequence: str = '"""',
        lora_config: DictConfig = None,
        generation_kwargs: DictConfig = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.actor_cfg = actor
        self.critic_cfg = critic
        self.reward_model_cfg = reward_model
        self.mean_reward = mean_reward
        self.rm_tokenizer = rm_tokenizer
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.tokenizer = tokenizer
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_mode = scheduler_mode
        self.gradient_checkpointing = gradient_checkpointing
        self.deepspeed_sharding = deepspeed_sharding
        self.use_peft = use_peft
        self.init_adapter_path = init_adapter_path
        self.rm_adapter_path = rm_adapter_path
        self.generate_references = generate_references
        self.lora_config = lora_config
        self.generation_kwargs = dict(generation_kwargs)
        self.stop_sequence = stop_sequence
        self._automatic_optimization = False
        self._grad_accumu_count = 0
        # PPO attributes
        self.ppo_config: PPOConfig = instantiate(ppo_config)
        self.gradient_accumulation_steps = self.ppo_config.gradient_accumulation_steps
        self.is_encoder_decoder = is_encoder_decoder
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        if self.ppo_config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(
                self.ppo_config.init_kl_coef,
                self.ppo_config.target,
                self.ppo_config.horizon,
            )
        else:
            self.kl_ctl = FixedKLController(self.ppo_config.init_kl_coef)

        self.evaluation_texts = {}

    def setup(self, stage: Optional[str] = None) -> None:
        if self.rm_tokenizer is not None:
            self.rm_tokenizer = instantiate(self.rm_tokenizer)

        if self.deepspeed_sharding and self.trainer.num_devices > 1:
            enable_transformers_pretrained_deepspeed_sharding(self)

        self.initialize_models()

    def initialize_models(self):
        actor_device_map = None
        rm_device_map = None
        if self.actor_cfg.load_in_8bit or self.actor_cfg.load_in_4bit:
            actor_device_map = {"": self.trainer.local_rank}
        if self.reward_model_cfg.load_in_8bit or self.reward_model_cfg.load_in_4bit:
            rm_device_map = {"": self.trainer.local_rank}
        actor_model = instantiate(self.actor_cfg, device_map=actor_device_map)
        actor_model.resize_token_embeddings(len(self.tokenizer.vocab))
        if self.use_peft:
            lora_config = instantiate(self.lora_config)
            actor_model = prepare_model_for_int8_training(
                actor_model, use_gradient_checkpointing=self.gradient_checkpointing
            )
            actor_model.enable_input_require_grads()
            if self.init_adapter_path is None:
                self.actor_model = get_peft_model(actor_model, lora_config)
                self.ppo_adapter_name = "default"
            else:
                self.actor_model = PeftModel.from_pretrained(
                    actor_model, self.init_adapter_path
                )
            self.ref_model = self.actor_model
        else:
            self.actor_model = actor_model
            self.ref_model = instantiate(self.actor_cfg, device_map=actor_device_map)
            self.ref_model.resize_token_embeddings(len(self.tokenizer_vocab))
            self.ref_model.requires_grad_(False)
        if self.critic_cfg is None:
            self.critic_model = MLPCritic(self.actor_model.config)
        if (
            self.rm_adapter_path is not None
        ):  # load RM as another set of adapter to the SFT model
            assert self.use_peft
            self.add_rm_adapter()
            self.reward_model = None
        else:  # use stand-alone RM
            self.reward_model = instantiate(
                self.reward_model_cfg, device_map=rm_device_map
            )
            self.reward_model.requires_grad_(False)

        if self.gradient_checkpointing:
            self.actor_model.gradient_checkpointing_enable()
            self.actor_model.config.use_cache = False

        self.unify_special_token_ids()

    def add_rm_adapter(self, adapter_name="reward_model_adapter"):
        rm_adapter_peft_config = LoraConfig.from_pretrained(self.rm_adapter_path)
        self.actor_model.add_adapter(adapter_name, rm_adapter_peft_config)
        self.rm_adapter_name = adapter_name

        filename = os.path.join(self.rm_adapter_path, "adapter_model.bin")
        if not os.path.exists(filename):
            adapter_filename = hf_hub_download(
                self.rm_adapter_path, "adapter_model.bin"
            )
        else:
            adapter_filename = filename

        adapter_state_dict = torch.load(adapter_filename, map_location="cpu")

        score_dict = {}
        copy_adapter_state_dict = adapter_state_dict.copy()

        for name, _ in copy_adapter_state_dict.items():
            if "score" in name:
                key_name = ".".join(name.split(".")[-1:])
                score_dict[key_name] = adapter_state_dict.pop(name)

        num_labels, hidden_dim = score_dict["weight"].shape
        has_bias = any(["bias" in name for name in adapter_state_dict.keys()])

        self.score = nn.Linear(hidden_dim, num_labels, bias=has_bias)
        self.score.load_state_dict(score_dict)
        self.score.requires_grad_(False)

        set_peft_model_state_dict(
            self.actor_model, adapter_state_dict, adapter_name=adapter_name
        )

    def unify_special_token_ids(self):
        for key in self.tokenizer.special_tokens_map:
            id_name = f"{key}_id"
            id_in_tokenizer = getattr(self.tokenizer, id_name)

            for model_name in ["actor_model", "ref_model"]:
                model = getattr(self, model_name)
                if hasattr(model.config, id_name):
                    id_in_config = getattr(model.config, id_name)
                    if id_in_config != id_in_tokenizer:
                        logger.warn(
                            f"{id_name} in {model_name}.config is {id_in_config}, but"
                            f" in tokenizer is {id_in_tokenizer}! Setting it to"
                            f" {id_in_tokenizer} for avoiding unpredictable behaviours."
                        )
                        setattr(model.config, id_name, id_in_tokenizer)

                if hasattr(model.generation_config, id_name):
                    id_in_gen_config = getattr(model.generation_config, id_name)
                    if id_in_gen_config != id_in_tokenizer:
                        logger.warn(
                            f" {id_name} in {model_name}.geneartion_config is {id_in_gen_config}, but"
                            f" in tokenizer is {id_in_tokenizer}! Setting it to"
                            f" {id_in_tokenizer} for avoiding unpredictable behaviours. "
                        )
                        setattr(model.generation_config, id_name, id_in_tokenizer)

    def configure_optimizers(self) -> Dict:
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = instantiate(self.optimizer_cfg, parameters)
        scheduler = instantiate(self.scheduler_cfg, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": self.scheduler_monitor,
                "strict": True,
                "name": None,
            },
        }

    def reward_model_forward(self, query_response_tensor):
        if self.reward_model is not None:
            return self.reward_model(query_response_tensor).logits
        else:
            self.actor_model.set_adapter(self.rm_adapter_name)
            outputs = self.actor_model(query_response_tensor, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            logits = self.score(hidden_states)

            batch_size = query_response_tensor.shape[0]
            sequence_lengths = (
                torch.ne(query_response_tensor, self.tokenizer.pad_token_id).sum(-1) - 1
            ).to(self.device)

            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]
            self.actor_model.set_adapter(self.ppo_adapter_name)

            return pooled_logits

    @torch.no_grad()
    def batch_rollout(self, prefix, batch):
        batch_size = batch["input_ids"].size(0)
        disable_tqdm = self.local_rank != 0
        queries = []
        responses = []
        rewards = []
        ref_rewards = []

        query_texts = []
        response_texts = []
        ref_response_texts = []
        mini_batch_start_idx = list(
            range(0, batch_size, self.ppo_config.rollout_batch_size)
        )
        for start_idx in tqdm(
            mini_batch_start_idx,
            desc="Batch rollout",
            total=len(mini_batch_start_idx),
            disable=disable_tqdm,
        ):
            query_tensor = batch.get("input_ids")[
                start_idx : start_idx + self.ppo_config.rollout_batch_size
            ]
            response_tensor = self.actor_model.generate(
                input_ids=query_tensor, **self.generation_kwargs
            )
            (
                query,
                response,
                response_tensor,
                query_response_tensor_rm,
            ) = process_query_response(
                query_tensor,
                response_tensor,
                self.tokenizer,
                stop=self.stop_sequence,
                current_device=self.device,
                rm_tokenizer=self.rm_tokenizer,
            )

            reward = (
                self.reward_model_forward(query_response_tensor_rm) - self.mean_reward
            )

            queries.extend(query_tensor)
            responses.extend(response_tensor)
            rewards.extend(reward.squeeze(-1))

            query_texts.extend(query)
            response_texts.extend(response)

            if self.generate_references:
                if hasattr(self.ref_model, "disable_adapter"):
                    with self.ref_model.disable_adapter():
                        ref_response_tensor = self.ref_model.generate(
                            input_ids=query_tensor, **self.generation_kwargs
                        )
                else:
                    ref_response_tensor = self.ref_model.generate(
                        input_ids=query_tensor, **self.generation_kwargs
                    )
                (
                    _,
                    ref_response,
                    _,
                    query_ref_response_tensor_rm,
                ) = process_query_response(
                    query_tensor,
                    ref_response_tensor,
                    self.tokenizer,
                    stop=self.stop_sequence,
                    current_device=self.device,
                    rm_tokenizer=self.rm_tokenizer,
                )

                ref_reward = (
                    self.reward_model_forward(query_ref_response_tensor_rm)
                    - self.mean_reward
                )
                ref_rewards.extend(ref_reward.squeeze(-1))
                ref_response_texts.extend(ref_response)

        # log reward and texts
        stacked_reward = torch.stack(rewards)
        mean_reward = stacked_reward.mean()
        std_reward = stacked_reward.std()
        self.log_dict(
            {
                f"training/{prefix}_reward": mean_reward,
                f"training/{prefix}_reward_std": std_reward,
            },
            sync_dist=True,
        )

        if self.generate_references:
            stacked_ref_reward = torch.stack(ref_rewards)
            columns = ["query", "response", "reward", "ref_response", "ref_reward"]
            data = [
                row
                for row in zip(
                    query_texts,
                    response_texts,
                    stacked_reward.cpu().tolist(),
                    ref_response_texts,
                    stacked_ref_reward.cpu().tolist(),
                )
            ]
        else:
            columns = ["query", "response", "reward"]
            data = [
                row
                for row in zip(
                    query_texts,
                    response_texts,
                    stacked_reward.cpu().tolist(),
                )
            ]

        if prefix == "train":
            if isinstance(self.logger, WandbLogger):
                self.logger.log_text(
                    key=f"{prefix}_samples", columns=columns, data=data
                )
        else:
            if not self.evaluation_texts:
                self.evaluation_texts = {
                    "columns": columns,
                    "data": data,
                }
            else:
                self.evaluation_texts["data"].extend(data)

        return (queries, responses, rewards)

    def _step_safety_checker(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        for name, tensor_list in zip(
            ["queries" "responses" "scores"], [queries, responses, scores]
        ):
            if not isinstance(tensor_list, list):
                raise ValueError(
                    f"{name} must be a list of tensors - got {type(tensor_list)}"
                )
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(
                    f"Elements in {name} must tensors - got {type(tensor_list[0])}"
                )

        # put queries, scores and responses on the correct device
        queries = [tensor.to(self.device) for tensor in queries]
        responses = [tensor.to(self.device) for tensor in responses]
        scores = [tensor.to(self.device) for tensor in scores]

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(
                    f"Scores must be 1-dimensional - got {score.dim()} for {score}"
                )
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator()
            [
                {"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries
            ].to(self.device)

            decoder_inputs = self.data_collator(
                [
                    {"input_ids": r, "attention_mask": torch.ones_like(r)}
                    for r in responses
                ]
            ).to(self.device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            self.tokenizer.padding_side = "left"
            input_data = self.data_collator(
                [
                    {
                        "input_ids": ids,
                        "attention_mask": (ids != self.tokenizer.pad_token_id).int(),
                    }
                    for ids in input_ids
                ]
            ).to(self.device)

        input_data.pop("labels", None)  # we don't want to compute LM losses

        return input_data

    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        loss_p, loss_v, train_stats, early_stop = self.loss(
            old_logprobs, values, rewards, logits, vpreds, logprobs, mask
        )
        loss = loss_p + loss_v

        updated = False
        opt = self.optimizers()
        if not early_stop:
            self.manual_backward(loss / self.gradient_accumulation_steps)
            self._grad_accumu_count += 1

            if self._grad_accumu_count % self.gradient_accumulation_steps == 0:
                if self.ppo_config.max_grad_norm is not None:
                    opt.clip_grad = self.ppo_config.max_grad_norm

                opt.step()
                opt.zero_grad()
                self._grad_accumu_count = 0
                updated = True

        if early_stop:
            opt.zero_grad()
        return train_stats, early_stop, updated

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        rewards, non_score_rewards = [], []
        kls = []
        for score, logprob, ref_logprob, mask in zip(
            scores, logprobs, ref_logprobs, masks
        ):
            kl = logprob - ref_logprob
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]
            reward[last_non_masked_index] += score
            rewards.append(reward)
        kls = torch.stack(kls)
        mean_kl = masked_mean(kls, masks)
        self.log("policy/refkl", mean_kl, sync_dist=True)
        return torch.stack(rewards), torch.stack(non_score_rewards)

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.ppo_config.gamma * nextvalues - values[:, t]
            lastgaelam = (
                delta + self.ppo_config.gamma * self.ppo_config.lam * lastgaelam
            )
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds,
            values - self.ppo_config.cliprange_value,
            values + self.ppo_config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).double(), mask)
        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.ppo_config.cliprange, 1.0 + self.ppo_config.cliprange
        )

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).double(), mask)

        loss = pg_loss + self.ppo_config.vf_coef * vf_loss

        entropy = masked_mean(entropy_from_logits(logits), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        if self.trainer.num_devices > 1:
            import torch.distributed as dist

            dist.barrier()
            dist.all_reduce(approxkl, dist.ReduceOp.SUM)
            approxkl /= (
                self.trainer.num_devices
            )  # this is to guarantee all processes early stop together
        policykl = masked_mean(old_logprobs - logprobs, mask)
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        early_stop = False
        if approxkl > self.ppo_config.target_kl:
            early_stop = True

        stats = dict(
            loss=dict(
                policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()
            ),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            value=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )

        # log stats
        flat_stats = flatten_dict(stats)
        self.log_dict(flat_stats, sync_dist=True)
        return (
            pg_loss,
            self.ppo_config.vf_coef * vf_loss,
            flat_stats,
            early_stop,
        )

    def batch_forward_pass(
        self,
        model,
        responses: torch.Tensor,
        model_inputs: dict,
        return_all_logits: bool = False,
    ):
        bs = len(responses)
        fbs = self.ppo_config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs]
                for key, value in model_inputs.items()
            }
            response_batch = responses[i * fbs : (i + 1) * fbs]
            outputs = model(**input_kwargs, output_hidden_states=True)
            logits = outputs.logits
            last_hidden = outputs.hidden_states[-1]
            values = self.critic_model(last_hidden)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            query_masks = torch.zeros_like(attention_mask)
            query_masks[:, :-1] = attention_mask[:, 1:]

            for j in range(fbs):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                    query_masks[j, :start] = 0
                    query_masks[j, end:] = 0
                else:
                    query_masks[j, : -len(response_batch[j]) - 1] = 0

                if len(response_batch[j]) < 2:
                    raise ValueError(
                        "Responses are too short. Make sure they are at least 4 tokens long."
                    )

            if return_all_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(query_masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_all_logits else all_logits,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def optimize(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        queries, responses, scores = self._step_safety_checker(
            queries, responses, scores
        )

        model_inputs = self.prepare_model_inputs(queries, responses)
        model_inputs_names = list(model_inputs.keys())

        if self.use_peft and self.gradient_checkpointing:
            self.actor_model.gradient_checkpointing_disable()
            self.actor_model.config.use_cache = True

        with torch.no_grad():
            all_logprobs, _, values, masks = self.batch_forward_pass(
                self.actor_model, responses, model_inputs
            )

            if hasattr(self.ref_model, "disable_adapter"):
                with self.ref_model.disable_adapter():
                    ref_logprobs, _, _, _ = self.batch_forward_pass(
                        self.ref_model, responses, model_inputs
                    )
            else:
                ref_logprobs, _, _, _ = self.batch_forward_pass(
                    self.ref_model, responses, model_inputs
                )

        if self.use_peft and self.gradient_checkpointing:
            self.actor_model.gradient_checkpointing_enable()
            self.actor_model.config.use_cache = False
        rewards, _ = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
        self.log("training/kl_coef", self.kl_ctl.value)

        mini_batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs,
            "values": values,
            "rewards": rewards,
            "masks": masks,
        }

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["queries", "responses"]:
                    return_dict[key] = [d[key] for d in data]
                else:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(
                        self.device
                    )
            return return_dict

        mini_batch_dict.update(model_inputs)
        mini_batch_data = Dataset.from_dict(mini_batch_dict)
        mini_batch_data.set_format("torch")
        mini_batch_dataloader = torch.utils.data.DataLoader(
            mini_batch_data,
            batch_size=self.ppo_config.mini_batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        approxkls = []
        target_updates = self.ppo_config.ppo_epochs * len(mini_batch_dataloader)
        actual_updates = 0
        for _ in range(self.ppo_config.ppo_epochs):
            for batch in mini_batch_dataloader:
                model_inputs = {k: batch[k] for k in model_inputs_names}
                logprobs, logits, vpreds, _ = self.batch_forward_pass(
                    self.actor_model,
                    batch["responses"],
                    model_inputs,
                    return_all_logits=True,
                )
                train_stats, early_stop, updated = self.train_minibatch(
                    batch["logprobs"],
                    batch["values"],
                    batch["rewards"],
                    logprobs,
                    logits,
                    vpreds,
                    batch["masks"],
                )
                approxkls.append(train_stats["policy/approxkl"])
                if updated:
                    actual_updates += self.gradient_accumulation_steps

                if early_stop:
                    break
            if early_stop:
                break

        self.log("training/update_ratio", actual_updates / target_updates)
        # Update the KL control - multiply the batch_size by the number of processes
        if approxkls:
            mean_approxkl = torch.stack(approxkls).mean().cpu().numpy()
            self.kl_ctl.update(mean_approxkl, len(rewards) * self.trainer.num_devices)

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        # rollout
        if self.use_peft and self.gradient_checkpointing:
            self.actor_model.gradient_checkpointing_disable()
            self.actor_model.config.use_cache = True

        rollout_data = self.batch_rollout(prefix, batch)

        if self.use_peft and self.gradient_checkpointing:
            self.actor_model.gradient_checkpointing_enable()
            self.actor_model.config.use_cache = False
        # calculate losses and optimize
        if prefix == "train":
            self.optimize(*rollout_data)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.common_step("train", batch)

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.common_step("val", batch)

    def on_validation_epoch_end(self):
        self.log_validation_test_texts("val")

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.common_step("test", batch)

    def on_test_epoch_end(self):
        self.log_validation_test_texts("test")

    def log_validation_test_texts(self, prefix):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_text(key=f"{prefix}_samples", **self.evaluation_texts)
        self.evaluation_texts = {}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer.local_rank == 0:  # main process
            hf_dir = f"{self.trainer.default_root_dir}/huggingface"
            # remove if exist:
            if os.path.isdir(hf_dir):
                shutil.rmtree(hf_dir)
            # save new
            self.actor_model.save_pretrained(hf_dir)
            self.tokenizer.save_pretrained(hf_dir)
        return checkpoint

    def state_dict(self, *args, **kwargs):
        kwargs.pop("prefix", None)
        states = super().state_dict(*args, **kwargs)
        keys_to_pop = []
        for key in states:  # only keep actor and critic models
            if not key.startswith("actor_model") and not key.startswith("critic_model"):
                keys_to_pop.append(key)
        for key in keys_to_pop:
            del states[key]
        return states
