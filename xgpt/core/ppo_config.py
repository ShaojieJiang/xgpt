from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig(object):
    def __init__(
        self,
        adap_kl_ctrl: Optional[bool] = True,
        init_kl_coef: Optional[float] = 0.2,
        target: Optional[float] = 6,
        target_kl: Optional[float] = 0.015,
        horizon: Optional[float] = 10000,
        gamma: Optional[float] = 0.99,
        lam: Optional[float] = 0.95,
        cliprange: Optional[float] = 0.2,
        cliprange_value: Optional[float] = 0.2,
        vf_coef: Optional[float] = 0.1,
        rollout_batch_size: int = 4,
        mini_batch_size: Optional[int] = 1,
        ppo_epochs: Optional[int] = 4,
        gradient_accumulation_steps: Optional[int] = 8,
        max_grad_norm: Optional[float] = None,
    ):
        self.adap_kl_ctrl = adap_kl_ctrl
        self.init_kl_coef = init_kl_coef
        self.target = target
        self.target_kl = target_kl
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.cliprange = cliprange
        self.cliprange_value = cliprange_value
        self.vf_coef = vf_coef
        self.rollout_batch_size = rollout_batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
