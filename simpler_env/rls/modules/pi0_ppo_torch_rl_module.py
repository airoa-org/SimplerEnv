from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule


class Pi0PPOTorchRLModule(TorchRLModule, DefaultPPORLModule):
    """Pi0 PPO RL Module for RLlib."""

    @override(RLModule)
    def setup(self):
        
        # TODO ここをpi0に置き換える

        assert len(self.action_space.shape) == 1
        action_dim = self.action_space.shape[0]

        self.pi = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim*2),
        )

        self.vf = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    @override(RLModule)
    def get_initial_state(self) -> dict:
        return {}

    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output = {}
        
        output[Columns.ACTION_DIST_INPUTS] = self.pi(
            batch[Columns.OBS].permute(0, 3, 1, 2),
        ) # (batch, 2*action_dim)
        return output

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output = {}
        output[Columns.ACTION_DIST_INPUTS] = self.pi(
            batch[Columns.OBS].permute(0, 3, 1, 2),
        )
        return output

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        vf_out = self.vf(
            batch[Columns.OBS].permute(0, 3, 1, 2),
        ) # (batch_size?, 1)
        return vf_out.squeeze(-1)
