from abc import abstractmethod
from typing import Dict

import numpy as np


class AiroaBasePolicy:
    @abstractmethod
    def infer(self, obs: Dict) -> Dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class OpenpiToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy

    def infer(self, obs: Dict) -> Dict:
        outputs = self.policy.infer(obs)
        outputs["terminate_episode"] = np.zeros(outputs["actions"].shape[0])
        return outputs

    def reset(self) -> None:
        self.policy.reset()
