
import numpy as np
import cv2
import torch
from typing import List, Dict, Optional, Sequence, Tuple

from simpler_env.policies.base import AiroaBasePolicy
from g3_haptics.utils.lerobot_dataset_utils import create_g3multi_embodiment
from g3_haptics.datasets.embodiment import EmbodimentTag

class G3Pi0mutiLerobotToAiroaPolicy(AiroaBasePolicy):
    def __init__(
        self,
        policy,
        dataset_cfg=None,
        policy_setup: str = "widowx_bridge",
    ):
        self.policy = policy
        self.policy.eval()
        if policy_setup == "widowx_bridge":
            h, w = 256, 256
            self.image_size = (w, h)  # cv2.resize は (W, H)
        elif policy_setup == "google_robot":
            self.image_size = (320, 256)
            
        self.embodiment = (
            create_g3multi_embodiment(dataset_cfg) if dataset_cfg is not None else None
        )
        
        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.tag = 0 
        else:
            self.tag = 1        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def step(self, obs: Dict) -> Dict:
        image = self._resize_image(obs["observation.images.image"])
        prompt = obs["task"]
        state = obs["observation.state"]

        observation = {
            "observation.state": obs["observation.state"],
            "observation.images.image": torch.from_numpy(image / 255.)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            .float(),
            "task": obs["task"],
        }

        if self.embodiment is not None:
            device = next(
                (v.device for v in observation.values() if torch.is_tensor(v)), None
            )
            tag = (
                torch.tensor(self.tag, dtype=torch.long, device=device)
                if device is not None
                else torch.tensor(self.tag, dtype=torch.long)
            )
            observation = self.embodiment.pad_item(
                observation
                | {
                    # FIXME: We cannot know embodiment
                    EmbodimentTag: tag
                }
            )

        with torch.inference_mode():
            actions = self.policy.select_action(observation)

        return actions

    def reset(self) -> None:
        self.policy.reset()

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
        return image