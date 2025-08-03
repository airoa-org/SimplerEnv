import argparse
from typing import Dict

import torch
import numpy as np
import cv2 as cv

from simpler_env.evaluation.adapter import AiroaToSimplerFractalAdapter
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy


def auto_model_fn(path):
    import sys; sys.path.append(path) # noqa
    from modeling_pi0 import PI0Policy
    return PI0Policy


class FractalLerobotPi0ToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy
        self.pred_action_horizon = 4
        self.image_size = (256, 256)

    def step(self, obs: Dict) -> Dict:
        image = self._resize_image(obs["image"])
        prompt = obs["prompt"]
        state = obs["state"]

        obs_lerobotpi0 = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).float(),
            "observation.images.image": torch.from_numpy(image / 255).permute(2, 0, 1).unsqueeze(0).float(),
            "task": [prompt], 
        }

        actions = self.policy.select_action(obs_lerobotpi0)[0][:self.pred_action_horizon].numpy()[0]

        outputs = {
            "actions": actions,
            "terminate_episode": np.zeros(actions.shape[0]),
        }

        return outputs

    def reset(self) -> None:
        self.policy.reset()

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    PI0Policy = auto_model_fn(ckpt_path)
    policy = FractalLerobotPi0ToAiroaPolicy(
        policy=PI0Policy.from_pretrained(ckpt_path)
    )

    env_policy = AiroaToSimplerFractalAdapter(policy=policy)

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=env_policy, ckpt_path=ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
