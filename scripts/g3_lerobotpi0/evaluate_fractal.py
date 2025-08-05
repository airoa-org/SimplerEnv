import argparse
from typing import Dict
import time

import torch
import numpy as np
import cv2 as cv

from simpler_env.evaluation.adapter import AiroaToSimplerFractalAdapter
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy
from simpler_env.utils.geometry import euler2axangle
from simpler_env.utils.action.action_ensemble import ActionEnsembler


def auto_model_fn(path):
    import sys; sys.path.append(path) # noqa
    from modeling_pi0 import PI0Policy
    return PI0Policy


class FractalLerobotPi0ToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy
        self.policy.eval()
        self.pred_action_horizon = 4
        self.action_ensemble = True
        self.action_ensemble_temp = -0.8
        self.image_size = (224, 224)

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def step(self, obs: Dict) -> Dict:
        image = self._resize_image(obs["image"])
        prompt = obs["prompt"]
        state = obs["state"]

        obs_lerobotpi0 = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).float().to(self.device),
            "observation.images.image": torch.from_numpy(image / 255).permute(2, 0, 1).unsqueeze(0).float().to(self.device),
            "task": [prompt], 
        }

        with torch.inference_mode():
            actions = self.policy.select_action(obs_lerobotpi0)[0][:self.pred_action_horizon].cpu().numpy()

        if self.action_ensemble:
            actions = self.action_ensembler.ensemble_action(actions)[None][0]
        else:
            actions = actions[0]

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


class AiroaToSimplerFractalStickyActionAdapter(AiroaToSimplerFractalAdapter):
    def __init__(self, policy):
        super().__init__(policy)
        self.sticky_gripper_num_repeat = 10 # same to lerobotpi0

    def reset(self, task_description):
        super().reset(task_description)
        self.previous_gripper_action = None
    
    def postprocess(self, outputs: Dict) -> Dict:
        action = outputs["actions"]
        roll, pitch, yaw = action[3:6]
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

        current_gripper_action = action[-1]

        if self.previous_gripper_action is None:
            relative_gripper_action = 0
            self.previous_gripper_action = current_gripper_action
        else:
            relative_gripper_action = self.previous_gripper_action - current_gripper_action

        # switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action
            self.previous_gripper_action = current_gripper_action

        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        action = np.concatenate(
            [
                action[:3],
                action_rotation_ax * action_rotation_angle,
                [relative_gripper_action],
            ]
        )

        return {
            "actions": action,
            "terminate_episode": outputs["terminate_episode"],
        }


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

    env_policy = AiroaToSimplerFractalStickyActionAdapter(policy=policy)

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=env_policy, ckpt_path=ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
