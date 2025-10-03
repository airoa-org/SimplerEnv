from typing import List, Optional
import os
from collections import deque

import torch
import numpy as np
from PIL import Image
from transforms3d.euler import euler2axangle

from simpler_env.policies.lerobotpi.geometry import mat2euler, quat2mat
from simpler_env.policies.lerobotpi.pi0_or_fast import LerobotPiFastInference, auto_model_fn
from simpler_env.utils.action.action_ensemble import ActionEnsembler


class G3LerobotPiFastInference(LerobotPiFastInference):
    def __init__(
        self,
        saved_model_path: str = "pretrained/pi0",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon: int = 4,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        action_ensemble: bool = True,
        action_ensemble_temp: float = -0.8,
        sticky_action: bool = True,
        n_action_steps: int = 4,
    ) -> None:
        gpu_idx = os.environ.get("GPU_IDX", 0)
        self.device = f"cuda:{gpu_idx}"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig/1.0.0" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data/0.1.0" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.sticky_action = sticky_action
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        # TODO: add pi0 loading ...
        PI0Policy = auto_model_fn(saved_model_path)
        self.vla = PI0Policy.from_pretrained(saved_model_path)
        self.vla.model.paligemma_with_expert.paligemma.language_model = self.vla.model.paligemma_with_expert.paligemma.language_model.model
        self.vla.model.paligemma_with_expert.gemma_expert.model = self.vla.model.paligemma_with_expert.gemma_expert.model.base_model
        self.vla.config.n_action_steps = n_action_steps
        self.vla.to(self.device)
        self.vla.reset()

        self.image_size = image_size
        self.action_scale = action_scale
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = self.vla.config.n_action_steps
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None


    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images: List[Image.Image] = self._obtain_image_history()

        eef_pos = kwargs.get("eef_pos", None)

        state = self.preprocess_widowx_proprio(eef_pos)
        observation = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device).float(),
            "observation.images.image_0": torch.from_numpy(images[0] / 255).permute(2, 0, 1).unsqueeze(0).to(self.device).float(),
            "observation.images.image_1": torch.from_numpy(images[0] / 255).permute(2, 0, 1).unsqueeze(0).to(self.device).float(),
            "observation.images.image_2": torch.from_numpy(images[0] / 255).permute(2, 0, 1).unsqueeze(0).to(self.device).float(),
            "observation.images.image_3": torch.from_numpy(images[0] / 255).permute(2, 0, 1).unsqueeze(0).to(self.device).float(),
            "task": [task_description],
        }

        actions = self.vla.select_action(observation)[0].cpu().numpy()

        if self.action_ensemble:
            action_chunk = [actions]
            for _ in range(self.vla.config.n_action_steps-1):
                actions = self.vla.select_action(observation)[0].cpu().numpy()
                action_chunk.append(actions)
            action_chunk = np.stack(action_chunk, axis=0)
            actions = self.action_ensembler.ensemble_action(action_chunk)[None][0]

        raw_action = {
            "world_vector": np.array(actions[:3]),
            "rotation_delta": np.array(actions[3:6]),
            "open_gripper": np.array(actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            if self.sticky_action:
                action["gripper"] = 0
                current_gripper_action = raw_action["open_gripper"]
                if self.previous_gripper_action is None:
                    relative_gripper_action = np.array([0])
                    self.previous_gripper_action = current_gripper_action
                else:
                    relative_gripper_action = self.previous_gripper_action - current_gripper_action

                # fix a bug in the SIMPLER code here
                # self.previous_gripper_action = current_gripper_action

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

                action["gripper"] = relative_gripper_action
            else:
                current_gripper_action = raw_action["open_gripper"]
                current_gripper_action = (current_gripper_action * 2) - 1
                current_gripper_action = - current_gripper_action
                action["gripper"] = current_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action
