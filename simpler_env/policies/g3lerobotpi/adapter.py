from collections import deque
import os
from typing import List, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
import logging
import torch
from PIL import Image

from simpler_env.utils.action.action_ensemble import ActionEnsembler

from simpler_env.utils.geometry import euler2axangle, mat2euler, quat2mat

from simpler_env.policies.g3lerobotpi.geometry import quat_to_rot6d

class BaseAdapter:
    def __init__(self, policy):
        self.policy = policy

    def reset(self, task_description):
        pass

    def preprocess(self, image: np.ndarray, eef_pos: np.ndarray, prompt: str) -> Dict:
        pass

    def postprocess(self, outputs: Dict) -> Dict:
        pass

    def step(self, image: np.ndarray, eef_pos: np.ndarray, prompt: Optional[str] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        inputs = self.preprocess(image, eef_pos, prompt)
        outputs = self.policy.step(inputs)
        state_gripper = inputs["state"][-1]
        action_gripper = outputs["actions"][-1]
        # print(f"state: {state_gripper} action: {action_gripper}")
        final_outputs = self.postprocess(outputs)
        simpler_outputs = {
            "world_vector": outputs["actions"][:3],
            "rot_axangle": outputs["actions"][3:6],
            "gripper": outputs["actions"][6:],
            "terminate_episode": outputs["terminate_episode"],
        }
        final_simpler_outputs = {
            "world_vector": final_outputs["actions"][:3],
            "rot_axangle": final_outputs["actions"][3:6],
            "gripper": final_outputs["actions"][6:],
            "terminate_episode": final_outputs["terminate_episode"],
        }
        return simpler_outputs, final_simpler_outputs

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        # Resize to 256x256 using Lanczos (approx via OpenCV's INTER_LANCZOS4)
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        # If float image likely in [0,1], scale to [0,255]
        if np.issubdtype(img.dtype, np.floating):
            maxv = np.nanmax(img)
            if maxv <= 1.0 + 1e-6:
                img = img * 255.0
        # Clip, round and cast to uint8
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        return img

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array([np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1) for a in predicted_raw_actions])
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


class AiroaToG3Pi0FractalBridgeAdapter(BaseAdapter):
    def __init__(
        self, 
        policy,
        policy_setup: str = "widowx_bridge",
        rot6d: bool = False,
        exec_horizon: int = 4,
        action_scale: float = 1.0,
        action_ensemble: bool = True,
        action_ensemble_temp: float = -0.8,
        sticky_action: bool = False,
        add_taks_prefix: bool = False
    ) -> None:
        super().__init__(policy)
        self.sticky_gripper_num_repeat = 10 # same to lerobotpi0
        self.policy = policy
        self.policy_setup = policy_setup
        self.rot6d = rot6d
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_scale = action_scale
        self.action_ensemble_temp = action_ensemble_temp
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = 5
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = exec_horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sticky_action = sticky_action
        self.add_taks_prefix = add_taks_prefix

        if self.policy_setup == "widowx_bridge":
            self.action_ensemble = True
            self.sticky_gripper_num_repeat = 1
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
            self.default_rot = np.array(
                [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
            )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        elif self.policy_setup == "google_robot":
            self.action_ensemble = True
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_plan = deque()

    def preprocess_widowx_proprio(self, eef_pos: np.ndarray, prompt: str):
        """convert ee rotation to the frame of top-down
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L167
        """
        # StateEncoding.POS_EULER: xyz + rpy + pad + gripper(openness)
        
        proprio = eef_pos.copy()
        if self.rot6d:
            rpy_bridge_converted = quat_to_rot6d(torch.from_numpy(eef_pos).unsqueeze(0)).squeeze(0).numpy()
            gripper_openness = proprio[7]  # from simpler, 0 for close, 1 for open
            state = np.concatenate(
                [
                    rpy_bridge_converted[:9],
                    [gripper_openness],
                ]
            )
        else:
            proprio = eef_pos
            rm_bridge = quat2mat(proprio[3:7])
            rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
            gripper_openness = proprio[7]  # from simpler, 0 for close, 1 for open
            state = np.concatenate(
                [
                    proprio[:3],
                    rpy_bridge_converted,
                    np.zeros(1),
                    [gripper_openness],
                ]
            )

        if self.add_taks_prefix:
            prompts = ["google robot: ", "widowx: "]
            prompt = prompts[1] + prompt

        return state, prompt

    def preprocess_google_robot_proprio(self, eef_pos: np.ndarray, prompt: str) -> dict:
        """convert wxyz quat from simpler to xyzw used in fractal"""
       
        if self.rot6d:
            gripper_width = eef_pos[-1]  # from simpler, 0 for close, 1 for open continuous
            gripper_closedness = gripper_width
            quat_wxyz = quat_to_rot6d(torch.from_numpy(eef_pos).unsqueeze(0)).squeeze(0).numpy()
            state = np.concatenate(
                (
                    quat_wxyz[:9],
                    [gripper_closedness],
                )
            )

        else: 
            gripper_width = eef_pos[-1]  # from simpler, 0 for close, 1 for open continuous
            gripper_closedness = (
                1 - gripper_width
            ) 
            quat_xyzw = np.roll(eef_pos[3:7], -1)
            state = np.concatenate(
                (
                    eef_pos[:3],
                    quat_xyzw,
                    [gripper_closedness],
                )
            )


        if self.add_taks_prefix:
            prompts = ["google robot: ", "widowx: "]
            prompt = prompts[0] + prompt

        return state, prompt
    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs):
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
        self._add_image_to_history(image)
        images: List[Image.Image] = self._obtain_image_history()

        eef_pos = kwargs.get("eef_pos", None)
        if self.policy_setup == "google_robot":
            state, task_description = self.preprocess_google_robot_proprio(eef_pos, task_description)
            image_key = "observation.images.image"
        elif self.policy_setup == "widowx_bridge":
            state, task_description = self.preprocess_widowx_proprio(eef_pos, task_description)
            image_key = "observation.images.image"

        if not self.action_plan:
            observation = {
                "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device).float(),
                image_key: images[0],
                "task": [task_description],
            }

            # model output gripper action, +1 = open, 0 = close
            action_chunk = self.policy.step(observation)[: self.pred_action_horizon].cpu().numpy()
            self.action_plan.extend(action_chunk[: self.exec_horizon])

        raw_actions = self.action_plan.popleft()

        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(raw_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"].astype(np.float32) * self.action_scale
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

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[np.ndarray]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        # images = [Image.fromarray(image).convert("RGB") for image in images]
        return images
