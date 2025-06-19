from collections import deque
from typing import Optional, Sequence
import os

import matplotlib.pyplot as plt
import numpy as np
# How to use SmolVLA
# https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/smolvla/modeling_smolvla.py
# https://huggingface.co/blog/smolvla#train-from-scratch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad
import torch
from transforms3d.euler import euler2axangle

from simpler_env.utils.action.action_ensemble import ActionEnsembler


class SmolVLAInference:
    def __init__(
        self,
        model: Optional[SmolVLAPolicy] = None,
        dataset_id: Optional[str] = None,
        model_type: str = "smolvla",
        policy_setup: str = "widowx_bridge",
        exec_horizon: int = 1,
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            dataset_id = "bridge_dataset" if dataset_id is None else dataset_id
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            dataset_id = "fractal20220817_data" if dataset_id is None else dataset_id
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for octo models.")
        self.policy_setup = policy_setup
        self.dataset_id = dataset_id

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert self.device == "cuda:0"

        if model is not None:
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = model
        elif model_type in ["smolvla"]:
            # released huggingface smolVLA models
            self.model_type = "lerobot/smolvla_base" # SmolVLAベースモデル
            # self.model_type = f"{os.path.expanduser('~/shared-storage')}/group_3/members/user_00005_25b505/workspace/lerobot/outputs/train/2025-06-10/10-50-59_smolvla/checkpoints/last" # Fine-tunedモデル with OXEデータセット
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = SmolVLAPolicy.from_pretrained(self.model_type)
            self.model.to(self.device)
        else:
            raise NotImplementedError()

        self.action_scale = action_scale
        self.pred_action_horizon = self.model.config.n_action_steps
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.model.config.n_obs_steps)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

    def _resize_image(self, image: np.ndarray) -> torch.Tensor:
        image = np.transpose(image, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
        image = np.expand_dims(image, axis=0)  # (3, H, W) -> (1, 3, H, W)

        # to torch tensor and to float32 and [0, 255] -> [0, 1]
        image = torch.from_numpy(image).to(torch.int).to(self.device)
        image = image.to(torch.float32) / 255.

        # resize image
        image = resize_with_pad(image, *self.model.config.resize_imgs_with_padding, pad_value=0)

        return image

    def _add_image_to_history(self, image: torch.Tensor) -> None:
        self.image_history.append(image)
        # Alternative implementation below; but looks like for real eval, filling the entire buffer at the first step is not necessary
        # if self.num_image_history == 0:
        #     self.image_history.extend([image] * self.horizon)
        # else:
        #     self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.pred_action_horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = torch.cat(list(self.image_history), dim=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this should be of float type, not a bool type
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

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
                # task description has changed; reset the policy state
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)

        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]

        batch = {
            "observation.images.image": images,
            "obaservation.state": None,
            "task": [self.task_description,],
        }
        raw_actions = self.model.select_action(batch=batch)
        raw_actions = raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)

        assert raw_actions.shape == (self.pred_action_horizon, 7)
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)
            raw_actions = raw_actions[None]  # [1, 7]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
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
            current_gripper_action = raw_action["open_gripper"]

            # This is one of the ways to implement gripper actions; we use an alternative implementation below for consistency with real
            # gripper_close_commanded = (current_gripper_action < 0.5)
            # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

            # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
            # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
            #     self.sticky_action_is_on = True
            #     self.sticky_gripper_action = relative_gripper_action

            # if self.sticky_action_is_on:
            #     self.gripper_action_repeat += 1
            #     relative_gripper_action = self.sticky_gripper_action

            # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
            #     self.sticky_action_is_on = False
            #     self.gripper_action_repeat = 0

            # action['gripper'] = np.array([relative_gripper_action])

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

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
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
