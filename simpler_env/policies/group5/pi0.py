#!/usr/bin/env python3
import torch  # PyTorch for deep learning and tensor operations
import numpy as np
import matplotlib.pyplot as plt

import einops
from typing import Optional, Sequence

# Lerobot-specific modules for policy construction and logging
from lerobot.common.policies.factory import make_policy
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.envs.utils import ( 
    rot6d_to_axis_angle,
    quat_to_rot6d
)

# Standard libraries and third-party packages
import logging  # Logging utility
from simpler_env.policies.base import AiroaBasePolicy


class Group5PiInference(AiroaBasePolicy):
    """
    Group5PiInference wraps a pretrained robot policy for inference.

    This class handles:
    - Loading a pretrained policy from a given configuration.
    - Preparing observations (images, states, gripper info).
    - Running inference to predict robot actions.
    - Resetting the policy and visualizing predictions.

    Attributes
    ----------
    cfg : EvalPipelineConfig
        The evaluation pipeline configuration containing policy, env, and device information.
    device : torch.device
        The device (CPU/GPU) to run inference on.
    policy : PreTrainedPolicy
        The pretrained policy loaded from the given path.
    """

    def __init__(self, cfg: EvalPipelineConfig) -> None:
        """
        Initialize the inference pipeline.

        Parameters
        ----------
        cfg : EvalPipelineConfig
            Configuration containing environment and policy information.
        """
        self.cfg = cfg
        self.device = get_safe_torch_device(self.cfg.device, log=True)
        self.policy = self.__make_policy()
        self.policy.eval()

    def __make_policy(self) -> PreTrainedPolicy:
        """
        Load a pretrained policy from the given config.

        Returns
        -------
        PreTrainedPolicy
            The loaded policy object.
        """
        assert self.cfg.policy.pretrained_path is not None, "Pretrained path must be specified for evaluation"
        logging.info(f"Using pretrained model for eval: {self.cfg.policy.pretrained_path}")
        
        logging.info("Making policy...")
        return make_policy(
            cfg=self.cfg.policy,
            device=self.cfg.device,
            env_cfg=self.cfg.env,
        )

    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        *args,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one inference step given the current observation.

        Parameters
        ----------
        image : np.ndarray
            Input image from the robot camera (H, W, C), dtype=uint8.
        task_description : str, optional
            Natural language task description.
        eef_pos : np.ndarray
            End-effector pose provided via kwargs (expected shape (7,)).

        Returns
        -------
        raw_action : dict
            Dictionary with unprocessed actions (world_vector, rotation_delta, open_gripper).
        action : dict
            Dictionary with post-processed actions suitable for execution.
        """
        rtype = self.cfg.env.type  # robot type: "google_robot" or "widowx_bridge"

        # --- Image preprocessing ---
        assert image.dtype == np.uint8, "Image must be uint8"
        image = torch.from_numpy(image)
        h, w, c = image.shape
        assert c < h and c < w, f"Expect channel-last image, got {image.shape=}"
        image = einops.rearrange(image, "h w c -> c h w").contiguous()
        image = image.type(torch.float32) / 255.0  # Normalize to [0,1]

        # --- Construct observation dict ---
        observation = {}
        eef_pos = kwargs.get("eef_pos", None)

        if rtype == "google_robot":
            # For Google Robot: interpret gripper closedness as 1 - width
            gripper_width = eef_pos[:7].reshape(1, -1)
            gripper_closedness = 1 - gripper_width
            if len(gripper_closedness.shape) == 1:
                gripper_closedness = gripper_closedness[:, None]

            pose = np.concatenate((eef_pos[:7].reshape(1, -1), gripper_closedness), axis=-1)
            pose = torch.from_numpy(pose).float()
            state = quat_to_rot6d(pose)  # Convert quaternion to 6D rotation
            observation["observation.state"] = state
            observation["observation.overhead_camera"] = image.unsqueeze(0)

            def normalize_gripper_action(gripper_action):
                return - np.clip(2.0 * gripper_action - 1.0, -1, 1)

        else:  # widowx_bridge robot
            gripper_width = eef_pos[:7].reshape(1, -1)
            gripper_closedness = gripper_width
            if len(gripper_closedness.shape) == 1:
                gripper_closedness = gripper_closedness[:, None]

            pose = np.concatenate((eef_pos[:7].reshape(1, -1), gripper_closedness), axis=-1)
            pose = torch.from_numpy(pose).float()
            state = quat_to_rot6d(pose)
            observation["observation.state"] = state
            observation["observation.images.3rd_view_camera"] = image.unsqueeze(0)

            def normalize_gripper_action(gripper_action):
                return np.clip(2.0 * gripper_action - 1.0, -1, 1)

        # Move observation tensors to device
        observation = {key: observation[key].to(self.device, non_blocking=True) for key in observation}
        observation["task"] = [task_description]

        # --- Policy inference ---
        with torch.inference_mode():
            raw_actions = self.policy.select_action(observation)
            raw_actions = raw_actions.detach().cpu()
        
        actions = rot6d_to_axis_angle(raw_actions).numpy()
        actions[:, -1] = normalize_gripper_action(actions[:, -1])
        actions = actions.squeeze(0)
        
            # # Adjust gripper control depending on robot type
            # if rtype == "google_robot":
            #     actions[:, -1] = -np.clip(2.0 * actions[:, -1] - 1.0, -1, 1)
            # else:
            #     actions[:, -1] = np.clip(2.0 * actions[:, -1] - 1.0, -1, 1)
            # actions = actions.squeeze(0)  # (1,7) -> (7,)

        # --- Build return dicts ---
        raw_action = {
            "world_vector": actions[:3],
            "rotation_delta": actions[3:6],
            "open_gripper": actions[6:7],  # [0,1] range: 1=open, 0=close
        }

        action = {
            "world_vector": actions[:3],
            "rot_axangle": actions[3:6],
            "gripper": actions[6:7],
            "terminate_episode": np.array([0.0])  # no chunking
        }

        return raw_action, action

    def reset(self, task_description: str) -> None:
        """
        Reset the policy before starting a new episode.

        Parameters
        ----------
        task_description : str
            Natural language description of the task.
        """
        self.policy.reset()
        logging.info("===== Resetting Environment =====")
        logging.info(f"Task description: {task_description}")

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        """
        Visualize predicted actions over an episode and save as a plot.

        Parameters
        ----------
        predicted_raw_actions : Sequence[np.ndarray]
            Sequence of predicted raw actions for each timestep.
        images : Sequence[np.ndarray]
            Sequence of images captured during the episode.
        save_path : str
            Path to save the visualization image.
        """
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        # Concatenate images into a strip (every 3rd image)
        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # Set up matplotlib figure with mosaic layout
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # Convert actions into array form for plotting
        pred_actions = np.array([
            np.concatenate([raw_action["world_vector"], raw_action["rotation_delta"], raw_action["open_gripper"]], axis=-1)
            for raw_action in predicted_raw_actions
        ])

        # Plot each action dimension over time
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        # Display image strip
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")

        plt.legend()
        plt.savefig(save_path)

