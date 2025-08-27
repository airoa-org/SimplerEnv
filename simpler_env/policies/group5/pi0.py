#!/usr/bin/env python3
# import os
import torch # PyTorch for deep learning and tensor operations
import numpy as np
import matplotlib.pyplot as plt
import einops
import cv2 as cv
# from PIL import Image

# from collections import deque
# from typing import List, Optional, Sequence, Dict
# from typing import Dict, Optional
from typing import Optional, Sequence

# Lerobot-specific modules for policy construction and logging
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_torch_device
# from lerobot.common.logger import log_output_dir
# from lerobot.common.utils.utils import (
#     init_logging,          # Initialize logging configuration
#     set_global_seed,       # Set global random seed for reproducibility
#     get_safe_torch_device, # Select safe torch device (e.g., GPU/CPU)
# )
# Configuration parser and schema definition for evaluation pipeline
# from lerobot.configs import parser
from lerobot.common.envs.utils import ( 
    rot6d_to_axis_angle,
    quat_to_rot6d
    )
from lerobot.configs.eval import EvalPipelineConfig

# Standard libraries and third-party packages
import logging             # Logging utility
# from pprint import pformat # Pretty-printing nested data structures
# from dataclasses import asdict # Convert dataclass instances to dictionaries

# from transforms3d.euler import euler2axangle
# from simpler_env.utils.action.action_ensemble import ActionEnsembler
from simpler_env.policies.base import AiroaBasePolicy
# from .geometry import mat2euler, quat2mat

class Group5PiInference(AiroaBasePolicy):
    def __init__(self, cfg: EvalPipelineConfig) -> None:
        self.cfg = cfg # EvalPipelineConfig
        self.device = get_safe_torch_device(self.cfg.device, log=True)
        self.policy = self.__make_policy() # make policy
        self.policy.eval()

    def __make_policy(self) -> PreTrainedPolicy:
        # Ensure a pretrained model path is provided for evaluation
        assert self.cfg.policy.pretrained_path is not None, "Pretrained path must be specified for evaluation"
        logging.info(f"Using pretrained model for eval: {self.cfg.policy.pretrained_path}")
        
        # Construct the policy (model)
        logging.info("Making policy...")
        return make_policy(
            cfg=self.cfg.policy,
            device=self.cfg.device,
            env_cfg=self.cfg.env,
        )
    
    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [image for image in images]
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
    
    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        # action , world_vector, rotation_delta, open_gripper
        rtype = self.cfg.env.type # robot type "google_robot" or "widowx_bridge"
        # get eef_pos
        # eef_pos = kwargs.get("eef_pos", None)
        # robot_type = self.cfg.env.type # robot type   
             
        assert image.dtype == np.uint8
        image = torch.from_numpy(image)

        h, w, c = image.shape
        assert c < h and c < w, f"expect channel last images, but instead got {image.shape=}"
        assert image.dtype == torch.uint8, f"expect torch.uint8, but instead {image.dtype=}"
        image = einops.rearrange(image, "h w c -> c h w").contiguous()
        image = image.type(torch.float32) / 255.0

        # image = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
        # Convert numpy array to tensor, channel last -> channel first, normalize [0,1]
        # image = torch.from_numpy(image)
        # h, w, c = image.shape
        # assert c < h and c < w, f"expect channel last images, but instead got {image.shape=}"
        # assert image.dtype == torch.uint8, f"expect torch.uint8, but instead {image.dtype=}"
        # image = einops.rearrange(image, "h w c -> c h w").contiguous()
        # image = image.type(torch.float32) / 255.0
        # image = image / 255.0
        # image = torch.from_numpy(image).unsqueeze(0).permute(2, 0, 1)

        observation = {}
        eef_pos = kwargs.get("eef_pos", None)
        if rtype == "google_robot":
            gripper_width = eef_pos[:,:7]
            gripper_closedness = ( 
                1 - gripper_width 
            )
            if len(gripper_closedness.shape) == 1:
                gripper_closedness = gripper_closedness[:, None]
            state = np.concatenate(
                ( 
                    eef_pos[:,:7], # x, y, z, w, x, y, z
                    gripper_closedness,
                ),
                axis=-1
            )
            state = torch.from_numpy(state).unsqueeze(0).float()
            observation["observation.state"] = torch.from_numpy(state).unsqueeze(0).float()
            observation["observation.images.image"] = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            # observation = {
            #     "observation.state": torch.from_numpy(state).unsqueeze(0).float(),
            #     "observation.images.image": torch.from_numpy(images[0] / 255.0).permute(2, 0, 1).unsqueeze(0),
            #     # "task": [task_description]
            # }
        # elif rtype == "window_bridge":
        else: # rtype == "widowx"
            gripper_width = eef_pos[:7].reshape(1, -1) # (1, 7) # eef_pos[:,:7]
            gripper_closedness = (
                gripper_width
            )
            if len(gripper_closedness.shape) == 1:
                gripper_closedness = gripper_closedness[:, None]
            
            pose = np.concatenate(
                (
                    eef_pos[:7].reshape(1, -1),
                    gripper_closedness,
                ),
                axis=-1
            )
            pose = torch.from_numpy(pose).float()
            state = quat_to_rot6d(pose) # (1, 10)
            # np.array([eef_pos[6]]).reshape(1, -1) # (1, 1)
            # (1, 7) + (1, 1) → (1, 8)
            # state = np.concatenate((gripper_width, gripper_closedness), axis=-1) # (1, 8)
            # gripper_closedness = ( 
            #     gripper_width 
            # )
            # if len(gripper_closedness.shape) == 1:
            #     gripper_closedness = gripper_closedness[:, None]
            # state = np.concatenate(
            #     ( 
            #         eef_pos[:7].reshape(1,-1), # eef_pos[:,:7], # x, y, z, w, x, y, z
            #         gripper_closedness,
            #     ),
            #     axis=-1
            # )
            observation["observation.state"] = state # .unsqueeze(0)
            observation["observation.images.3rd_view_camera"] = image.unsqueeze(0)
            # observation["observation.images.image_0"] = image.unsqueeze(0) # torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            # torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            # observation = {
            #     "observation.state": torch.from_numpy(state_numpy).unsqueeze(0).float(),
            #     "observation.images.image_0": torch.from_numpy(images[0] / 255).permute(2, 0, 1).unsqueeze(0),
            #     # "task": [task_description]
            # }
            # else:
            #     assert observation != None

        observation = {key: observation[key].to(self.device, non_blocking=True) for key in observation}
        # observation["observation.images.3rd_view_camera"] = None
        observation["task"] = [ task_description ]

        with torch.inference_mode():
            raw_actions = self.policy.select_action(observation)
            raw_actions = raw_actions.detach().cpu()
            actions = rot6d_to_axis_angle(raw_actions)
            
            raw_actions = raw_actions.numpy()
            actions = actions.numpy()

            if rtype == "google_robot":
                actions[:,-1] = - np.clip(2.0 * actions[:, -1] - 1.0, -1, 1)
            else: # rtype == "widowx"
                actions[:,-1] = np.clip(2.0 * actions[:, -1] - 1.0, -1, 1) 
            
            raw_actions = raw_actions.squeeze(0) # (10, )
            actions = actions.squeeze(0) # (7, )
            
        # print("==============")
        # print(actions.shape)
        # print(raw_actions.shape)

        raw_action = {
            "world_vector": actions[:3],
            "rotation_delta": actions[3:6],
            "open_gripper": actions[6:7],  # range [0, 1]; 1 = open; 0 = close
        }

        action = {
            "world_vector": actions[:3],
            "rot_axangle": actions[3:6],
            "gripper": actions[6:7], 
            "terminate_episode": np.array([0.0]) # not using action chunking
        }

        return raw_action, action
    
    def reset(self, task_description: str) -> None:
        # reset policy
        self.policy.reset()
        # debug function
        if True:
            logging.info("===== Resetting Environment =====")
            logging.info(f"Task description: {task_description}")

