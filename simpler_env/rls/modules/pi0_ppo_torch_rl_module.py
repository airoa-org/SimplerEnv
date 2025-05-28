from typing import Any, Dict, Optional
import os
import sys

sys.path.append("/home/user_00029_25b505/lerobot")

import numpy as np
import cv2 as cv
import torch
from torch import nn
import torch.nn.functional as F
from transforms3d.euler import euler2axangle
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import PolicyFeature, FeatureType


class Pi0PPOTorchRLModule(TorchRLModule, DefaultPPORLModule):
    """Pi0 PPO RL Module for RLlib."""

    def __init__(self, *args, **kwargs) -> None:
        self.image_size = kwargs["model_config"].get("image_size")
        self.action_scale = kwargs["model_config"].get("action_scale")
        
        super().__init__(*args, **kwargs)
        self.policy_setup = kwargs["model_config"].get("policy_setup", None)
        self.unnorm_key = kwargs["model_config"].get("unnorm_key", None)

        if self.policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig/1.0.0" if self.unnorm_key is None else self.unnorm_key
            # TODO unnorm_keyの使い道確認
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
            self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        elif self.policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data/0.1.0" if self.unnorm_key is None else self.unnorm_key
        else:
            raise ValueError(f"Unsupported policy_setup: {self.policy_setup}")
        
        print(f"*** policy_setup: {self.policy_setup}, unnorm_key: {self.unnorm_key} ***")

    @override(RLModule)
    def setup(self):
        # TODO RLlibのdeviceと合っているか確認
        gpu_idx = os.environ.get("GPU_IDX", 0)
        self.device = f"cuda:{gpu_idx}"
        os.environ["TOKENIZERS_PARALLELISM"] = "false" # TODO 確認
        
        # assert len(self.action_space.shape) == 1
        # action_dim = self.action_space.shape[0]
        # self.pi = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 16, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, action_dim*2),
        # )

        print(f"Setup!!!")
        self.pi = PI0Policy.from_pretrained("lerobot/pi0")
        self.pi.config.input_features = {
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=tuple([3] + self.image_size),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(8,), # TODO (7,)じゃなくてよい？
            ),
        }
        self.pi.config.output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=self.action_space.shape,
            )
        } 

        self.pi.model.paligemma_with_expert.config.train_expert_only = True
        self.pi.model.paligemma_with_expert.set_requires_grad()
        self.pi.to(self.device)
        self.pi.reset()

        self.vf = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.vf.to(self.device)

        print(F"Finish setup!!")

        # TODO self.pi.train()必要？

    def preprocess_widowx_proprio(self, eef_pos) -> np.array:
        """convert ee rotation to the frame of top-down
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L167
        """
        # StateEncoding.POS_EULER: xyz + rpy + pad + gripper(openness)

        raise NotImpementedError() # TODO batch処理に変更

        # proprio = eef_pos
        # rm_bridge = quat2mat(proprio[3:7])
        # rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        # gripper_openness = proprio[7] # from simpler, 0 for close, 1 for open
        # raw_proprio = np.concatenate(
        #     [
        #         proprio[:3],
        #         rpy_bridge_converted,
        #         np.zeros(1),
        #         [gripper_openness],
        #     ]
        # )
        # print(f"raw_proprio: {raw_proprio}")
        # return raw_proprio
    
    def preprocess_googple_robot_proprio(self, eef_pos) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L204
        """
        # StateEncoding.POS_QUAT: xyz + q_xyzw + gripper(closeness)
        
        quat_xyzw = torch.roll(eef_pos[:, 3:7], shifts=-1, dims=1)
        gripper_width = eef_pos[:, 7:]  # from simpler, 0 for close, 1 for open
        # need invert as the training data comes from closeness
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = torch.cat(
            (
                eef_pos[:, :3],
                quat_xyzw,
                gripper_closedness,
            ),
            dim=1,
        )
        return raw_proprio
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        # image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        image = image.permute(0, 3, 1, 2) # (b, c, h, w)
        image = F.interpolate(image, size=self.image_size, mode='bilinear', align_corners=False) # TODO これで問題ないか確認
        image = image.permute(0, 2, 3, 1) # (b, h, w, c)
        return image

    @override(RLModule)
    def get_initial_state(self) -> dict:
        return {}

    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        self.pi.to(self.device) # TODO RLlibの問題なのかPaligemmaの問題なのか、なぜかcpuになっているので原因調査
        output = {}
        
        # output[Columns.ACTION_DIST_INPUTS] = self.pi(
        #     batch[Columns.OBS].permute(0, 3, 1, 2),
        # ) # (batch, 2*action_dim)

        # Get image observations
        images  = batch[Columns.OBS]["image"]
        images = self._resize_image(images)

        # Get state observations
        eef_pos = batch[Columns.OBS]["eef_pos"]
        if self.policy_setup == "widowx_bridge":
            state = self.preprocess_widowx_proprio(eef_pos)
            image_key = "observation.images.image_0"
        elif self.policy_setup == "google_robot":
            state = self.preprocess_googple_robot_proprio(eef_pos)
            image_key = "observation.images.image"
        else:
            raise ValueError(f"Unsupported policy setup: {self.policy_setup}")

        observations = {
            "observation.state": state.to(self.device),
            image_key: images.permute(0, 3, 1, 2).to(self.device),
            "language_tokens": batch[Columns.OBS]["task_instruction"].to(self.device),
            "language_masks": batch[Columns.OBS]["task_instruction_mask"].to(torch.bool).to(self.device),
        }

        actions = self._calc_actions(observations)[:, 0, :]
        output[Columns.ACTION_DIST_INPUTS] = torch.cat(
            (
                actions,
                torch.full(
                    size=actions.shape,
                    fill_value=0.05,
                ).to(torch.float32).to(self.device).log(), # TODO sigmaの決め方を検討
                # log_stdなので、logをとる
            ),
            dim=1,
        )

        # TODO action ensemble (inferenceの時にのみ組み込むとか、Wrapperに組み込むとか？)

        # TODO 全体的に、勾配計算ができているかチェックする

        return output

    
    def _calc_actions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.pi.config.adapt_to_pi_aloha:
            raise NotImpementedError("adapt_to_pi_aloha is not implemented yet.")

        batch = self.pi.normalize_inputs(batch)

        images, img_masks = self.pi.prepare_images(batch)

        state = self.pi.prepare_state(batch)

        lang_tokens = batch["language_tokens"]
        lang_masks = batch["language_masks"]

        actions = self.pi.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=None
        )
        original_action_dim = self.pi.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim] # (batch, length, action_dim)

        actions = self.pi.unnormalize_outputs({"action": actions})["action"] # TODO 微分可能性確認        
        return actions

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # TODO value functionをどう学習するか
        vf_out = self.vf(
            batch[Columns.OBS]["image"].permute(0, 3, 1, 2),
        ) # (batch_size?, 1)
        return vf_out.squeeze(-1)
