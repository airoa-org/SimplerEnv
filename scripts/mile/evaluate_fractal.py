import argparse
from typing import Dict

import numpy as np
import torch

from simpler_env.evaluation.adapter import AiroaToSimplerFractalAdapter
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy


class MileToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load checkpoint and rebuild model from saved config
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "config" in checkpoint and "model_state_dict" in checkpoint:
            # Newer training util format
            from mile.models.robot_mile import RobotMile  # Imported lazily to keep SimplerEnv deps minimal

            self.config = checkpoint["config"]
            self.model = RobotMile(self.config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback: checkpoint is a plain state_dict and we need a default config
            from mile.configs.fractal_config import get_fractal_config
            from mile.models.robot_mile import RobotMile

            self.config = get_fractal_config()
            self.model = RobotMile(self.config).to(self.device)
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Internal recurrent state helpers
        self._num_actions = getattr(self.config.MODEL, "NUM_JOINTS", 7)
        self._prev_action = torch.zeros(1, 1, self._num_actions, device=self.device, dtype=torch.float32)

    def step(self, obs: Dict) -> Dict:
        # obs keys from adapter.preprocess: image (H, W, C) uint8, prompt (str), state (np.ndarray)
        image = obs["image"]
        if isinstance(image, torch.Tensor):
            img_t = image
            if img_t.ndim == 3 and img_t.shape[-1] in (3, 4):
                img_t = img_t[..., :3]
                img_t = img_t.permute(2, 0, 1).contiguous()
        else:
            img = np.asarray(image)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            # HWC uint8 -> CHW float32 in [0, 1]
            img_t = torch.from_numpy(img).to(torch.float32) / 255.0
            img_t = img_t.permute(2, 0, 1).contiguous()

        # Batch and sequence dims: (B=1, S=1, C, H, W)
        img_t = img_t.unsqueeze(0).unsqueeze(0).to(self.device)

        # Joint/state vector: pad/truncate to expected dim
        state_vec = np.asarray(obs.get("state", np.zeros(self._num_actions, dtype=np.float32)), dtype=np.float32)
        joint_dim = getattr(self.config.MODEL.JOINT, "INPUT_DIM", state_vec.shape[-1])
        if state_vec.shape[-1] < joint_dim:
            pad = np.zeros(joint_dim - state_vec.shape[-1], dtype=np.float32)
            state_vec = np.concatenate([state_vec, pad], axis=-1)
        elif state_vec.shape[-1] > joint_dim:
            state_vec = state_vec[:joint_dim]
        joint_states = torch.from_numpy(state_vec).view(1, 1, -1).to(self.device)

        # Text prompt
        prompt = obs.get("prompt", "")
        text_instructions = [str(prompt)]

        batch = {
            "image": img_t,  # (1, 1, 3, H, W)
            "joint_states": joint_states,  # (1, 1, D)
            "text_instructions": text_instructions,  # list[str]
            "action": self._prev_action,  # previous action for recurrence (1, 1, A)
            # Provide a dedicated previous action tensor without a sequence dimension for robust deployment
            "prev_action": self._prev_action.view(1, -1),  # (1, A)
        }

        with torch.no_grad():
            # Use deployment forward for fast step-wise inference
            out = self.model.deployment_forward(batch, is_dreaming=False)
            actions = out["joint_actions"][0, 0].detach().to("cpu").numpy()

        # Update prev action buffer (shape must be (1, 1, A))
        self._prev_action = torch.from_numpy(actions).view(1, 1, -1).to(self.device)

        return {
            "actions": actions,
            # Use 1D array so downstream code can safely index [0]
            "terminate_episode": np.array([0.0], dtype=np.float32),
        }

    def reset(self) -> None:
        # Clear recurrent state inside the model if present
        if hasattr(self.model, "last_h"):
            self.model.last_h = None
        if hasattr(self.model, "last_sample"):
            self.model.last_sample = None
        if hasattr(self.model, "count"):
            self.model.count = 0
        self._prev_action = torch.zeros(1, 1, self._num_actions, device=self.device, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation (MILE)")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the MILE checkpoint to evaluate.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="mile_default",
        help="(Unused placeholder) Config name for MILE. Present for CLI compatibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = MileToAiroaPolicy(ckpt_path=ckpt_path)

    env_policy = AiroaToSimplerFractalAdapter(policy=policy)

    print("Policy initialized (MILE). Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=env_policy, ckpt_path=args.ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")


