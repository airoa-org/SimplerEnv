import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import mediapy as media
import numpy as np
import torch
from termcolor import colored
from lerobot.common import envs, policies  # noqa: F401
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from g3_haptics.configs.train import TrainPipelineConfigG3Haptics

import simpler_env
from simpler_env.policies.smolvla.smolvla_model import SmolVLAInference
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


@dataclass
class G3EvalPipelineConfig(EvalPipelineConfig):
    env: envs.EnvConfig = None
    cfg_all: TrainPipelineConfigG3Haptics | None = None

    task: str = "google_robot_pick_coke_can"
    n_trajs: int = 100

    def __post_init__(self):
        super().__post_init__()
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.cfg_all = TrainPipelineConfigG3Haptics.from_pretrained(policy_path, cli_overrides=cli_overrides)


@parser.wrap()
def eval_main(cfg: G3EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making policy.")

    # Calculate train and val episodes
    ds_meta = LeRobotDatasetMetadata(
        cfg.cfg_all.dataset.repo_id, root=cfg.cfg_all.dataset.root, revision=cfg.cfg_all.dataset.revision
    )

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )
    policy.eval()
    model = SmolVLAInference(model=policy)

    # build environment
    env = simpler_env.make(cfg.task)

    # run inference
    start = time.time()
    success_arr = []
    for ep_id in range(cfg.n_trajs):
        print(F"Running episode {ep_id + 1}/{cfg.n_trajs}...")
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        # for long-horizon environments, we check if the current subtask is the final subtask
        is_final_subtask = env.is_final_subtask()

        model.reset(instruction)
        print(instruction)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = model.step(image, instruction)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )
            print(timestep, info)
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                # update instruction for long horizon tasks
                instruction = new_instruction
                print(instruction)
            is_final_subtask = env.is_final_subtask()
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        episode_stats = info.get("episode_stats", {})
        success_arr.append(success)
        print(f"Episode {ep_id} success: {success}")
        media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)

    print(
        "**Overall Success**",
        np.mean(success_arr),
        f"({np.sum(success_arr)}/{len(success_arr)})",
    )
    print(f"TIME: {(time.time() - start) / 60:.2f} [min]")


if __name__ == "__main__":
    init_logging()
    eval_main()
