import argparse

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import action_chunk_broker

from simpler_env.evaluation.adapter import AiroaToSimplerBridgeAdapter, AiroaToSimplerFractalAdapter
from simpler_env.evaluation.config import ManiSkill2Config
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

from .policy import OpenpiToAiroaPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run ManiSkill2 evaluation with specified checkpoints")
    parser.add_argument(
        "--ckpt-paths",
        nargs="+",
        required=True,
        help="List of checkpoint paths to evaluate (e.g., --ckpt-paths /data/checkpoints/21000/ /data/checkpoints/10000/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_paths = args.ckpt_paths

    # Scene 2 configuration
    scene2_config = {
        "scene_name": "bridge_table_1_v2",
        "robot": "widowx_sink_camera_setup",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "robot_init_x": 0.127,
        "robot_init_y": 0.06,
        "environments": ["PutEggplantInBasketScene-v0"],
        "max_episode_steps": 120,
    }

    # Process Scene 1
    env_names = ["PutCarrotOnPlateInScene-v0", "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutSpoonOnTableClothInScene-v0"]
    for ckpt_path in ckpt_paths:
        for env_name in env_names:
            print(f"Running Scene 1: ckpt_path={ckpt_path}, env={env_name}")

            # Create config matching bash script parameters
            cfg = ManiSkill2Config(
                robot="widowx",
                policy_setup="widowx_bridge",
                control_freq=5,
                sim_freq=500,
                max_episode_steps=60,
                env_name=env_name,
                scene_name="bridge_table_1_v1",
                rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
                robot_init_x_range=[0.147, 0.147, 1],
                robot_init_y_range=[0.028, 0.028, 1],
                obj_variation_mode="episode",
                obj_episode_range=[0, 24],
                robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0],
                robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
                additional_env_build_kwargs={},
                ckpt_path=ckpt_path,
            )

            # Create policy based on policy_model
            policy = OpenpiToAiroaPolicy(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=_policy_config.create_trained_policy(
                        _config.get_config("pi0_bridge_low_mem_finetune"),
                        ckpt_path,
                    ),
                    action_horizon=10,
                ),
            )

            # Use appropriate adapter based on policy setup
            env_policy = AiroaToSimplerBridgeAdapter(policy=policy)

            # Run evaluation
            success_arr = maniskill2_evaluator(env_policy, cfg)
            print(f"Success rate for {env_name}: {success_arr}")

    # Process Scene 2
    env_names = ["PutEggplantInBasketScene-v0"]
    for ckpt_path in ckpt_paths:
        for env_name in env_names:
            print(f"Running Scene 2: ckpt_path={ckpt_path}, env={env_name}")

            # Create config matching bash script parameters
            cfg = ManiSkill2Config(
                robot="widowx_sink_camera_setup",
                policy_setup="widowx_bridge",
                control_freq=5,
                sim_freq=500,
                max_episode_steps=120,
                env_name=env_name,
                scene_name="bridge_table_1_v2",
                rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
                robot_init_x_range=[0.127, 0.127, 1],
                robot_init_y_range=[0.06, 0.06, 1],
                obj_variation_mode="episode",
                obj_episode_range=[0, 24],
                robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0],
                robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
                additional_env_build_kwargs={},
                ckpt_path=ckpt_path,
            )

            # Create policy based on policy_model
            policy = OpenpiToAiroaPolicy(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=_policy_config.create_trained_policy(
                        _config.get_config("pi0_bridge_low_mem_finetune"),
                        ckpt_path,
                    ),
                    action_horizon=10,
                ),
            )

            # Use appropriate adapter based on policy setup
            env_policy = AiroaToSimplerBridgeAdapter(policy=policy)

            # Run evaluation
            success_arr = maniskill2_evaluator(env_policy, cfg)
            print(f"Success rate for {env_name}: {success_arr}")
