import argparse
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import action_chunk_broker
from simpler_env.evaluation.adapter import OpenpiSimplerBridgeAdapter, OpenpiSimplerFractalAdapter
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

    # Init RNG values
    init_rng_values = [0, 2, 4]

    # Scene 1 configuration
    scene1_config = {
        "scene_name": "bridge_table_1_v1",
        "robot": "widowx",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
        "environments": ["StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutCarrotOnPlateInScene-v0", "PutSpoonOnTableClothInScene-v0"],
        "max_episode_steps": 60,
    }

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
    for init_rng in init_rng_values:
        for ckpt_path in ckpt_paths:
            for env_name in scene1_config["environments"]:
                print(f"Running Scene 1: ckpt_path={ckpt_path}, init_rng={init_rng}, env={env_name}")

                # Create additional_env_build_kwargs
                additional_env_build_kwargs = {}

                # Create config matching bash script parameters
                cfg = ManiSkill2Config(
                    robot=scene1_config["robot"],
                    policy_model="rt1",  # Adapted for the policy setup
                    policy_setup="widowx_bridge",
                    octo_init_rng=init_rng,
                    control_freq=5,
                    sim_freq=500,
                    max_episode_steps=scene1_config["max_episode_steps"],
                    env_name=env_name,
                    scene_name=scene1_config["scene_name"],
                    rgb_overlay_path=scene1_config["rgb_overlay_path"],
                    robot_init_x_range=[scene1_config["robot_init_x"], scene1_config["robot_init_x"], 1],
                    robot_init_y_range=[scene1_config["robot_init_y"], scene1_config["robot_init_y"], 1],
                    obj_variation_mode="episode",
                    obj_episode_range=[0, 24],
                    robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0],
                    robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
                    additional_env_build_kwargs=additional_env_build_kwargs,
                    additional_env_save_tags=[f"octo_init_rng_{init_rng}"],
                    ckpt_path=ckpt_path,
                )

                # Create policy based on policy_model
                policy = OpenpiToAiroaPolicy(
                    policy=action_chunk_broker.ActionChunkBroker(
                        policy=_policy_config.create_trained_policy(
                            _config.get_config("pi0_fractal_low_mem_finetune"),
                            ckpt_path,
                        ),
                        action_horizon=10,
                    ),
                )

                # Use appropriate adapter based on policy setup
                env_policy = OpenpiSimplerBridgeAdapter(policy=policy)

                # Run evaluation
                success_arr = maniskill2_evaluator(env_policy, cfg)
                print(f"Success rate for {env_name}: {success_arr}")

    # Process Scene 2
    for init_rng in init_rng_values:
        for ckpt_path in ckpt_paths:
            for env_name in scene2_config["environments"]:
                print(f"Running Scene 2: ckpt_path={ckpt_path}, init_rng={init_rng}, env={env_name}")

                # Create additional_env_build_kwargs
                additional_env_build_kwargs = {}

                # Create config matching bash script parameters
                cfg = ManiSkill2Config(
                    robot=scene2_config["robot"],
                    policy_model="rt1",  # Adapted for the policy setup
                    policy_setup="widowx_bridge",
                    octo_init_rng=init_rng,
                    control_freq=5,
                    sim_freq=500,
                    max_episode_steps=scene2_config["max_episode_steps"],
                    env_name=env_name,
                    scene_name=scene2_config["scene_name"],
                    rgb_overlay_path=scene2_config["rgb_overlay_path"],
                    robot_init_x_range=[scene2_config["robot_init_x"], scene2_config["robot_init_x"], 1],
                    robot_init_y_range=[scene2_config["robot_init_y"], scene2_config["robot_init_y"], 1],
                    obj_variation_mode="episode",
                    obj_episode_range=[0, 24],
                    robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0],
                    robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
                    additional_env_build_kwargs=additional_env_build_kwargs,
                    additional_env_save_tags=[f"octo_init_rng_{init_rng}"],
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
                env_policy = OpenpiSimplerBridgeAdapter(policy=policy)

                # Run evaluation
                success_arr = maniskill2_evaluator(env_policy, cfg)
                print(f"Success rate for {env_name}: {success_arr}")
