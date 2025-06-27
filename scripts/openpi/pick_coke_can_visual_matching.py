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

    # lr_switch=laying horizontally but flipped left-right to match real eval;
    # upright=standing; laid_vertically=laying vertically
    coke_can_options_arr = [{"lr_switch": True}, {"upright": True}, {"laid_vertically": True}]

    # URDF variations
    urdf_version_arr = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]

    # Environment settings
    env_name = "GraspSingleOpenedCokeCanInScene-v0"
    scene_name = "google_pick_coke_can_1_v4"
    rgb_overlay_path = "./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png"

    # Triple nested loop like in bash script
    for urdf_version in urdf_version_arr:
        for ckpt_path in ckpt_paths:
            for coke_can_option in coke_can_options_arr:
                print(f"Running: ckpt_path={ckpt_path}, urdf_version={urdf_version}, coke_can_option={coke_can_option}")

                # Create additional_env_build_kwargs
                additional_env_build_kwargs = {**coke_can_option, "urdf_version": urdf_version}

                # Create config matching bash script parameters
                cfg = ManiSkill2Config(
                    robot="google_robot_static",
                    policy_model="rt1",
                    policy_setup="google_robot",
                    control_freq=3,
                    sim_freq=501,
                    max_episode_steps=80,
                    env_name=env_name,
                    scene_name=scene_name,
                    rgb_overlay_path=rgb_overlay_path,
                    robot_init_x_range=[0.35, 0.35, 1],  # 0.35 0.35 1 in bash
                    robot_init_y_range=[0.20, 0.20, 1],  # 0.20 0.20 1 in bash
                    obj_init_x_range=[-0.35, -0.12, 5],  # -0.35 -0.12 5 in bash
                    obj_init_y_range=[-0.02, 0.42, 5],  # -0.02 0.42 5 in bash
                    robot_init_rot_quat_center=[0.0, 0.0, 0.0, 1.0],  # robot-init-rot-quat-center
                    robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],  # robot-init-rot-rpy-range
                    additional_env_build_kwargs=additional_env_build_kwargs,
                    ckpt_path=ckpt_path,  # --ckpt-path None in bash
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

                env_policy = OpenpiSimplerFractalAdapter(policy=policy)

                # Run evaluation
                success_arr = maniskill2_evaluator(env_policy, cfg)
                print(f"Success rate: {success_arr}")
