from typing import Any, Dict, List

from . import random_envs
from ..policies.base import AiroaBasePolicy
from .config import ManiSkill2Config
from .evaluate import _run_single_evaluation


def widowx_task1_pick_object(env_policy: AiroaBasePolicy, ckpt_path: str, control_freq: int = 5) -> List[List[bool]]:

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=120,
        env_name="GraspRandomObjectInScene-v0",
        scene_name="bridge_table_1_v1",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x_range=[0.147, 0.147, 1],
        robot_init_y_range=[0.028, 0.028, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task1_pick_object",
    )
    results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def widowx_task2_stack_cube(env_policy: AiroaBasePolicy, ckpt_path: str, control_freq: int = 5) -> List[List[bool]]:

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=120,
        env_name="StackRandomGreenYellowCubeInScene-v0",
        scene_name="bridge_table_1_v1",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x_range=[0.147, 0.147, 1],
        robot_init_y_range=[0.028, 0.028, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task2_stack_cube",
    )
    results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def widowx_task3_put_object_on_top(env_policy: AiroaBasePolicy, ckpt_path: str, control_freq: int = 5) -> List[List[bool]]:

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=120,
        env_name="PutRandomObjectOnRandomTopInScene-v0",
        scene_name="bridge_table_1_v1",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x_range=[0.147, 0.147, 1],
        robot_init_y_range=[0.028, 0.028, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task3_put_object_on_top",
    )
    results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results


def widowx_task4_put_object_in_basket(env_policy: AiroaBasePolicy, ckpt_path: str, control_freq: int = 5) -> List[List[bool]]:

    results: List[List[bool]] = []

    cfg = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx_sink_camera_setup",
        control_freq=control_freq,
        sim_freq=500,
        max_episode_steps=240,
        env_name="PutRandomObjectInBasketScene-v0",
        scene_name="bridge_table_1_v2",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        robot_init_x_range=[0.127, 0.127, 1],
        robot_init_y_range=[0.06, 0.06, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 30],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name="widowx_task4_put_object_in_basket",
    )
    results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    return results
