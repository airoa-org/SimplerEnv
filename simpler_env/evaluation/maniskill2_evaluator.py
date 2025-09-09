"""
Evaluate a model on ManiSkill2 environment.
"""

import csv
import os

import numpy as np
from tqdm import tqdm
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_interval_video, write_video


def run_maniskill2_eval_single_episode(
    model,
    task_name,
    episode_id,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    # __import__('ipdb').set_trace()
    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    episode_seed = np.random.randint(0, 100000)

    obs, _ = env.reset(options=env_reset_options, seed=episode_seed)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"
    cost_time = 999
    # action_ensemble = model.action_ensemble_temp  if hasattr(model, "action_ensemble") else "none"

    # Step the environment
    task_descriptions = []
    with tqdm(total=max_episode_steps, desc=f"Episode {episode_id}", leave=False) as pbar:
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = model.step(image, task_description, eef_pos=obs["agent"]["eef_pos"])
            predicted_actions.append(raw_action)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

            # step the environment
            obs, reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )

            cost_time = min(cost_time, timestep) if info["success"] else 999

            success = "success" if done else "failure"
            new_task_description = env.get_language_instruction()
            if new_task_description != task_description:
                task_description = new_task_description
                print(task_description)
            is_final_subtask = env.is_final_subtask()

            # print(timestep, info)

            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
            images.append(image)
            task_descriptions.append(task_description)
            timestep += 1
            pbar.update(1)

    episode_stats = info.get("episode_stats", {})

    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    base_logging_dir = os.path.join(logging_dir, ckpt_path_basename)

    # このロジックはあったほうがいいかも
    # if not os.path.exists(base_logging_dir):
    #     ckpt_logging_dir = base_logging_dir
    # else:
    #     i = 1
    #     while True:
    #         new_logging_dir = f"{base_logging_dir}_{i}"
    #         if not os.path.exists(new_logging_dir):
    #             ckpt_logging_dir = new_logging_dir
    #             break
    #         i += 1
    # logging_dir = os.path.join(ckpt_logging_dir, task_name)
    # os.makedirs(logging_dir, exist_ok=True)

    ckpt_logging_dir = base_logging_dir
    logging_dir = os.path.join(ckpt_logging_dir, task_name)
    os.makedirs(logging_dir, exist_ok=True)

    # save video
    if success == "success":
        success_emoji = "✅"
    else:
        success_emoji = "❌"

    episode_stats = info.get("episode_stats", {})
    if obj_variation_mode == "xy":
        video_name = f"{success}_{episode_id}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_{episode_id}_obj_episode_{obj_episode_id}"
    else:
        raise Exception(f"Unknown obj_variation_mode: {obj_variation_mode}")
    
    video_path = os.path.join(logging_dir, f"{video_name}.mp4")
    write_video(video_path, images, fps=5)
    print(f"{success_emoji} Video saved to {video_path}")

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    # save summary
    summary_file = os.path.join(ckpt_logging_dir, "summary.csv")
    file_exists = os.path.exists(summary_file)
    with open(summary_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["task_name", "episode_id", "success", "task_description", "cost_time"]
        data_row = [task_name, episode_id, success, task_description, str(cost_time)]
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)

    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"

    if obj_variation_mode == "xy" or obj_variation_mode == "episode_xy":
        add_info = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        add_info = f"{success}_idx_{episode_id}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        add_info = add_info + f"_{k}_{v}"

    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"

    r, p, y = quat2euler(robot_init_quat)
    details_summary_file = os.path.join(ckpt_logging_dir, "details_summary.csv")
    file_exists = os.path.exists(details_summary_file)

    with open(details_summary_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "task_name",
            "episode_id",
            "success",
            "task_description",
            "cost_time",
            "scene_name",
            "control_mode",
            "env_save_name",
            "rgb_overlay_path",
            "robot_init_xy",
            "robot_init_rpy",
            "add_info",
            "additional_env_build_kwargs",
        ]
        data_row = [
            task_name,
            episode_id,
            success,
            task_description,
            str(cost_time),
            scene_name,
            control_mode,
            env_save_name,
            rgb_overlay_path_str,
            f"{robot_init_x:.3f}_{robot_init_y:.3f}",
            f"{r:.3f}_{p:.3f}_{y:.3f}",
            add_info,
            additional_env_build_kwargs,
        ]

        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # run inference
    if args.robot_variation_mode == "xy":
        for robot_init_x in args.robot_init_xs:
            for robot_init_y in args.robot_init_ys:
                for robot_init_quat in args.robot_init_quats:
                    success_arr += _run_single_evaluation(model, args, control_mode, robot_init_x, robot_init_y, robot_init_quat)

    if args.robot_variation_mode == "episode_xy":
        for robot_episode_id in range(args.robot_episode_range[0], args.robot_episode_range[1]):
            robot_init_x = np.random.uniform(args.robot_init_x_range[0], args.robot_init_x_range[1])
            robot_init_y = np.random.uniform(args.robot_init_y_range[0], args.robot_init_y_range[1])
            for robot_init_quat in args.robot_init_quats:
                success_arr +=  _run_single_evaluation(model, args, control_mode, robot_init_x, robot_init_y, robot_init_quat)

    return success_arr


def _run_single_evaluation(model, args, control_mode, robot_init_x, robot_init_y, robot_init_quat):
    success_arr = []
    kwargs = dict(
        model=model,
        task_name=args.task_name,
        episode_id=args.episode_id,
        ckpt_path=args.ckpt_path,
        robot_name=args.robot,
        env_name=args.env_name,
        scene_name=args.scene_name,
        robot_init_x=robot_init_x,
        robot_init_y=robot_init_y,
        robot_init_quat=robot_init_quat,
        control_mode=control_mode,
        additional_env_build_kwargs=args.additional_env_build_kwargs,
        rgb_overlay_path=args.rgb_overlay_path,
        control_freq=args.control_freq,
        sim_freq=args.sim_freq,
        max_episode_steps=args.max_episode_steps,
        enable_raytracing=args.enable_raytracing,
        additional_env_save_tags=args.additional_env_save_tags,
        obs_camera_name=args.obs_camera_name,
        logging_dir=args.logging_dir,
    )
    if args.obj_variation_mode == "xy":
        for obj_init_x in args.obj_init_xs:
            for obj_init_y in args.obj_init_ys:
                success = run_maniskill2_eval_single_episode(
                    obj_init_x=obj_init_x,
                    obj_init_y=obj_init_y,
                    **kwargs,
                )
                success_arr.append(success)
    elif args.obj_variation_mode == "episode":
        import random

        sampled_ids = random.sample(range(1000), args.obj_episode_range[1])
        for idx, obj_episode_id in enumerate(sampled_ids):
            success = run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs)
            success_arr.append(success)
    elif args.obj_variation_mode == "episode_xy":
        for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
            obj_init_x = np.random.uniform(args.obj_init_x_range[0], args.obj_init_x_range[1])
            obj_init_y = np.random.uniform(args.obj_init_y_range[0], args.obj_init_y_range[1])
            success = run_maniskill2_eval_single_episode(obj_init_x=obj_init_x, obj_init_y=obj_init_y, **kwargs)
            success_arr.append(success)
    else:
        raise NotImplementedError()

    return success_arr
