# from simpler_env.eval import EvalutePolicy
import numpy as np
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

import simpler_env

# from simpler_env.eval import EvalutePolicy, BasePolicy


class OpenpiSimplerBridgeAdapter:
    def __init__(self):
        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

    def reset(self):
        pass

    def preprocess(self, image: np.ndarray, eef_pos: np.ndarray, prompt: str) -> dict:

        proprio = eef_pos
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )

        inputs = {
            "image": image,
            "prompt": prompt,
            "state": proprio,
        }
        return inputs

    def postprocess(self, outputs: dict) -> dict:
        action = outputs["actions"]
        roll, pitch, yaw = action[3:6]
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = action[-1]
        action_gripper = 2.0 * (action_gripper > 0.5) - 1.0

        action = np.concatenate(
            [
                action[:3],
                action_rotation_ax * action_rotation_angle,
                [action_gripper],
            ]
        )
        return {
            "actions": action,
            "terminate_episode": outputs["terminate_episode"],
        }


class OpenpiSimplerFractalAdapter:
    def __init__(self):
        # Constants
        self.sticky_gripper_num_repeat = 15  # same used in Octo. Note this is for every individual action, not every action chunk. Control freq is 3Hz, so roughly sticky for 5 seconds.

    def reset(self):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0

    def preprocess(self, image: np.ndarray, eef_pos: np.ndarray, prompt: str) -> dict:
        """convert wxyz quat from simpler to xyzw used in fractal"""
        quat_xyzw = np.roll(eef_pos[3:7], -1)
        gripper_width = eef_pos[7]  # from simpler, 0 for close, 1 for open continuous
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        proprio = np.concatenate(
            (
                eef_pos[:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )

        # H W C [0, 255]
        inputs = {
            "image": image,
            "prompt": prompt,
            "state": proprio,
        }
        return inputs

    def postprocess(self, outputs: dict) -> dict:
        action = outputs["actions"]
        roll, pitch, yaw = action[3:6]
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L187"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 open, 1 close for simpler

        gripper_action = action[-1]

        gripper_action = (gripper_action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

        # without sticky
        relative_gripper_action = -gripper_action
        # print(f"gripper_action B: {relative_gripper_action}, {self.sticky_action_is_on}")
        # if self.previous_gripper_action is None:
        #     relative_gripper_action = -1  # open
        # else:
        #     relative_gripper_action = -action
        # self.previous_gripper_action = action

        # switch to sticky closing
        # if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
        #     self.sticky_action_is_on = True
        #     self.sticky_gripper_action = relative_gripper_action

        # # sticky closing
        # if self.sticky_action_is_on:
        #     self.gripper_action_repeat += 1
        #     relative_gripper_action = self.sticky_gripper_action

        # # reaching maximum sticky
        # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
        #     self.sticky_action_is_on = False
        #     self.gripper_action_repeat = 0
        #     self.sticky_gripper_action = 0.0

        # print(f"gripper_action A: {relative_gripper_action}")

        action = np.concatenate(
            [
                action[:3],
                action_rotation_ax * action_rotation_angle,
                [relative_gripper_action],
            ]
        )

        return {
            "actions": action,
            "terminate_episode": outputs["terminate_episode"],
        }


from typing import Optional, Sequence

import matplotlib.pyplot as plt
import tensorflow as tf


class PolicyToSimpler:
    def __init__(self, adapter, policy):
        self.adapter = adapter
        self.policy = policy

    def reset(self, task_description: str):
        self.adapter.reset()

    def step(
        self, image: np.ndarray, eef_pos: np.ndarray, task_description: Optional[str] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        prompt = task_description
        inputs = self.adapter.preprocess(image, eef_pos, prompt)
        outputs = self.policy.infer(inputs)
        state_gripper = inputs["state"][-1]
        action_gripper = outputs["actions"][-1]
        print(f"state: {state_gripper} action: {action_gripper}")
        final_outputs = self.adapter.postprocess(outputs)
        simpler_outputs = {
            "world_vector": outputs["actions"][:3],
            "rot_axangle": outputs["actions"][3:6],
            "gripper": outputs["actions"][6:],
            "terminate_episode": outputs["terminate_episode"],
        }
        final_simpler_outputs = {
            "world_vector": final_outputs["actions"][:3],
            "rot_axangle": final_outputs["actions"][3:6],
            "gripper": final_outputs["actions"][6:],
            "terminate_episode": final_outputs["terminate_episode"],
        }
        return simpler_outputs, final_simpler_outputs

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(256, 256),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array([np.concatenate([a["world_vector"], a["rot_axangle"], a["gripper"]], axis=-1) for a in predicted_raw_actions])
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


from simpler_env.evaluation.argparse import get_args

# from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator


"""
Evaluate a model on ManiSkill2 environment.
"""

import os

import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video


def run_maniskill2_eval_single_episode(
    model,
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

    obj_variation_mode = "xy"
    # initialize environment
    # env_reset_options = {
    #     "robot_init_options": {
    #         "init_xy": np.array([robot_init_x, robot_init_y]),
    #         "init_rot_quat": robot_init_quat,
    #     }
    # }
    # if obj_init_x is not None:
    #     assert obj_init_y is not None
    #     obj_variation_mode = "xy"
    #     env_reset_options["obj_init_options"] = {
    #         "init_xy": np.array([obj_init_x, obj_init_y]),
    #     }
    # else:
    #     assert obj_episode_id is not None
    #     obj_variation_mode = "episode"
    #     env_reset_options["obj_init_options"] = {
    #         "episode_id": obj_episode_id,
    #     }
    obs, _ = env.reset()
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

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        joint_pos = obs["agent"]["qpos"]
        gripper_state = joint_pos[-2:].mean()

        ee_pose_world = env.agent.robot.get_links()[-1].get_pose()
        base_pose_world = env.agent.robot.get_links()[0].get_pose()
        ee_pose_rel = base_pose_world.inv() * ee_pose_world
        pos = ee_pose_rel.p  # [x, y, z]
        quat = ee_pose_rel.q  # [qw, qx, qy, qz]

        eef_pos = np.concatenate([pos, quat, [gripper_state]])
        # eef_pos = obs["agent"]["eef_pos"]  # obs["agent"].keys() has no "eef_pos"

        raw_action, action = model.step(image, eef_pos, task_description)
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

        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path = "data"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
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
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr


from abc import abstractmethod
from typing import Dict

import tree


class ActionChunkBroker:
    def __init__(self, policy, action_horizon: int):
        self._policy = policy

        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class AiroaBasePolicy:
    @abstractmethod
    def infer(self, obs: Dict) -> Dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class OpenpiToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy

    def infer(self, obs: Dict) -> Dict:
        outputs = self.policy.infer(obs)
        outputs["terminate_episode"] = np.zeros(outputs["actions"].shape[0])
        return outputs

    def reset(self) -> None:
        self.policy.reset()


if __name__ == "__main__":
    args = get_args()

    # # Fractal Debug
    # adapter = OpenpiSimplerFractalAdapter(
    # )

    # policy = _policy_config.create_trained_policy(
    #     _config.get_config("pi0_fractal_low_mem_finetune"),
    #     "checkpoints/pi0_fractal_low_mem_finetune2/my_experiment/17000",
    # )

    # Bridge Debug
    adapter = OpenpiSimplerBridgeAdapter()

    policy = _policy_config.create_trained_policy(
        _config.get_config("pi0_bridge_low_mem_finetune"),
        "/data/checkpoints/21000/",
    )

    policy = ActionChunkBroker(
        policy=policy,
        action_horizon=10,
    )

    policy = OpenpiToAiroaPolicy(
        policy=policy,
    )

    evaluter_policy = PolicyToSimpler(
        adapter=adapter,
        policy=policy,
    )

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(evaluter_policy, args)
    print(args)
    print(" " * 50, "Average success", np.mean(success_arr))
