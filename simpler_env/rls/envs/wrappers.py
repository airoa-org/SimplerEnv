import gymnasium as gym
import numpy as np
from transformers import AutoTokenizer
from transforms3d.euler import euler2axangle
import torch
    

class SimplerEnvRGBObservation(gym.ObservationWrapper):
    def __init__(self, env, camera_type: str = "overhead_camera"):
        super().__init__(env)

        assert camera_type in ["base_camera", "overhead_camera"]
        self.camera_type = camera_type
    
    def observation(self, observation):
        rgb = observation["image"][self.camera_type]["rgb"]
        rgb = rgb.astype(np.float32) / 255.0
        return rgb
    
    @property
    def observation_space(self):
        image_observation_space = self.env.observation_space["image"][self.camera_type]["rgb"]
        assert image_observation_space.dtype == np.uint8
        return gym.spaces.Box(
            low=0,
            high=1,
            shape=image_observation_space.shape,
            dtype=np.float32,
        )


class LerobotPI0Wrapper(gym.Wrapper):
    def __init__(self, env, policy_setup: str, unnorm_key: str = None):
        super().__init__(env)

        assert policy_setup in ["google_robot", "widowx_bridge"]

        self.policy_setup = policy_setup
        self.unnorm_key = None # TODO いらないかも？
        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.tokenizer_max_length = 10 # TODO from config

        # TODO ここら辺の動作を確認
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_scale = 1.0 # TODO from args

        if self.policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig/1.0.0" if self.unnorm_key is None else self.unnorm_key
            self.sticky_gripper_num_repeat = 1
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
            self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        elif self.policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data/0.1.0" if self.unnorm_key is None else self.unnorm_key
            self.action_ensemble = True
            self.sticky_gripper_num_repeat = 10
        else:
            raise ValueError(f"Unsupported policy_setup: {self.policy_setup}")
    
    def reset(self, **kwargs):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        observation, info = self.env.reset(**kwargs)
        observation = self.convert_observation(observation)
        return observation, info
    
    def step(self, action):
        action = self.convert_action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.convert_observation(observation)
        return observation, reward, terminated, truncated, info
    
    def convert_action(self, action):
        action_dict = {
            "world_vector": action[:3],
            "rotation_delta": action[3:6],
            "open_gripper": action[6:],  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action_dict["world_vector"] = action_dict["world_vector"] * self.action_scale
        # action_rotation_delta = np.asarray(
        #     raw_action["rotation_delta"], dtype=np.float64
        # )
        action_rotation_delta = action_dict["rotation_delta"]
        roll, pitch, yaw = action_rotation_delta[0], action_rotation_delta[1], action_rotation_delta[2]
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)

        action_rotation_axangle = action_rotation_ax * action_rotation_angle 
        action_dict["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action_dict["gripper"] = 0
            current_gripper_action = action_dict["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action_dict["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action_dict["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        else:
            raise ValueError(f"Unknown policy setup: {self.policy_setup}")
        
        converted_action = np.concatenate(
            [
                action_dict["world_vector"],
                action_dict["rot_axangle"],
                action_dict["gripper"],
            ],
            axis=0
        )
        return converted_action

    def convert_observation(self, observation):

        if self.policy_setup == "widowx_bridge":
            image = observation["image"]["3rd_view_camera"]["rgb"]
            image = image.astype(np.float32) / 255.0
        elif self.policy_setup == "google_robot":
            image = observation["image"]["overhead_camera"]["rgb"]
            image = image.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown policy setup: {self.policy_setup}")

        eef_pos = observation["agent"]["eef_pos"].astype(np.float32)

        tokenized_task_instruction, mask = self.get_tokenized_task_instruction()

        converted_observation = {
            "image": image,
            "eef_pos": eef_pos,
            "task_instruction": tokenized_task_instruction,
            "task_instruction_mask": mask,
        }
        return converted_observation
    
    def get_tokenized_task_instruction(self):
        task_instruction = self.env.unwrapped.get_language_instruction()
        assert type(task_instruction) is str, f"Type of task_instruction should be str, but got {type(task_instruction)}"

        task_instructions = [task_instruction if task_instruction.endswith("\n") else f"{task_instruction}\n"]

        tokenized_prompt = self.language_tokenizer.__call__(
            task_instructions,
            padding="max_length",
            padding_side="right",
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        
        tokenized_task_instruction = tokenized_prompt["input_ids"][0]
        mask = tokenized_prompt["attention_mask"][0]

        return tokenized_task_instruction, mask
    
    @property
    def action_space(self):
        # Original action space
        # Box(
        #    [-1.        -1.        -1.        -1.5707964 -1.5707964 -1.5707964 -1.       ],
        #    [ 1.         1.         1.         1.5707964  1.5707964  1.5707964  1.       ],
        #    (7,),
        #    float32,
        #)
        return self.env.action_space # TODO

    @property
    def observation_space(self):
        if self.policy_setup == "widowx":
            image_observation_space = self.env.observation_space["image"]["3rd_view_camera"]["rgb"]
        elif self.policy_setup == "google_robot":
            image_observation_space = self.env.observation_space["image"]["overhead_camera"]["rgb"]
        else:
            raise ValueError(f"Unknown policy setup: {self.policy_setup}")

        assert image_observation_space.dtype == np.uint8, (
            f"Expected image observation space dtype to be np.uint8, but got {image_observation_space.dtype}"
        )

        return gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=1,
                shape=image_observation_space.shape,
                dtype=np.float32,
            ),
            "eef_pos": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(8,),
                dtype=np.float32,
            ),
            "task_instruction": gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.tokenizer_max_length,),
                dtype=np.int64,
            ),
            "task_instruction_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.tokenizer_max_length,),
                dtype=np.int64,
            ),
        })
