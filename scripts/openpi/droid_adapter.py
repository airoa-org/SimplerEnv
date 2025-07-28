import numpy as np
from typing import Dict
from simpler_env.evaluation.adapter import BaseAdapter
from simpler_env.utils.geometry import euler2axangle


class DroidAdapter(BaseAdapter):
    """
    Custom adapter for DROID policy evaluation
    Handles observation preprocessing for DROID-trained models
    """
    
    def __init__(self, policy):
        super().__init__(policy)
        self.sticky_gripper_num_repeat = 15
        
    def reset(self, task_description):
        """Reset adapter state for new episode"""
        # task_description is not used but required by interface
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        
    def preprocess(self, image: np.ndarray, eef_pos: np.ndarray, prompt: str) -> Dict:
        """
        Convert SimplerEnv observation format to DROID policy input format
        
        Args:
            image: RGB image (H, W, 3) in [0, 255]
            eef_pos: End-effector pose [x, y, z, qw, qx, qy, qz, gripper_width]
            prompt: Task description string
            
        Returns:
            Dict with keys expected by DROID policy
        """
        # Convert quaternion from wxyz (SimulationEnv) to xyzw (DROID)
        if len(eef_pos) >= 7:
            quat_xyzw = np.roll(eef_pos[3:7], -1)  # wxyz -> xyzw
            gripper_width = eef_pos[7] if len(eef_pos) > 7 else 0.5
        else:
            # Fallback if eef_pos is shorter than expected
            quat_xyzw = np.array([0, 0, 0, 1])  # identity quaternion
            gripper_width = 0.5
            
        # Convert gripper width to closedness for consistency with training data
        gripper_closedness = 1.0 - gripper_width
        
        # Create base proprioception state (8 dimensions for DROID)
        proprio_base = np.concatenate([
            eef_pos[:3],           # position (3D)
            quat_xyzw,             # quaternion (4D) 
            [gripper_closedness]   # gripper state (1D)
        ])
        
        # Note: This proprio is not used in the final output but kept for compatibility
        
        # DROID configuration: joint_position(7) + gripper_position(1) = 8 total
        # But model expects action_dim=32, so DroidInputs will pad 8->32
        joint_pos = proprio_base[:7]  # 7 elements for joint positions
        gripper_pos = np.array([gripper_closedness])  # 1 element for gripper
        
        # For direct state access (fallback) - will be padded to 32-dim by DroidInputs
        state_fallback = proprio_base  # Use 8-dim state, will be padded to 32
        
        # print(f"DEBUG DroidAdapter: joint_position shape: {joint_pos.shape}, gripper_position shape: {gripper_pos.shape}")
        # print(f"DEBUG DroidAdapter: state shape: {state_fallback.shape}")
        # print(f"DEBUG DroidAdapter: combined would be shape: {(len(joint_pos) + len(gripper_pos),)}")
        
        observation = {
            "observation/exterior_image_1_left": image,
            "observation/wrist_image_left": image,  # Use same image for both cameras
            "observation/joint_position": joint_pos,  # For DroidInputs
            "observation/gripper_position": gripper_pos,  # For DroidInputs
            "state": state_fallback,  # For FractalInputs - 7-dim for DROID
            "prompt": prompt,
        }
        
        # print(f"DEBUG DroidAdapter: returning observation with keys: {list(observation.keys())}")
        return observation
        
    def postprocess(self, outputs: Dict) -> Dict:
        """
        Convert DROID policy output to SimplerEnv action format
        """
        action = outputs["actions"]
        # print(f"DEBUG postprocess: action shape: {action.shape}")
        # print(f"DEBUG postprocess: action: {action}")
        # print(f"DEBUG postprocess: outputs keys: {outputs.keys()}")
        
        # Handle multi-timestep actions - use only first timestep
        if len(action.shape) > 1:
            action_first = action[0]
        else:
            action_first = action
            
        # Extract rotation (euler angles) and convert to axis-angle
        # For 32-dim padded DROID actions: [x, y, z, roll, pitch, yaw, gripper, 0, 0, ..., 0]
        # The actual DROID action is in the first 7-8 elements
        if len(action_first) >= 6:
            roll, pitch, yaw = action_first[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        else:
            action_rotation_ax = np.array([0, 0, 1])
            action_rotation_angle = 0
            
        # Extract gripper action and convert to SimplerEnv format
        # For DROID actions, gripper is at index 6 (7th element), even in padded 32-dim
        gripper_action = action_first[6] if len(action_first) > 6 else 0.0
        
        # Convert from [0,1] (close/open) to [-1,1] (close/open) for SimplerEnv
        gripper_action = (gripper_action * 2) - 1  # [0,1] -> [-1,1]
        
        # Apply sticky gripper logic (same as Fractal adapter)
        relative_gripper_action = -gripper_action
        
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action
            
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action
            
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
            
        # Combine position, rotation, and gripper actions
        final_action = np.concatenate([
            action_first[:3],  # position
            action_rotation_ax * action_rotation_angle,  # rotation as axis-angle
            [relative_gripper_action],  # gripper
        ])
        
        return {
            "actions": final_action,
            "terminate_episode": outputs.get("terminate_episode", False),
        }