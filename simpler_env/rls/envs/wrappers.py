import gymnasium as gym
import numpy as np
    

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
