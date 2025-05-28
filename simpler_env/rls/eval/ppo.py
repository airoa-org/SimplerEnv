import time
import os

import torch
from tqdm import trange
import mediapy as media
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

from simpler_env.rls.envs import make_env
from simpler_env.rls.modules import Pi0PPOTorchRLModule


def main():
    VIDEO_NUM = 2
    NUM_EPISODES = 100
    model_path = f"{os.path.expanduser('~')}/SimplerEnv/results/rllib_test/20250524_151232"
    
    assert VIDEO_NUM <= NUM_EPISODES
    os.makedirs(f"{model_path}/video", exist_ok=True)

    # Initialize Ray
    ray.init(include_dashboard=False, ignore_reinit_error=True)

    # Register the environment
    register_env(
        "SimplerEnv",
        make_env,
    )

    # Load the trained model
    algo = Algorithm.from_checkpoint(f"{model_path}/checkpoints")
    rl_module = algo.get_module("default_policy")
    rl_module.eval() 
    action_dist_cls = rl_module.get_inference_action_dist_cls()

    # Create the environment
    env = make_env({
        "env_id": "google_robot_pick_coke_can",
        "simpler_env_rgb_observation_wrapper": True,
    })
    
    # Evaluate the model
    returns = []
    lens = []
    s = time.time()
    for i in trange(NUM_EPISODES):
        obs, _ = env.reset()
        success = truncated = False
        ep_return = 0.0
        ep_len = 0
        images = []

        while not (success or truncated):
            if i < VIDEO_NUM:
                image = (obs * 255.0).astype(np.uint8)
                images.append(image)

            # Get the action from the RL module
            torch_obs_batch = torch.from_numpy(obs).unsqueeze(0).to(torch.float32)
            action_logits = rl_module.forward_inference({"obs": torch_obs_batch})["action_dist_inputs"]
            action = action_dist_cls.from_logits(action_logits).sample()
            
            # Take a step in the environment
            obs, reward, success, truncated, _ = env.step(action.cpu().numpy().reshape(-1,))
            ep_return += reward
            ep_len += 1
        
        # Recode the video
        if i < VIDEO_NUM:
            image = (obs * 255.0).astype(np.uint8)
            images.append(image)
            media.write_video(f"{model_path}/video/epi_{i}.mp4", images, fps=5)

        # Record the episode length and return
        returns.append(ep_return)
        lens.append(ep_len)

    print(f"\nMean return over {num_episodes} episodes: {sum(returns)/len(returns):.2f}")
    print(f"Time: {(time.time() - s)/60:.2f} [min]")

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
