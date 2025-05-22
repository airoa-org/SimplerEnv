import simpler_env
from simpler_env.rls.envs.wrappers import SimplerEnvRGBObservation


def make_env(env_config):
    env_id = env_config.get("env_id")
    env = simpler_env.make(
        env_id,
    )
    
    if env_config.get("simpler_env_rgb_observation_wrapper", False):
        env = SimplerEnvRGBObservation(env)
    
    return env
