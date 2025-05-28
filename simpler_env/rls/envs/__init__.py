import simpler_env
from simpler_env.rls.envs.wrappers import SimplerEnvRGBObservation, LerobotPI0Wrapper


def make_env(env_config):
    env_id = env_config.get("env_id")
    env = simpler_env.make(
        env_id,
    )
    
    if env_config.get("simpler_env_rgb_observation_wrapper", False):
        env = SimplerEnvRGBObservation(env)
    
    if env_config.get("lerobot_pi0_wrapper", False):
        env = LerobotPI0Wrapper(env, policy_setup=env_config.get("policy_setup"))
    
    return env
