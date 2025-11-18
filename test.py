# load ppo_model_copy.zip and test the model
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium_template import MinimalEnv

# load the model
model = PPO.load('ppo_model_copy.zip')

# create the environment
env = MinimalEnv()

# test the model
model.test(env)
