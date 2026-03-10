from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import tensorflow as tf
import datetime
from dotenv import load_dotenv

import wandb
import os
load_dotenv("../.env")

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY is None:
    raise ValueError("WANDB_API_KEY not found")

wandb.login(key=WANDB_API_KEY)
wandb.init(project="deeptactics-arena", sync_tensorboard=True)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=train_log_dir)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
