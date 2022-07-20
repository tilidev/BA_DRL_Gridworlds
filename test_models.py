# TODO
# file for testing the models
import gym
from stable_baselines3 import DQN
import gym_minigrid
from gym_minigrid.wrappers import TensorObsWrapper
from gym_minigrid.envs import RiskyPathEnv
from time import sleep

env = gym.make("MiniGrid-RiskyPath-v0")
env = TensorObsWrapper(env)

model = DQN.load("saved_models/dqn_risky_0")

obs = env.reset()
for i in range(30):
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, done, info = env.step(action)
    print(reward)
    env.render('human', tile_size=32)
    sleep(0.5)
    if done:
        env.reset()