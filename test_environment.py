from time import sleep
import gym

from gymminigrid.gym_minigrid import *

env = gym.make("MiniGrid-RiskyPath-v0", show_agent_dir=False, wall_rebound=True)

obs = env.reset()

for i in range(10):
    env.render()
    sleep(1)
    obs, reward, done, info = env.step(env.action_space.sample())

    if done:
        obs = env.reset()