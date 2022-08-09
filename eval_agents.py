import time

import argparse

import gym
import gym_minigrid
from gym_minigrid.envs import RiskyPathEnv

from stable_baselines3.a2c import A2C

from gym_minigrid.wrappers import TensorObsWrapper

parser = argparse.ArgumentParser(
    description="Script for evaluating and visualizing trained agents"
)

parser.add_argument(
    'path',
    type=str,
    help="Path to the trained model"
)

parser.add_argument(
    '--visualize',
    action='store_true'
)

parser.add_argument(
    '--env_config',
    type=str,
    # TODO
)

def visualize_agent_default(path):
    env = gym.make("MiniGrid-RiskyPath-v0")
    env = TensorObsWrapper(env)

    model = A2C.load(path)

    for i in range(5):
        done = False
        obs = env.reset()
        print(f"ep {i}")

        while not done:
            env.render(tile_size=32)
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, info = env.step(action)
            time.sleep(0.5)




if __name__ == "__main__":
    args = parser.parse_args()

    visualize_agent_default(args.path)



