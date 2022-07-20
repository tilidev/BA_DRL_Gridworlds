from time import sleep

import gym_minigrid
from gym_minigrid.envs import RiskyPathEnv

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, TensorObsWrapper

# TODO restructure this file


seeds = [763, 4744, 5672, 4267, 3377]
seeds_2 = [4356, 2689, 2819, 5224, 529]

def five_runs():
    for i in range(len(seeds)):
        print("Running experiment nr.", i+1)

        env = gym.make('MiniGrid-RiskyPath-v0')

        # if input is tensor
        env = TensorObsWrapper(env)

        # if input is rgb-image
        #env = RGBImgObsWrapper(env)
        #env = ImgObsWrapper(env)

        model = DQN(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log="saved_logs/dqn_risky_log/",
            seed=seeds[i]
        )

        model.learn(total_timesteps=250_000, tb_log_name=f"run_{i}")

        model.save(f"saved_models/dqn_risky_{i}")

        print(f"Finished run nr. {i+1}")
        print("Sleeping for cooldown")

        sleep(30)

def one_run():
    env = gym.make('MiniGrid-RiskyPath-v0')

    # if input is tensor
    env = TensorObsWrapper(env)

    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="saved_logs/dqn_risky_log_test/",
        seed=seeds_2[0]
    )

    model.learn(total_timesteps=250_000, tb_log_name=f"run")

if __name__ == "__main__":
    one_run()
