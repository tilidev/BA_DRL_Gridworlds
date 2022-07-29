from time import sleep

from torch import device
from experiment_config import GridworldExperiment

import gym_minigrid
from gym_minigrid.envs import RiskyPathEnv

import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, TensorObsWrapper


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

def test_experiment_config():
    from experiment_config import GridworldExperiment
    load = GridworldExperiment.load_env_json_config
    exp : GridworldExperiment = load("env_config.json", "exp_001")
    exp.add_a2c_config("conf_1", verbose=1)
    exp.add_dqn_config("conf_1", verbose=1)
    exp.run_experiment(
        2,
        "dqn",
        60_000
    )


if __name__ == "__main__":
    test_experiment_config()
