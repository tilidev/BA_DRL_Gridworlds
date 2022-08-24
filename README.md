# BA_Deep_Reinforcement_Learning

- [ ] Introduction here

## Quick Start

- [ ] Installation steps & tested versions
- [ ] Give different possibilites of using this repo:
    - [ ] Explain how to use the trained models
    - [ ] Explain how to follow the exact steps I made 

## Repository - Quick Overview

This subsection quickly introduces the repository and its contents for
better orientation. 

### Configuration files

Configuration files help setting up an experiment you want to execute. The environment, algorithm, intrinsic motivation ACHTUNG HIER EVENTUELL AUCH DISTRSHIFT
Specific configurations are used by setting the json-key as the corresponding command-line option when executing `run_experiment.py`.

- `algo_config.json`: The named json-configurations for algorithms to run. Currently only [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#) and [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) are supported. The configurations coincide with the arguments that can be passed to the `__init__()`-method, however some restrictions apply because of the experimental setup. These restrictions can be found in `experiment_config.py`.

- `env_config.json`: The named json-configurations

- `im_config.json`: TODO

### RiskyPathEnvironment - Gridworld Codebase

The adapted and modified Environment from HIER RATLIFF; ROBERT ZITIEREN has been implemented using the [Gym Minigrid](https://github.com/Farama-Foundation/gym-minigrid)-framework. Given that the framework did not meet certain requirements w.r.t. action space, observation space and rendering, I forked the project to create the "RiskyPathEnvironment".
The fork is referenced as a submodule in this repository, in order to keep the experimentation and Gridworld implementation seperate.

- `gym_minigrid/envs/risky.py`: This is the most important file for the environment. FERTIGSCHREIBEN

- `gym_minigrid/wrappers.py`: FERTIGSCHREIBEN

### Experiment Execution & Learning Scripts

- `experiment_config.py`: TODO

- `run_experiment.py`: TODO

- `callback.py`: TODO

- `special_wrappers.py`: TODO

### Testing & Evaluation

- `manual_control.py`: TODO

- `eval_agents.ipynb`: TODO

- `plots.ipynb`: TODO

## Experiment Structure

- [ ] Explain how I structured my experiments and describe them

The experiments can be broken down into two main phases. </br>
**Phase 1** consists of all experiments where the RiskyPathEnv is
used with the `TensorObsWrapper`. When the wrapper's `step()` method is called
by the SB3-algorithms (I use DQN and A2C), the returned observation is a
relatively low-dimensional input when compared to rgb-arrays that are
returned in the `ImgObsWrapper`. The effect of different environment
specifications on the learning algorithms' performance can thus be
analyzed in a more efficient way, as learning on this kind of input drastically
reduces complexity and computation time. </br>
**Phase 2** consists of all experiments where the observation is an rgb-array
(returned by the `ImgObsWrapper`). For learning on this kind of input, the
sb3-algorithms will use `CnnPolicy` TODO

Below you can find a listing of all experiments, annotated with the environment
settings and the learning & network parameters of the algorithm employed.
Furthermore, the path to the saved model (if available) and the tensorboard
logs are provided.