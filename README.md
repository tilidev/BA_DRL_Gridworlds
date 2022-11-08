# BA_Deep_Reinforcement_Learning

- [ ] Introduction here

## Quick Start

- [ ] Installation steps & tested versions
- [ ] Give different possibilites of using this repo:
    - [ ] Explain how to use the trained models
    - [ ] Explain how to follow the exact steps I made

To get an overview of this project's structure, refer to the section [Quick Overview](##repository-quick-overview).

### Installation of packages

The recommended Python version for working with this codebase is 3.10.5. The repository contains a submodule implemented in [Gym Minigrid](https://github.com/Farama-Foundation/Minigrid) which expects at least version 3.5 or later.

The required packages to this repository (and subsequently, its submodule) are listed below:

**Conda** (tested on mac m1 with miniconda3):

```sh
conda create -n thesis_env python=3.10.5
conda activate thesis_env
conda install gym==0.21.0
conda install matplotlib==3.5.1
conda install stable-baselines3==1.1.0
conda install tensorboard==2.9.1
```

Additionally, you may install seaborn for plots in the jupyter notebooks:

```sh
conda install seaborn==0.11.2
```

**venv** (tested on linux and mac m1):

Careful, unlike conda, venv depends on the currently installed python version. To make sure which python version you are using, type `python3 --version` in your terminal.

```sh
python3 -m venv thesis_env
source thesis_env/bin/activate
python3 -m pip install gym==0.21.0
python3 -m pip install matplotlib==3.5.1
python3 -m pip install stable-baselines3==1.1.0
python3 -m pip install tensorboard==2.9.1
python3 -m pip install seaborn==0.11.2
```

For both approaches, all required dependencies should be automatically installed. If this is not the case, try to install the required packages as per the provided requirements files (linux or mac m1).

### Test Experiment

To see if all works as expected, execute the following command:

- [ ] INSERT COMMAND HERE

After execution finishes, two folder should have been created: `saved_logs` and `saved_models`. The saved tensorboard log can be analyzed in your browser as such:

- [ ] INSERT TENSORBOARD HERE

Check the trained model (visually):

- [ ] INSERT COMMAND WITH SEVERAL EPISODES HERE

or

- [ ] INSERT COMMAND WITH SEVERAL RENDERED EPISODES HERE

**Caution:** The trained model is not expected to solve the environment satisfactorily as this should only be a test of whether the scripts execute as expected.

## Repository - Quick Overview

This subsection quickly introduces the repository and its contents for better orientation.

### Configuration files

Configuration files help setting up an experiment you want to execute. The environment, algorithm, intrinsic motivation ACHTUNG HIER EVENTUELL AUCH DISTRSHIFT
Specific configurations are used by setting the json-key as the corresponding command-line option when executing `run_experiment.py`.

- `algo_config.json`: The named json-configurations for algorithms to run. Currently only [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#) and [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) are supported. The configurations coincide with the arguments that can be passed to the `__init__()`-method, however some restrictions apply because of the experimental setup. These restrictions can be found in `experiment_config.py`.

- `env_config.json`: The named json-configurations of the gridworld environment to use for an experiment. Includes things such as maximal number of timesteps, slipping factor, reward model, goal randomization etc.

- `im_config.json`: The named json-configurations for experiments that use the `IntrinsicMotivationWrapper` (defined in `special_wrapper.py`). Includes settings such as whether to use only (count-based) intrinsic rewards or a combination of extrinsic/intrinsic rewards and when to stop using intrinsic rewards etc. Currently, intrinsic motivation rewards can only be computed for tensor observations of the environment.

### RiskyPathEnvironment - Gridworld Codebase

The adapted and modified Environment from [Ratliff and Mazumdar](https://ieeexplore.ieee.org/abstract/document/8754789/?casa_token=TwnHBsmi0CUAAAAA:37JpJLQ_QGdvPa4KmarmReliknIH1IbbRKc6nTSARUPVfg7nEEt-oKdA24UTJoLH_rrRRXyPUA) has been implemented using the [Gym Minigrid](https://github.com/Farama-Foundation/gym-minigrid)-framework. Given that the framework did not meet certain requirements w.r.t. action space, observation space and rendering, I forked the project to create the "RiskyPathEnvironment".
The fork is referenced as a submodule in this repository, in order to keep the experimentation and Gridworld implementation seperate.

- `gym_minigrid/envs/risky.py`: This is the most important file and defines the environment's logic. For further reference, refer to my [Gym Minigrid fork](https://github.com/tilidev/gym_minigrid/tree/package_only)

- `gym_minigrid/minigrid.py`: Definitions of the internal workings of gym minigrid. This file has been slightly adapted to allow conformity with different action and observation spaces ([Gym Minigrid fork](https://github.com/tilidev/gym_minigrid/tree/package_only)).

- `gym_minigrid/wrappers.py`: The wrappers have been extended with the `TensorObsWrapper` which builds on the internal tensor representation logic defined in `risky.py`.

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

This 

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
