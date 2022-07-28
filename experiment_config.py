# TODO structure experiment configuration (e.g. Dataclass or JSON structure)
import json

import gym
import stable_baselines3
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

import gym_minigrid
from gym_minigrid.envs import RiskyPathEnv
from gym_minigrid.envs.risky import DEFAULT_REWARDS
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, TensorObsWrapper

# Logik: benutze Gri

class GridworldExperiment:
    SEEDS = (763, 4744, 5672, 4267, 3377, 4356, 2689, 2819, 5224, 529)

    def __init__(
        self,
        exp_id: str,
        max_steps: int = 150,
        slip_probability: float = 0.,
        wall_rebound: bool = False,
        spiky_active: bool = False,
        reward_spec: dict = DEFAULT_REWARDS,
        show_agent_dir: bool = False,
        lava_positions: list[tuple] = None,
        agent_start_pos: tuple = (2,9),
        goal_positions: list[tuple] = [(1,3)],
        spiky_positions: list[tuple] = None
    ):
        """Initialize the experiment with environment-specific
        information. Algorithm configuration is seperate from the object
        initialization, as this leads to reusable environment configurations
        when rerunning the same experiment with different algorithm parameters.

        Args:
            exp_id (str): the unique experiment identifier
            max_steps (int): the number of time steps before episode end
            slip_probability (float): agent slipping probability -> [0, 1)
            wall_rebound (bool): activate rebound against wall
            spiky_active (bool): whether spiky tiles should be active
            reward_spec (dict): the dictionary containing the reward model
            show_agent_dir (bool): only relevant for rgb_array observation
            lava_positions (list[tuple], optional): specify the lava positions
            agent_start_pos (tuple): the agent's starting position
            goal_positions (list[tuple]): specify the goal positions
            spiky_positions (list[tuple], optional): the spiky tile positions
        """
        self.experiment_id = exp_id

        # environment spec
        self.max_steps = max_steps
        self.slip_proba = slip_probability
        self.wall_rebound = wall_rebound
        self.spiky_active = spiky_active
        self.reward_spec = reward_spec
        self.show_agent_dir = show_agent_dir
        self.agent_start_pos = agent_start_pos
        self.goal_positions = goal_positions
        self.agent_start_pos = agent_start_pos
        self.goal_positions = goal_positions
        self.lava_positions = lava_positions
        self.spiky_positions = spiky_positions

        # if observation_type == "tensor":
        #     env = TensorObsWrapper(env)
        # elif observation_type == "rgb_array":
        #     env = RGBImgObsWrapper(env, self.tile_size_px)
        #     env = ImgObsWrapper(env)
        # self.env = env

    def add_dqn_config(
        self,
        config_name : str,
        **kwargs
    ):
        """Configure the parameters that will be passed to the DQN algorithm.
        The kwargs should correspond to the DQN arguments. The environment is
        automatically added to the configuration.
        """
        self.dqn_kwargs = kwargs
        self.dqn_config_name = config_name

    def add_a2c_config(
        self,
        config_name : str,
        **kwargs
    ):
        """Configure the parameters that will be passed to the a2c algorithm
        """
        self.a2c_kwargs = kwargs
        self.a2c_config_name = config_name

    def run_experiment(
        self,
        num_runs: int,
        algo: str,
        observation_type : str = "tensor",
        tile_size_px: int = None,
        save_log: bool = True,
        log_directory: str = "saved_logs/",
        save_model: bool = True,
        model_directory: str = "saved_models/"
    ):
        # TODO handle model creation & deletion (memory saving) here
        # TODO handle environment creation, wrapping & deletion here
        # Model will need created environment when initialized
        # TODO write seed at the end of the log name

        # basic consistency checks
        assert num_runs <= len(self.SEEDS), \
            "Cannot execute more runs than specified random seeds"
        assert algo.lower() in ["a2c", "dqn"], \
            "Must use one of the algorithms 'a2c' or 'dqn'"
        assert observation_type.lower() in ["tensor", "rgb_array"], \
            "Observations must conform to 'tensor' or 'rgb_array'"

        if algo.lower() == "dqn":
            algo_config_name = self.dqn_config_name
        elif algo.lower() == "a2c":
            algo_config_name = self.a2c_config_name

        # define path for saving 
        obs_type_path = "tensor_obs/" if observation_type == "tensor" \
             else f"pixel_obs_{tile_size_px}/"
        full_path_suffix = \
            self.experiment_id + "/" + obs_type_path + algo.lower() + \
            "/" + algo_config_name + "/"

        # make environment
        env = gym.make(
            "MiniGrid-RiskyPath-v0",
            
        )


        # initialize model


        # TODO Save run with seed as suffix

        
        

    @classmethod
    def load_env_json_config(
        cls,
        json_path: str,
        keys: list[str] | str
    ):
        """Return an experiment object from a json configuration file.
        Keys may be passed in a list, which leads to returning a list
        of experiment objects.

        Args:
            json_path (str): the path to the json file
            keys (list[str] | str): the experiment id(s) to be passed
        """
        # Convenience checking for input
        if isinstance(keys, str):
            internal_keys = [keys]
        else:
            internal_keys = keys
        
        exp_objects = []

        # Initialize the experiment objects
        with open(json_path, "r") as file:
            deserialized = json.load(file)
            for key in internal_keys:
                new_exp = cls(key, **deserialized[key])
                exp_objects.append(new_exp)

        if len(exp_objects) > 1:
            return exp_objects
        elif len(exp_objects) == 0:
            return exp_objects[0]
        return None
