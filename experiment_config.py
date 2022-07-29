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
        **kwargs
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
        self.env_kwargs = kwargs

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
        total_timesteps: int,
        observation_type : str = "tensor",
        policy_type : str = "MlpPolicy",
        tile_size_px: int = None,
        save_log: bool = True,
        log_directory: str = "saved_logs/",
        save_model: bool = True,
        model_directory: str = "saved_models/"
    ):
        """Run the experiment with preset algorithm configurations for a
        certain number of runs. The path to the saved log and model are
        automatically generated. Each individual run is saved in the folder
        corresponding folder structure and is saved with the seed in the
        filename for purposes of reproducibility.

        Args:
            num_runs (int): Number of runs, should be maximally 10.
            algo (str): Either 'dqn' or 'a2c'
            total_timesteps (int): Number of timesteps to learn for each run
            observation_type (str, optional): Either 'tensor' or 'rgb_array'
            policy_type (str, optional): Either 'MlpPolicy' or 'CnnPolicy'
            tile_size_px (int, optional): Set this render size for cnn policy
            save_log (bool, optional): Whether to save a tensorboard log
            log_directory (str, optional): The log folder (e.g. "saved_logs/")
            save_model (bool, optional): Whether to save the trained model
            model_directory (str, optional): The model folder ("saved_models/")
        """    


        # basic consistency checks
        assert num_runs <= len(self.SEEDS), \
            "Cannot execute more runs than specified random seeds"
        assert algo.lower() in ["a2c", "dqn"], \
            "Must use one of the algorithms 'a2c' or 'dqn'"
        assert observation_type.lower() in ["tensor", "rgb_array"], \
            "Observations must conform to 'tensor' or 'rgb_array'"
        assert policy_type in ["MlpPolicy", "CnnPolicy"], \
            "policy_type must conform to types defined by stable baselines 3"
        if policy_type == "CnnPolicy" \
        and observation_type.lower() == "rgb_array":
            assert isinstance(tile_size_px, int), \
                "Render size for tiles must be set when using rgb input to cnn"

        try:
            if algo.lower() == "dqn":
                algo_config_name = self.dqn_config_name
            elif algo.lower() == "a2c":
                algo_config_name = self.a2c_config_name
        except AttributeError as e:
            print(
                "\nWARNING: Try setting algorithm parameters with the " \
                + "add_<a2c, dqn>_config() methods before running " \
                + "an experiment.\n"
            )
            raise

        # define path for saving
        if save_log:
            obs_type_path = "tensor_obs/" if observation_type == "tensor" \
                else f"pixel_obs_{tile_size_px}/"
            full_path_suffix = \
                self.experiment_id + "/" + obs_type_path + algo.lower() + \
                "/" + algo_config_name + "/"

        for i in range(num_runs):

            # make environment
            env = gym.make(
                "MiniGrid-RiskyPath-v0",
                **self.env_kwargs
            )

            # wrap environment according to observation type
            if observation_type.lower() == "tensor":
                env = TensorObsWrapper(env)
            elif observation_type.lower() == "rgb_array":
                env = RGBImgObsWrapper(env, tile_size=tile_size_px)
                env = ImgObsWrapper(env)

            # initialize model
            try:
                if algo.lower() == "dqn":
                    model = DQN(
                        policy_type,
                        env,
                        **self.dqn_kwargs,
                        tensorboard_log=log_directory \
                            if log_directory is not None else None
                    )
                elif algo.lower() == "a2c":
                    model = A2C(
                        policy_type,
                        env,
                        **self.a2c_kwargs,
                        tensorboard_log=log_directory \
                            if log_directory is not None else None
                    )
            except AttributeError as e:
                print(
                    "\nWARNING: Try setting algorithm parameters with the " \
                    + "add_<a2c, dqn>_config() methods before running " \
                    + "an experiment.\n"
                )
                raise

            # learn model, save if necessary
            model.learn(
                total_timesteps,
                tb_log_name=full_path_suffix + f"seed_{self.SEEDS[i]}"
            )

            if save_model:
                model.save(
                    model_directory \
                    + full_path_suffix \
                    + f"seed_{self.SEEDS[i]}"
                )

            # explicitly remove model and environment
            del model
            del env
        

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
        elif len(exp_objects) == 1:
            return exp_objects[0]
        return None