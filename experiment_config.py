import json
import os

import gym
import stable_baselines3
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from callback import InfoCallback

import gym_minigrid
from gym_minigrid.envs import RiskyPathEnv
from gym_minigrid.envs.risky import DEFAULT_REWARDS
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, TensorObsWrapper
from special_wrappers import IntrinsicMotivationWrapper, RandomizeGoalWrapper


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
            lava_positions (list[tuple]): specify the lava positions
            agent_start_pos (tuple): the agent's starting position
            goal_positions (list[tuple]): specify the goal positions
            spiky_positions (list[tuple]): the spiky tile positions
        """
        self.experiment_id = exp_id
        
        # fix experiment not working when no intrinsic motivation wrapping
        self.im_config = False

        # set goal-tile placement randomization
        self.goal_rnd = None
        if 'goal_rnd' in kwargs:
            self.goal_rnd = kwargs.pop('goal_rnd')

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
        assert 'policy' not in kwargs, \
            "Policy must be specified in run_experiment method"
        self.dqn_kwargs = kwargs
        self.dqn_config_name = config_name

    def add_a2c_config(
        self,
        config_name : str,
        **kwargs
    ):
        """Configure the parameters that will be passed to the a2c algorithm
        """
        assert 'policy' not in kwargs, \
            "Policy must be specified in run_experiment method"
        self.a2c_kwargs = kwargs
        self.a2c_config_name = config_name

    def add_im_config(
        self,
        config_name : str,
        **kwargs
    ):
        """Configure parameters passed for Intrinsic Motivation wrapper
        """
        assert "total_steps" not in kwargs, \
            "'total_steps' is automatically set during experiment execution"
        self.im_config = True
        self.im_config_name = config_name
        self.im_kwargs = kwargs
    
    def _check_existing_runs(self, log_path: str) -> list[int]:
        """Will return a list for seeds for which there are already logs in
        the given path. List is empty if there are no logs with the given seed.
        """
        res = []

        if os.path.exists(log_path):
            for seed in self.SEEDS:
                for entry in os.listdir(log_path):
                    # use _ to make sure seed is not itself in another seed
                    if f"_{seed}_" in entry:
                        res.append(seed)
        
        return res

    def run_experiment(
        self,
        num_runs: int,
        algo: str,
        total_timesteps: int,
        observation_type : str = "tensor",
        policy_type : str = "MlpPolicy",
        tile_size_px: int = None,
        directional_agent: bool = False,
        save_log: bool = True,
        log_directory: str = "saved_logs/",
        save_model: bool = True,
        model_directory: str = "saved_models/",
        force_cuda: bool = False,
        callback: str = 'progress',
        ignore_logs: bool = False
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
            observation_type (str): Either 'tensor' or 'rgb_array'
            policy_type (str): Either 'MlpPolicy' or 'CnnPolicy'
            tile_size_px (int, optional): Set tile render size for cnn policy
            save_log (bool): Whether to save a tensorboard log
            log_directory (str): The log folder (e.g. "saved_logs/")
            save_model (bool): Whether to save the trained model
            model_directory (str): The model folder ("saved_models/")
            force_cuda (bool): Force execution with cuda (else error)
            callback (str): #TODO
            ignore_logs (bool): Will still execute already existing runs
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
        if policy_type == "CnnPolicy":
            assert observation_type.lower() == "rgb_array", \
                "Cnn only works on Images"
            assert isinstance(tile_size_px, int), \
                "Render size for tiles must be set when using rgb input to cnn"

        try:
            if algo.lower() == "dqn":
                algo_config_name = self.dqn_config_name
            elif algo.lower() == "a2c":
                algo_config_name = self.a2c_config_name
        except AttributeError as e:
            # Attribute error raised when no algorithm configuration was passed
            print(
                "\nWARNING: Try setting algorithm parameters with the " \
                + "add_<a2c, dqn>_config() methods before running " \
                + "an experiment.\n"
            )
            raise

        # update env kwargs for agent directionality
        self.env_kwargs.update({"show_agent_dir" : directional_agent})

        available_seeds = self.SEEDS

        # define path for saving
        if save_log:
            obs_type_path = "tensor_obs/" if observation_type == "tensor" \
                else f"pixel_obs_{tile_size_px}/"
            full_path_suffix = \
                self.experiment_id + "/" + obs_type_path + algo.lower() + \
                "/" + algo_config_name + "/"

            if self.im_config:
                full_path_suffix += self.im_config_name + "/"
            
            if not ignore_logs:
                # check for are already existing logs for certain seeds
                pth = log_directory + full_path_suffix
                seeds_w_log = self._check_existing_runs(pth)
                if len(seeds_w_log) > 0:
                    seed_enumeration = ", ".join([str(s) for s in seeds_w_log])
                    print(f"Skipping seeds {seed_enumeration}")
                    if len(seeds_w_log) == len(self.SEEDS):
                        print(
                            "WARNING: There are already runs for each seed" \
                                + f" in directory {pth}."
                        )
                # reduce list of seeds to be used
                available_seeds = [x for x in self.SEEDS if x not in seeds_w_log]
                # available_seeds = list(set(self.SEEDS).difference(seeds_w_log))
        else:
            full_path_suffix = ""

        available_seeds = available_seeds[:num_runs]
        for s in available_seeds:

            # make environment
            env = gym.make(
                "MiniGrid-RiskyPath-v0",
                **self.env_kwargs
            )

            eval_env = gym.make(
                "MiniGrid-RiskyPath-v0",
                **self.env_kwargs
            )

            # wrap environment according to observation type
            if observation_type.lower() == "tensor":
                env = TensorObsWrapper(env)
                eval_env = TensorObsWrapper(eval_env)
            elif observation_type.lower() == "rgb_array":
                env = RGBImgObsWrapper(env, tile_size=tile_size_px)
                eval_env = RGBImgObsWrapper(eval_env, tile_size=tile_size_px)
                env = ImgObsWrapper(env)
                eval_env = ImgObsWrapper(eval_env)

                # apply same wrapping as base_algorithm for rgb inputs
                # eval_env must be VecTransposeImage
                eval_env = DummyVecEnv([lambda: eval_env])
                eval_env = VecTransposeImage(eval_env)
            # env = Monitor(env, info_keywords=("is_success",))

            # NOTE EvalEnv only evaluates performance on the original environment
            if self.im_config:
                env = IntrinsicMotivationWrapper(
                    env,
                    total_timesteps,
                    **self.im_kwargs
                )

            if self.goal_rnd is not None:
                env = RandomizeGoalWrapper(env, randomization=self.goal_rnd)
                # TODO evaluate eval_env on best model with RandomizeGoalWrapper
            
            # initialize model
            try:
                if algo.lower() == "dqn":
                    model = DQN(
                        policy_type,
                        env,
                        **self.dqn_kwargs,
                        tensorboard_log=log_directory \
                            if save_log else None,
                        seed=s
                    )
                elif algo.lower() == "a2c":
                    model = A2C(
                        policy_type,
                        env,
                        **self.a2c_kwargs,
                        tensorboard_log=log_directory \
                            if save_log else None,
                        seed=s
                    )
                print("Model seed:", model.seed)
            except AttributeError as e:
                print(
                    "\nWARNING: Try setting algorithm parameters with the " \
                    + "add_<a2c, dqn>_config() methods before running " \
                    + "an experiment.\n"
                )
                raise

            # check model
            print(f"Pytorch device: {model.device}")
            if force_cuda and model.device.type != "cuda":
                assert False, "Force Cuda execution but cuda not available"

            # save paths
            log_path = full_path_suffix + f"seed_{s}" \
                + ("_directional" if directional_agent else "")

            save_path_model = model_directory \
                + full_path_suffix \
                + f"seed_{s}" \
                + ("_directional" if directional_agent else "")

            # Initialize callback if provided
            if callback == 'progress':
                cb = InfoCallback(total_timesteps)
            elif callback == 'save_best':
                cb = [
                    EvalCallback(
                        eval_env=eval_env,
                        log_path=log_directory + log_path + "_evals",
                        best_model_save_path=save_path_model + "_best_model"
                    ),
                    InfoCallback(total_timesteps)
                ]
            else:
                raise NotImplementedError()

            print(f"Running experiment '{self.experiment_id}'")
            print(f"Saving tensorboard logs to: {log_directory+log_path}")

            # learn model, save if necessary
            model.learn(
                total_timesteps,
                tb_log_name=log_path,
                callback=cb
            )

            if save_model:
                print(f"Saving trained model to: {save_path_model}")
                model.save(save_path_model)

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