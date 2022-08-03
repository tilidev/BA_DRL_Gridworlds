import argparse
import json

from experiment_config import GridworldExperiment

parser = argparse.ArgumentParser(
    description="Run experiments specified by the experiment_config file"
)

# Per command, one experiment is executed
#TODO implement
parser.add_argument(
    '--exp',
    type=str,
    default="exp_001",
    help="The json key in the experiment config for the environment parameters"
)

#TODO implement
parser.add_argument(
    '--algo',
    type=str,
    default="all",
    choices=["a2c", "dqn", "all"],
    help="The algorithm with which to compute the experiment"
)

# TODO implement
parser.add_argument(
    '--observations',
    type=str,
    default="all",
    choices=['tensor', 'rgb_array', 'all'],
    help="The type of observations on which to run the experiments"
)

#TODO implement
parser.add_argument(
    '--directional_agent',
    action='store_true',
    help="Will render the agent facing in its intended direction"
)

#TODO implement
parser.add_argument(
    '--nolog',
    action='store_true',
    help="Will not save the log file of the experiment"
)

#TODO implement
parser.add_argument(
    '--no_model',
    action='store_true',
    help="Will not save the agent's model"
)

#TODO implement
parser.add_argument(
    '--num_timesteps',
    type=int,
    default=250_000,
    help="The number of time steps for each run"
)

#TODO implement
parser.add_argument(
    '--num_runs',
    type=int,
    default=5,
    help="The number of model trainings per algorithm. Max is 10"
)

#TODO implement
parser.add_argument(
    '--algo_config_key',
    type=str,
    default=None,
    help="If provided, will take algorithm parameters from 'algo_config.json'"
)

parser.add_argument(
    '--test_run',
    action='store_true',
    help="Will check if everything works as expected and output logs & models"
)

parser.add_argument(
    '--force_cuda',
    action='store_true',
    help="Will enforce usage of cuda if possible, AssertionError if not"
)

# TODO implement
parser.add_argument(
    '--callback',
    type=str,
    default='progress',
    help="A callback to provide to model learning. " \
        + "Options are defined in experiment_config.py"
)

# TODO provide argument to run a hyperparameter tuning and return the results

# NOTE optional: specify new directory for logs/models
# NOTE optional: sequential vs parallel execution?
# TODO NOTE optional: Callbacks hinzufügen (stop training, eval_callback etc.)


def check_args(args: argparse.Namespace):
    # TODO check that experiment exists
    # TODO check that algo config can only be passed when only one algo is chosen
    # TODO check that algo config key exists when passed
    # TODO check that environment config and algo config are legal
        # check that algo config does not conflict with tensor and rgb observation
        # It should be possible to set policy kwargs but not to set a custom policy class

    pass

def exec_test_run():
    # load experiment 1
    exp: GridworldExperiment = \
        GridworldExperiment.load_env_json_config("env_config.json", "exp_001")

    import torch as th
    cuda_available = th.cuda.is_available()
    print(f"CUDA available? -> {cuda_available}")

    exp.add_dqn_config("conf_1")
    exp.run_experiment(
        1,
        "dqn",
        60_000,
        log_directory="test_log/",
        model_directory="test_model/",
        force_cuda=args.force_cuda,
        callback='progress'
    )
    print("Executed DQN tensor observation run")
    
    exp.add_a2c_config("conf_1")
    exp.run_experiment(
        1,
        "a2c",
        50_000,
        log_directory="test_log/",
        model_directory="test_model/",
        force_cuda=args.force_cuda,
        callback='progress'
    )
    print("Executed a2c tensor observation run")

    exp.add_dqn_config("conf_2")
    exp.run_experiment(
        1,
        "dqn",
        10_000,
        log_directory="test_log/",
        model_directory="test_model/",
        policy_type="CnnPolicy",
        observation_type="rgb_array",
        tile_size_px=8,
        force_cuda=args.force_cuda,
        callback='progress'
    )
    print("Executed dqn rgb_array observation run")

    exp.add_a2c_config("conf_2")
    exp.run_experiment(
        1,
        "a2c",
        10_000,
        log_directory="test_log/",
        model_directory="test_model/",
        policy_type="CnnPolicy",
        observation_type="rgb_array",
        tile_size_px=8,
        force_cuda=args.force_cuda,
        callback='progress'
    )
    print("Executed a2c rgb_array observation run")

    print("Successfully executed runs!")

def exec_experiments():
    
    # Initialize experiment object with key parameter
    exp : GridworldExperiment = \
        GridworldExperiment.load_env_json_config("env_config.json", args.exp)

    # initialize algorithm config
    if args.algo_config_key is None:
        algo_config = {}
        algo_config_name = "algo_default"
    else: 
        with open('algo_config.json', 'r') as algo_file:
            deserialized = json.load(algo_file)
            algo_config = deserialized[args.algo_config_key]
            algo_config_name = args.algo_config_key
    
    # set configuration for each algorithm
    # (no config can be passed when using both algorithms)
    exp.add_a2c_config(algo_config_name, **algo_config)
    exp.add_dqn_config(algo_config_name, **algo_config)

    if args.algo == 'all':
        for name in ['dqn', 'a2c']:
            exp.run_experiment(
                num_runs=args.num_runs,
                algo=name,
                total_timesteps=args.num_timesteps,
                # TODO NOT DONE --> Continue Here
            )
    else:
        exp.run_experiment(
            num_runs=args.num_runs,
            algo=args.algo,
            total_timesteps=args.num_timesteps,
            # NOT DONE
        )
    

if __name__ == "__main__":
    args = parser.parse_args()

    check_args(args)

    if args.test_run:
        exec_test_run()
    else:
        exec_experiments()
