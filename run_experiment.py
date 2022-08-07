import argparse
import json
import time

from experiment_config import GridworldExperiment

parser = argparse.ArgumentParser(
    description="Run experiments specified by the experiment_config file"
)

# Per command, one experiment (environment configuration) is executed
parser.add_argument(
    '--exp',
    type=str,
    default="exp_001",
    help="The json key in the experiment config for the environment parameters"
)

parser.add_argument(
    '--algo',
    type=str,
    default="all",
    choices=["a2c", "dqn", "all"],
    help="The algorithm with which to compute the experiment"
)

parser.add_argument(
    '--observations',
    type=str,
    default="all",
    choices=['tensor', 'rgb_array', 'all'],
    help="The type of observations on which to run the experiments"
)

parser.add_argument(
    '--tile_size_px',
    type=int,
    default=8,
    choices=[8, 16, 32],
    help="The pixel length and width for one tile. Only applies to rgb output"
)

parser.add_argument(
    '--directional_agent',
    action='store_true',
    help="Will render the agent facing in its intended direction"
)

parser.add_argument(
    '--nolog',
    action='store_true',
    help="Will not save the log file of the experiment"
)

parser.add_argument(
    '--nomodel',
    action='store_true',
    help="Will not save the agent's model"
)

parser.add_argument(
    '--num_timesteps',
    type=int,
    default=250_000,
    help="The number of time steps for each run"
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=5,
    help="The number of model trainings per algorithm. Max is 10"
)

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

parser.add_argument(
    '--ignore_existing_logs',
    action='store_true',
    help="Will start runs with the first seed even" \
        + " if corresponding logs already exists"
)

# TODO implement
parser.add_argument(
    '--callback',
    type=str,
    default='progress',
    help="A callback to provide to model learning. " \
        + "Options are defined in experiment_config.py"
)

parser.add_argument(
    '--job_name',
    type=str,
    default="unnamed",
    help="Will be printed to stdout for easy identification of logs"
)

# TODO provide argument to run a hyperparameter tuning and return the results

# NOTE optional: specify new directory for logs/models
# TODO NOTE optional: Callbacks hinzufügen (stop training, eval_callback etc.)


def check_args(args: argparse.Namespace):
    # check environment configuration exists
    with open('env_config.json', 'r') as f:
        env_configurations = json.load(f)
        assert args.exp in env_configurations, \
            "Experiment was not found in env_config.json"
        
        # check env configuration does not conflict with later methods
        assert 'show_agent_dir' not in env_configurations[args.exp], \
            "Agent directionality should only be set in the " \
            + "--directional_agent command line option"

    # there should not be more runs than specified seeds
    assert args.num_runs <= 10, \
        "You should maximally schedule 10 runs"

    # Check that algorithm configuration will only be applied to one algorithm
    assert (args.algo_config_key is None) or (args.algo != 'all'), \
        "--algo='all' is not allowed when an algorithm configuration is passed"
    
    # check that algo config key exists when passed
    with open('algo_config.json', 'r') as f:
        algo_configurations = json.load(f)
        k = args.algo_config_key
        if k is not None:
            assert k in algo_configurations, \
                "Algorithm configuration was not found in algo_config.json"

            # prevent algo config conflict with tensor and rgb observation
            assert 'policy' not in algo_configurations[k], \
                "Policy is set depending on the --observations option" \
                + " and should not be set in the algorithm configuration"
            
            assert 'seed' not in algo_configurations[k], \
                "RNG seeds are set during execution and should not be passed"

            assert 'device' not in algo_configurations[k], \
                "Device is automatically set during execution. " \
                + "The --force_cuda option can be used to ensure gpu execution"
            
            assert 'tensorboard_log' not in algo_configurations[k], \
                "Tensorboard directory is set during script execution"

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

    print(f"Job name: {args.job_name}")
    
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

    # Run experiments depending on algorithms and observation types
    if args.algo == 'all':
        algorithms = ('dqn', 'a2c')
    else:
        algorithms = (args.algo,)
    if args.observations == 'all':
        observation_types = ('tensor', 'rgb_array')
    else:
        observation_types = (args.observations,)
    rgb_kwargs = {
        "tile_size_px" : args.tile_size_px,
        "policy_type" : "CnnPolicy",
    }

    training_start = round(time.time())
    print("INFO: Starting training")
    for algo in algorithms:
        for obs_type in observation_types:
            exp.run_experiment(
                num_runs=args.num_runs,
                algo=algo,
                total_timesteps=args.num_timesteps,
                observation_type=obs_type,
                save_log=(not args.nolog),
                save_model=(not args.nomodel),
                force_cuda=args.force_cuda,
                callback=args.callback,
                directional_agent=args.directional_agent,
                ignore_logs=args.ignore_existing_logs,
                **(rgb_kwargs if obs_type == "rgb_array" else {})
            )
    training_exec_time = round(time.time()) - training_start

    print(f"INFO: Training terminated after {training_exec_time} seconds")
    underline_suffix = len(str(training_exec_time)) * "-"
    print(f"----------------------------------------{underline_suffix}")
    

if __name__ == "__main__":
    args = parser.parse_args()

    check_args(args)

    if args.test_run:
        exec_test_run()
    else:
        exec_experiments()
