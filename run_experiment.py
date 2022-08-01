import argparse

from experiment_config import GridworldExperiment

parser = argparse.ArgumentParser(
    description="Run experiments specified by the experiment_config file"
)

# Per command, one experiment is executed
parser.add_argument(
    '--exp',
    type=str,
    default="exp_001",
    help="The json key in the experiment config for the environment parameters"
)

parser.add_argument(
    '--algo',
    type=str,
    default="both",
    choices=["a2c", "dqn", "both"],
    help="The algorithm with which to compute the experiment"
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
    '--no_model',
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
# TODO should test both dqn and a2c on tensor and rgb output and rgb with cuda

parser.add_argument(
    '--force_cuda',
    action='store_true',
    help="Will enforce usage of cuda if possible, AssertionError if not"
)

# TODO provide argument to run a hyperparameter tuning and return the results

# NOTE optional: specify new directory for logs/models
# NOTE optional: sequential vs parallel execution?


def check_args(args: argparse.Namespace):
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
        model_directory="test_model/"
    )
    print("Executed DQN tensor observation run")
    
    exp.add_a2c_config("conf_1")
    exp.run_experiment(
        1,
        "a2c",
        50_000,
        log_directory="test_log/",
        model_directory="test_model/"
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
        force_cuda=args.force_cuda
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
        force_cuda=args.force_cuda
    )
    print("Executed a2c rgb_array observation run")

    print("Successfully executed runs!")

    

def exec_experiments():
    pass

if __name__ == "__main__":
    args = parser.parse_args()

    check_args(args)

    if args.test_run:
        exec_test_run()
    
