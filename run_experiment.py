import argparse

parser = argparse.ArgumentParser(
    description="Run experiments specified by the experiment_config file"
)

parser.add_argument()

# TODO args here, ideas below with question marks at the end of line
# TODO which experiments exactly, e.g. 'all' or comma-separated list?
# TODO overwrite existing logs/models?
# TODO specify new directory for logs/models
# TODO test mode to check everything works fine (cuda/cpu time benchmark)?
# TODO sequential vs parallel execution?
# TODO number of runs (dependent on seeds, seed should be in run name)?
# TODO specify whether to only run one algorithm in particular?
# TODO specify which algorithm configuration should be used (should be json and not optional!)

# TODO NOTE Maybe add notification system for Phone?

args = parser.parse_args()

