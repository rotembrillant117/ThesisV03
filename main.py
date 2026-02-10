import json
import sys
from src.vocabularisers.trial import get_all_trials
from src.utils.results_controller import create_results_directory
from src.stats.run_stats import run_stats

def parse_args(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    args_path = sys.argv[1]
    data = parse_args(args_path)
    create_results_directory(data)

    # 1. Train and Get All Trials
    print("--- Starting Training / Retrieving Trials ---")
    all_trials = get_all_trials(data)
    # run_stats(all_trials, data['vocab_size'])




