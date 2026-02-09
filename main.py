import json
import sys
from src.vocabularisers.utils import get_all_trials
from src.tokenizers.tokenizers import get_tokenizers
from src.stats.pipeline_stats import run_cued_pipeline_inspection



def parse_args(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    args_path = sys.argv[1]
    data = parse_args(args_path)

    # 1. Train and Get All Trials
    print("--- Starting Training / Retrieving Trials ---")
    all_trials = get_all_trials(data)

    # 2. Get Tokenizers from Trials
    print("--- Instantiating Tokenizers ---")
    tokenizers = get_tokenizers(all_trials)

    # 3. Pipeline Inspection (Analysis)
    run_cued_pipeline_inspection(tokenizers)




