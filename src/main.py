import json
import sys
from vocabularisers.train_vocabularisers import train
from tokenizers.tokenizers import get_tokenizers



def parse_args(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data





if __name__ == '__main__':
    args_path = sys.argv[1]
    data = parse_args(args_path)




