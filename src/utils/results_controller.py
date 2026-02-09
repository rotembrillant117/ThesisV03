from pathlib import Path

STATS_DIR = Path(Path(__file__).resolve().parent.parent.parent) / 'stats_results'

def create_results_directory(data):
    Path(STATS_DIR).mkdir(parents=True, exist_ok=True)
    vocab_size = data['vocab_size']
    algos = data['algos']
    # l1 = data['l1']['language']
    l2 = [data['l2'][i]['language'] for i in range(len(data['l2']))]
    for v in vocab_size:
        for l in l2:
            for algo in algos:
                Path(STATS_DIR / f"{v}" / f"{l}" / f"{algo}").mkdir(parents=True, exist_ok=True)

def get_results_directory():
    pass