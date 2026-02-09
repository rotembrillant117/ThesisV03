from src.stats.basic_stats import *
from src.vocabularisers.trial import *
from .stats_utils import get_categories






def run_stats(all_trials, vocab_size):

    for lang in all_trials.keys():
        for algo in all_trials[lang].keys():
            cur_trial = all_trials[lang][algo]
            categories = get_categories(cur_trial)
            tok_cases = tokenization_cases(cur_trial.get_base_tokenizers(), cur_trial.get_ff(), "en", cur_trial.get_l2(),categories)
            plot_tokenization_cases(tok_cases, algo, "en", lang, categories,"ff", cur_trial.get_results_directory())

