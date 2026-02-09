from .train_vocabularisers import train_vocabulariser
from ..tokenizers.tokenizers import get_tokenizers
from ..utils.training_data_utils import get_ff_by_path, get_crosslingual_homographs
from src.utils.results_controller import get_results_directory


class Trial:

    def __init__(self, arti_vocabulariser, tokenizers, ff_data_path, l2, algo, vocab_size):
        self.arti_vocabulariser = arti_vocabulariser
        self.tokenizers = tokenizers
        self.vocab_size = vocab_size
        self.ff_data = get_ff_by_path(ff_data_path)
        self.l2 = l2
        self.algo = algo
        self.homographs = get_crosslingual_homographs("en", l2)
        self.results_directory = get_results_directory(self.vocab_size, self.l2, self.algo)

    def get_base_tokenizers(self):
        return self.tokenizers[:3]

    def get_cued_tokenizer(self):
        return self.tokenizers[3]

    def get_base_vocabulariser(self):
        return self.arti_vocabulariser[:3]

    def get_cued_vocabulariser(self):
        return self.arti_vocabulariser[3]

    def get_ff(self):
        ff_words = set()
        for i in range(len(self.ff_data)):
            ff_words.add(self.ff_data[i]["False Friend"])
        return ff_words

    def get_homographs(self):
        return self.homographs

    def get_algo(self):
        return self.algo

    def get_results_directory(self):
        return self.results_directory

    def get_l2(self):
        return self.l2


def get_trial(algo, l2, vocab_size, l1_corpus_path, l2_corpus_path, l1_l2_corpus_path, cues_corpus_path):
    l2_artifacts, l2_v = train_vocabulariser(algo, l2, vocab_size, l2_corpus_path)
    l1_artifacts, l1_v  = train_vocabulariser(algo, "en", vocab_size, l1_corpus_path)
    l1_l2_artifacts, l1_l2_v = train_vocabulariser(algo, f"en_{l2}", vocab_size, l1_l2_corpus_path)
    cues_artifacts, cues_v = train_vocabulariser(algo, f"en_{l2}", vocab_size, cues_corpus_path)
    return [(l1_artifacts, l1_v), (l2_artifacts, l2_v), (l1_l2_artifacts, l1_l2_v), (cues_artifacts, cues_v)]

def init_trials(data):
    algorithms = data['algos']
    vocab_size = data['vocab_size']
    l1_data = data['l1']
    l2_data = data['l2']
    all_trials = {}
    for i in range(len(l2_data)):
        cur_l2_data = l2_data[i]
        all_trials[cur_l2_data['language']] = {}
        for algo in algorithms:
            trial = get_trial(algo, cur_l2_data["language"], vocab_size, l1_data["training_data"],
                              cur_l2_data["training_data"], cur_l2_data["multilingual_training_data"],
                              cur_l2_data['training_data_cues'])
            all_trials[cur_l2_data['language']][algo] = trial

    return all_trials

def get_lang_data(data, lang):
    for d in data['l2']:
        if d['language'] == lang:
            return d

def get_all_trials(data):
    all_trials = init_trials(data)
    all_tokenizers = get_tokenizers(all_trials)

    encapsulated_trials = {}
    for lang in all_trials.keys():
        encapsulated_trials[lang] = {}
        for algo in all_trials[lang].keys():
            cur_trial = all_trials[lang][algo]
            cur_tokenizers = all_tokenizers[lang][algo]
            lang_data = get_lang_data(data, lang)
            encapsulated_trials[lang][algo] = Trial(cur_trial, cur_tokenizers, lang_data['ff'], lang, algo, data['vocab_size'])

    return encapsulated_trials



