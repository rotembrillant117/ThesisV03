from train_vocabularisers import train_vocabulariser

def get_trial(algo, l2, vocab_size, l1_corpus_path, l2_corpus_path, l1_l2_corpus_path, cues_corpus_path):
    l2_artifacts, l2_v = train_vocabulariser(algo, l2, vocab_size, l2_corpus_path)
    l1_artifacts, l1_v  = train_vocabulariser(algo, "en", vocab_size, l1_corpus_path)
    l1_l2_artifacts, l1_l2_v = train_vocabulariser(algo, f"en_{l2}", vocab_size, l1_l2_corpus_path)
    cues_artifacts, cues_v = train_vocabulariser(algo, f"en_{l2}", vocab_size, cues_corpus_path)
    return [(l1_artifacts, l1_v), (l2_artifacts, l2_v), (l1_l2_artifacts, l1_l2_v), (cues_artifacts, cues_v)]

def get_all_trials(data):
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
                              cur_l2_data["training_data"], cur_l2_data["multilingual_training_data"], cur_l2_data['training_data_cues'])
            all_trials[cur_l2_data['language']][algo] = trial

    return all_trials