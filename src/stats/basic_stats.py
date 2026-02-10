import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.unicode import get_language_map


def tokenization_cases(tokenizers_list, word_list, l1, l2, categories):
    """
    This function computes an analysis on how different tokenizers split words
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param word_list: list of words
    :param l1: the first language
    :param l2: the second language
    :param categories: tokenization cases
    :return: dictionary of {tokenization_case : [list of words]}
    """
    # init cases with value 0
    num_tokens_diff = {k: [] for k in categories}

    for word in word_list:
        word_tokenization = []
        num_tokens = []
        for t in tokenizers_list:
            res = t.prepareAndTokenise(word)
            word_tokenization.append(res)
            num_tokens.append(len(res))
        # Same splits throughout all tokenizers
        if word_tokenization[0] == word_tokenization[2] and word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff["same_splits"].append(word)
        # Same tokenization between language1 and multilingual tokenizer
        elif word_tokenization[0] == word_tokenization[2]:
            num_tokens_diff[f"{l1}_t==multi_t"].append(word)
        # Same tokenization between language2 and multilingual tokenizer
        elif word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff[f"{l2}_t==multi_t"].append(word)
        # All different tokenization
        elif word_tokenization[0] != word_tokenization[1] and word_tokenization[0] != word_tokenization[2] and \
                word_tokenization[1] != word_tokenization[2]:
            num_tokens_diff["different_splits"].append(word)
        # Same tokenization between language1 and langauge2, but different from Multi tokenizer
        elif word_tokenization[0] == word_tokenization[1]:
            num_tokens_diff[f"{l1}_t=={l2}_t"].append(word)
    return num_tokens_diff


def plot_tokenization_cases(num_tokens_diff, algo, l1, l2, categories, word_types, dir):
    """
    This function plots the tokenization cases
    :param num_tokens_diff: dictionary {tokenization_case: [list of words]}
    :param algo: algo name
    :param l1: language 1
    :param l2: language 2
    :param categories: tokenization cases
    :param word_types: False Friends words or other list of words
    :param dir: directory to save graph
    :return:
    """

    plt.figure(figsize=(15, 14))
    x_axis = categories
    y_axis = [len(num_tokens_diff[key]) for key in x_axis]
    distribution = [f"{key}: {len(num_tokens_diff[key])}" for key in x_axis]
    num_words = sum(y_axis)
    fig_save_path = f"{dir}/02_token_cases_{word_types}_{l1}_{l2}_{algo}.png"
    title = f"Tokenization Cases\n{l1}, {l2}\nAlgo: {algo}\nNum words: {num_words}\nDistribution: {distribution}"
    plt.bar(x_axis, y_axis)
    plt.xticks(rotation=30, fontsize=13)
    plt.xlabel("Tokenization Splits", fontsize=15)
    plt.ylabel("Amount of Tokenization Case", fontsize=15)
    plt.title(title, fontsize=18)
    plt.savefig(fig_save_path)
    plt.close()
    
def get_avg_token_length_over_vocab(artifact):
    """
    Calculates the average token length for tokenizer vocabulary
    :param artifact: the artifact object
    :return:
    """
    vocab = artifact.getVocabulary()
    num_chars = sum([len(v) for v in vocab])
    return num_chars / len(vocab)


def get_token_length_distribution(artifact):
    """
    Returns the token length distribution oh the tokens in the tokenizer vocabulary
    :param artifact: the artifact object
    :return:
    """
    vocab = artifact.getVocabulary()
    distribution = dict()
    for v in vocab:
        distribution[len(v)] = distribution.get(len(v), 0) + 1
    for k, v in distribution.items():
        distribution[k] = v / len(vocab)
    sorted_dis = {key: distribution[key] for key in sorted(distribution.keys())}
    return sorted_dis


def write_tokenization_split(tokenizers, ff_data, l2, file_handle):
    """
    Writes the tokenization splits of different tokenizers to a .txt file
    :param tokenizers: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer, cued tokenizer]
    :param ff_data: the ff data (list of words)
    :param l2: language 2
    :param file_handle: open file handle to write to
    :return:
    """
    lang_map = get_language_map()
    cue_map = lang_map.get(l2, {})

    file_handle.write(f"Word | L1_Tok | L2_Tok | Multi_Tok | Cued_Tok\n")

    for ff in ff_data:
        word = ff
        row = [word]

        # Base tokenizers (first 3)
        for t in tokenizers[:3]:
            row.append(str(t.prepareAndTokenise(word)))

        # Cued tokenizer (4th)
        if len(tokenizers) > 3:
            cued_tok = tokenizers[3]
            # Injecting cue to the first letter
            first_char = word[0]
            replacement = cue_map.get(first_char, first_char)
            cued_word = replacement + word[1:]

            row.append(str(cued_tok.prepareAndTokenise(cued_word)))

        file_handle.write(" | ".join(row) + "\n")


def do_basic_stats(trial, vocab_size):
    """
    Collects basic stats and writes them to a file in the trial's stats directory.
    """
    stats_dir = trial.get_stats_directory()

    stats_path = stats_dir / "basic_stats.txt"

    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Basic Stats for Algo: {trial.algo}, L2: {trial.l2}, Vocab Size: {vocab_size}\n")
        f.write("=" * 50 + "\n\n")

        # 1. Vocab Stats
        # trial.get_vocabularisers() returns list of (artifacts, vocabulariser) tuples
        vocab_info = trial.get_vocabularisers()

        names = ["L1 (en)", f"L2 ({trial.l2})", f"Multi (en_{trial.l2})", f"Cued (en_{trial.l2})"]

        for idx, (name, (artifact, _)) in enumerate(zip(names, vocab_info)):
            f.write(f"--- {name} ---\n")

            avg_len = get_avg_token_length_over_vocab(artifact)
            f.write(f"Avg Token Length (Vocab): {avg_len:.4f}\n")

            dist = get_token_length_distribution(artifact)
            # Find the most common lengths for a cleaner summary
            sorted_dist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
            top_5 = sorted_dist[:5]
            f.write(f"Top 5 Lengths (Length: Freq): {top_5}\n")
            f.write(f"Full Distribution: {dist}\n\n")

        # 2. Tokenization Splits
        f.write("=" * 50 + "\n")
        f.write("Tokenization Splits (False Friends)\n")
        f.write("=" * 50 + "\n")

        ff_list = sorted(list(trial.get_ff()))
        write_tokenization_split(trial.get_tokenizers(), ff_list, trial.l2, f)

