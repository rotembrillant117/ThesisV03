import matplotlib.pyplot as plt
from pathlib import Path


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

def write_tokenization_split(tokenizers, ff_data, l1, l2, algo, dir):
    """
    Writes the tokenization splits of different tokenizers to a .txt file
    :param tokenizers: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param ff_data: the ff data
    :param l1: language 1 (english)
    :param l2: language 2
    :param algo: the algorithm used
    :param dir: path to save .txt file
    :return:
    """
    with open(f"{dir}/{algo}.txt", 'w', encoding='utf-8') as f:
        f.write(f"{l1}_tokenizer, {l2}_tokenizer, {l1}_{l2}_tokenizer\n")
        for ff in ff_data:
            to_write = f""
            for t in tokenizers:
                to_write += f"{t.prepareAndTokenise(ff)}"
            to_write += "\n"
            f.write(to_write)

def do_basic_stats(trial, vocab_size):

    with open(Path(trial.get_stats_directory() / "basic_stats.txt"), 'w', encoding='utf-8') as f:
        lines = f.readlines()

