import random
import re
import os
from training_data_utils import DATA_DIR

TRAIN_DATA_DIR = DATA_DIR / 'raw' /'training_data'

def clean_training_data():
    for dir in dirs:
        if dir == "words":
            continue
        for l_file in os.listdir(TRAIN_DATA_DIR / dir):
            if l_file.endswith(".txt"):
                file_path = TRAIN_DATA_DIR / dir / l_file
                clean_row_numbers(file_path)
                lower_case_corpus(file_path)

def get_directories(wd):
    dirs = []
    for x in os.listdir(wd):
        if os.path.isdir(f"{wd}/{x}"):
            dirs.append(x)
    return dirs

def create_multi_text_file(path1, path2, file_name, num_rows=300_000, seed=42):
    """
    Creates a .txt file that combines two different text files by randomly sampling half of the lines
    from each input file using a specific random seed.

    :param path1: Path to file of first language
    :param path2: Path to file of second language
    :param file_name: Name of the combined output file
    :param num_rows: Total number of rows in the output file (half from each input)
    :param seed: Random seed for reproducibility
    """
    rows_from_each = num_rows // 2

    with open(path1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    with open(path2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()

    random.seed(seed)
    sampled1 = random.sample(lines1, rows_from_each)
    random.seed(seed + 1)
    sampled2 = random.sample(lines2, rows_from_each)

    with open(file_name, 'w', encoding='utf-8') as f_out:
        f_out.writelines(sampled1 + sampled2)

def clean_row_numbers(file_path):
    pattern = re.compile(r'^\s*\d+\s*')

    # Read, clean, and overwrite
    with open(file_path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for line in lines:
            cleaned_line = pattern.sub('', line)
            f.write(cleaned_line)

def lower_case_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lowercase_content = content.lower()

    with open(path, "w", encoding="utf-8") as f:
        f.write(lowercase_content)

def get_corpus_path_pairs(english_corpus_path, dirs):
    corpus_path_pairs = []
    for dir in dirs:
        if dir == "words" or dir == "en":
            continue
        else:
            for l_file in os.listdir(TRAIN_DATA_DIR / dir):
                if l_file.endswith(".txt"):
                    file_path = TRAIN_DATA_DIR / dir / l_file
                    corpus_path_pairs.append([("en", english_corpus_path), (dir, file_path)])
    return corpus_path_pairs


dirs = get_directories(TRAIN_DATA_DIR)
clean_training_data()
corpus_path_pairs = get_corpus_path_pairs(TRAIN_DATA_DIR / 'en' / 'eng-simple_wikipedia_2021_300K-sentences.txt', dirs)
for corpus_pair in corpus_path_pairs:
    english_data = corpus_pair[0]
    l2_data = corpus_pair[1]
    file_name = TRAIN_DATA_DIR / l2_data[0] / f"{english_data[0]}_{l2_data[0]}.txt"
    create_multi_text_file(english_data[1], l2_data[1], file_name)




