from tktkt.factories.preprocessors import ModernEnglishPreprocessor_SentencePieceCompatible, KudoSpaceMarker
from src.vocabularisers.xSageVocabulariser import xSageVocabulariser
from src.vocabularisers.xBPEVocabulariser import xBPEVocabulariser
from src.vocabularisers.xKudoPieceVocabulariser import xKudoVocabulariser
from src.utils.training_data_utils import load_local_corpus_to_hf
import sys
import json

def parse_args(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def train_vocabulariser(algo, language, vocab_size, training_data_path):

    corpus_ds = load_local_corpus_to_hf(training_data_path)
    preprocessor = ModernEnglishPreprocessor_SentencePieceCompatible(marker_location=KudoSpaceMarker.location)
    if "SAGE" in algo:
        base_algo = algo.split("_")[0]
        base_artifacts = train_vocabulariser(base_algo, language, vocab_size, training_data_path)
        vocabulariser = xSageVocabulariser(base_artifacts, vocab_size, language, base_algo)
    elif "BPE" in algo:
        vocabulariser = xBPEVocabulariser(preprocessor, vocab_size, language)
    else: #KUDO
        vocabulariser = xKudoVocabulariser(preprocessor, vocab_size, language)

    results = vocabulariser.vocabulariseFromHf(corpus_ds, text_field="text")
    return results

if __name__ == "__main__":
    args_path = sys.argv[1:][0]
    data = parse_args(args_path)
    algorithms = data['algos']
    vocab_size = data['vocab_size']
    # Training l1 vocabularisers
    l1_data = data['l1']
    for algo in algorithms:
        train_vocabulariser(algo, l1_data['language'], vocab_size, l1_data['training_data'])

    # Training l2 vocabularisers and multilingual vocabularisers
    l2_data = data['l2']
    for i in range(1):
        for algo in algorithms:
            train_vocabulariser(algo, l2_data[i]['language'], vocab_size, l2_data[i]['training_data'])
            train_vocabulariser(algo, f"{l1_data['language']}_{l2_data[i]['language']}", vocab_size, l2_data[i]['multilingual_training_data'])


