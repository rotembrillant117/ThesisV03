from src.vocabularisers.xSageVocabulariser import xSageVocabulariser
from src.vocabularisers.xBPEVocabulariser import xBPEVocabulariser
from src.vocabularisers.xKudoPieceVocabulariser import xKudoVocabulariser
from src.utils.training_data_utils import load_local_corpus_to_hf
from src.preprocessors.cue_preprocessor import CuePreprocessor, CuePrefab2
from tktkt.preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation
from tktkt.factories.preprocessors import ModernEnglishPreprocessor_SentencePieceCompatible

# def patched_train(actual_vocab_size, **remaining_arguments):
#
#     # We must explicitly set the arguments tktkt normally sets,
#     # And we MUST set split_by_unicode_script=False.
#
#     spm.SentencePieceTrainer.Train(
#         **remaining_arguments,
#         vocab_size=actual_vocab_size + 1,
#         hard_vocab_limit=True,
#         byte_fallback=False,
#         vocabulary_output_piece_score=True,
#         control_symbols=[],
#         user_defined_symbols=[],
#         bos_id=-1, eos_id=-1, pad_id=-1,
#         normalization_rule_name="identity",
#         add_dummy_prefix=False,
#         remove_extra_whitespaces=False,
#         split_by_whitespace=False,
#         split_by_unicode_script=False,  # Lets language cues merge
#         split_by_number=True,
#         split_digits=False,
#         allow_whitespace_only_pieces=False
#     )
# KudoPieceVocabulariser._callSentencePieceTrainer = staticmethod(patched_train)


def train_vocabulariser(algo, language, vocab_size, training_data_path):

    corpus_ds = load_local_corpus_to_hf(training_data_path)
    marker = BoundaryMarker("_", detached=False, location=BoundaryMarkerLocation.START)
    preprocessor = CuePrefab2(marker=marker)
    # preprocessor = CuePreprocessor(marker=marker)
    if "SAGE" in algo:
        base_algo = algo.split("_")[0]
        base_artifacts, _ = train_vocabulariser(base_algo, language, vocab_size*8, training_data_path)
        vocabulariser = xSageVocabulariser(base_artifacts, vocab_size, language, base_algo)
    elif "BPE" in algo:
        vocabulariser = xBPEVocabulariser(preprocessor, vocab_size, language)
    else: #KUDO
        vocabulariser = xKudoVocabulariser(preprocessor, vocab_size, language)

    results = vocabulariser.vocabulariseFromHf(corpus_ds, text_field="text")
    return results, vocabulariser


def train(data):

    algorithms = data['algos']
    vocab_size = data['vocab_size']
    # Training l1 vocabularisers
    l1_data = data['l1']
    for algo in algorithms:
        train_vocabulariser(algo, l1_data['language'], vocab_size, l1_data['training_data'])

    # Training l2 vocabularisers, multilingual vocabularisers and language cued vocabularisers
    l2_data = data['l2']
    for i in range(1):
        for algo in algorithms:
            train_vocabulariser(algo, l2_data[i]['language'], vocab_size, l2_data[i]['training_data'])
            train_vocabulariser(algo, f"{l1_data['language']}_{l2_data[i]['language']}", vocab_size, l2_data[i]['multilingual_training_data'])
            train_vocabulariser(algo, f"{l1_data['language']}_{l2_data[i]['language']}_cues", vocab_size, l2_data[i]['training_data_cues'])

