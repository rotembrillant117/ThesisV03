from tktkt.models.sage.inference import SageTokeniser
from tktkt.models.huggingface.bpe import HuggingFaceBPETokeniser
from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser

def get_tokenizers(all_trials):
    tokenizers = {}
    for language in all_trials.keys():
        tokenizers[language] = {}
        for algo in all_trials[language].keys():
            tokenizers[language][algo] = []
            for artifacts, vocabulariser in all_trials[language][algo]:

                if "SAGE" in algo:
                    tok = SageTokeniser(
                        preprocessor=vocabulariser.preprocessor,
                        vocab=artifacts.getVocabulary()
                    )
                elif "BPE" in algo:
                    tok = HuggingFaceBPETokeniser(
                        preprocessor=vocabulariser.preprocessor,
                        vocab=artifacts.getVocabulary(),
                        merges=artifacts.getMerges())
                else: # KUDO
                    tok = KudoPieceTokeniser(
                        preprocessor=vocabulariser.preprocessor,
                        model_file=artifacts.getModelFile())

                tokenizers[language][algo].append(tok)
    return tokenizers
