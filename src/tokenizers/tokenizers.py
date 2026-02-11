from tktkt.models.sage.inference import SageTokeniser
from tktkt.models.huggingface.bpe import HuggingFaceBPETokeniser
from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser
from tktkt.interfaces.tokenisers import TokeniserWithVocabulary, WithSpecials
from tktkt.models.sage.vocabularisation import SageVocabulariser


class FixedSageTokeniser(SageTokeniser):
    def __init__(self, preprocessor, vocab):
        init_vocab_hex = {
            SageVocabulariser._toHexString(t): i for t, i in vocab.items()
        }

        next_id = max(vocab.values()) + 1
        extended_vocab_dict = dict(vocab)

        for i in range(256):
            b_hex = bytes([i]).hex()
            if b_hex not in init_vocab_hex:

                # Check if it's a valid, printable character first
                char_str = None
                try:
                    # Only single-byte UTF-8 (ASCII 0-127) can be decoded directly from 1 byte
                    if i < 128:
                        c = chr(i)
                        # We might want to keep control chars as bytes, but letters/digits as chars
                        if c.isprintable() and c != " ":
                            char_str = c
                except:
                    pass

                token_str = char_str if char_str else f"<byte:{b_hex}>"

                # Assign to both hex map (for backend) and string map (for python wrapper)
                init_vocab_hex[b_hex] = next_id
                extended_vocab_dict[token_str] = next_id

                next_id += 1
        from tktkt.interfaces.identifiers import Vocab

        sorted_types = sorted(extended_vocab_dict.keys(), key=extended_vocab_dict.get)

        extended_vocab = Vocab(
            ordered_types=sorted_types,
            specials=vocab.specials,
            unk_id=vocab.UNK
        )

        TokeniserWithVocabulary.__init__(self, preprocessor=preprocessor, vocab=extended_vocab)

        from sage_tokenizer.model import SaGeTokenizer
        self.backend = SaGeTokenizer(initial_vocabulary=init_vocab_hex)

def get_tokenizers(all_trials):
    tokenizers = {}
    for language in all_trials.keys():
        tokenizers[language] = {}
        for algo in all_trials[language].keys():
            tokenizers[language][algo] = []
            for artifacts, vocabulariser in all_trials[language][algo]:

                if "SAGE" in algo:
                    tok = FixedSageTokeniser(
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
