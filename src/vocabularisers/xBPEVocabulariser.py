from tktkt.models.bpe.vocabularisation import BPEVocabulariser, BpeTrainerImplementation
from tktkt.interfaces.preprocessors import Preprocessor
from tktkt.util.strings import shash


class xBPEVocabulariser(BPEVocabulariser):

    def __init__(self, preprocessor: Preprocessor, vocab_size, language):

        super().__init__(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            implementation=BpeTrainerImplementation.SENTENCEPIECE,
            character_coverage=0.9995)
        self.language = language

    def _identifierPartial(self) -> str:
        return shash(repr(self.preprocessor)) + "_" + shash(f"V={self._size}_l={self._max_token_length}_c={self._character_coverage}_lang={self.language}")