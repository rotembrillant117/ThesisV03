from tktkt.models.kudopiece.vocabularisation import KudoPieceVocabulariser, KudoPieceArguments
from tktkt.interfaces.preprocessors import Preprocessor
from tktkt.util.strings import shash


class xKudoVocabulariser(KudoPieceVocabulariser):
    """
    A simplified SentencePiece (KudoPiece) Vocabulariser.
    """

    def __init__(self, preprocessor: Preprocessor, vocab_size, language):


        super().__init__(preprocessor=preprocessor, final_vocab_size=vocab_size, arguments=KudoPieceArguments())
        self.language = language

    def _identifierPartial(self) -> str:
        return shash(repr(self.preprocessor)) + "_" + shash(f"V={self._size}_{repr(self._arguments)}_lang={self.language}")