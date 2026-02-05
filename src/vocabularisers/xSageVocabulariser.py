from tktkt.models.sage.vocabularisation import SageVocabulariser
from tktkt.models.sage.schedules import DoubleLinearSchedule, ExponentialDilation
from tktkt.util.strings import shash


class xSageVocabulariser(SageVocabulariser):
    def __init__(self, initial_artifacts, target_vocab_size, language, initial_vocab_builder):
        self.vocab_schedule = DoubleLinearSchedule(start=target_vocab_size*8,
                                                   mid=target_vocab_size*2,
                                                   end=target_vocab_size,
                                                   t_mid=0.5)
        self.language = language
        self.initial_vocab_builder = initial_vocab_builder
        super().__init__(initial_artifacts, vocabulary_schedule=self.vocab_schedule)

    def _identifierPartial(self) -> str:
        return (shash(repr(self.preprocessor)) + "_" + shash(repr(self.vocabulary_points) + repr(self.recompute_embeddings_at))
                + "_" + shash(f"lang={self.language}_{self.initial_vocab_builder}"))