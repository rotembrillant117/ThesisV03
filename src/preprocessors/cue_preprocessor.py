from tktkt.interfaces.preprocessors import InvertibleTextMapper, Preprocessor
from tktkt.preparation.splitters import PretokeniserSequence, MapperAsPretokeniser
from tktkt.factories.preprocessors import ModernEnglishPretokeniser, TruncateAndNormalise
from tktkt.preparation.boundaries import BoundaryMarker
from src.utils.unicode import get_language_map

class CueMapping(InvertibleTextMapper):
    """
    Maps language cues (which might be multi-byte or exotic scripts like Armenian/Georgian)
    to single-character Latin Extensions.

    This ensures that SentencePiece treats them as standard letters and merges them
    correctly with the rest of the word, avoiding the "Missing Parent" crash
    and ensuring we get tokens like "Ʒbraham" (which decodes to "աbraham").
    """

    def __init__(self):
        self.forward_map = {}
        self.backward_map = {}

        all_maps = get_language_map()

        unique_cues = set()
        for lang, mapping in all_maps.items():
            for ascii_char, cue_char in mapping.items():
                unique_cues.add(cue_char)

        # Assign each cue a "Safe Latin Char"
        # We start from U+0180 (Latin Extended-B) which contains distinct
        # but single-char letters like ƀ, Ɓ, Ƃ, etc.

        # Sort for determinism
        sorted_cues = sorted(list(unique_cues))

        current_safe_codepoint = 0x0180

        for cue in sorted_cues:
            safe_char = chr(current_safe_codepoint)
            self.forward_map[cue] = safe_char
            self.backward_map[safe_char] = cue

            current_safe_codepoint += 1

        # Create translation tables
        self.trans_forward = str.maketrans(self.forward_map)
        self.trans_backward = str.maketrans(self.backward_map)

    def convert(self, text: str) -> str:
        return text.translate(self.trans_forward)

    def invert(self, text: str) -> str:
        return text.translate(self.trans_backward)


class CueSplitter(PretokeniserSequence):
    """
    The splitting logic for CuePreprocessor.
    """
    def __init__(self, marker: BoundaryMarker):
        super().__init__([
            # Split + Boundary (using ModernEnglish logic but invalidating pseudobytes)
            # We disable post-boundary splitting to keep "_" attached to the cue char.
            ModernEnglishPretokeniser(marker=marker, do_pseudobytes=False, do_split_after_placing_boundaries=False),
        ])

class CuePreprocessor(Preprocessor):
    """
    A robust preprocessor for SentencePiece that:
    1. Normalises text (NFKC) [Uninvertible Mapping].
    2. Maps Language Cues -> Safe Latin Chars (1-to-1) [Invertible Mapping].
    3. Splits on whitespace and punctuation [Splitter].
    """
    def __init__(self, marker: BoundaryMarker):
        super().__init__(
            uninvertible_mapping=TruncateAndNormalise(truncate_after_chars=1_000_000),
            invertible_mapping=CueMapping(),
            splitter=CueSplitter(marker=marker))
