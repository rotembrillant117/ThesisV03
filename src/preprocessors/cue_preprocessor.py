from tktkt.interfaces.preprocessors import InvertibleTextMapper, Preprocessor
from src.utils.unicode import get_safe_latin_chars
from tktkt.factories.preprocessors import ModernEnglishPretokeniser, TruncateAndNormalise
from tktkt.preparation.boundaries import BoundaryMarker
from src.utils.unicode import get_language_map
from tktkt.preparation.splitters import (
    PretokeniserSequence, IsolatePunctuation, HyphenMode,
    WhitespacePretokeniser, IsolateEnglishContractions, PolariseApostrophes,
    AddWordBoundary, GroupDigits, IsolateConnectingHyphens, AddCapitalMarker
)

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

        # Get enough safe chars for all unique cues
        safe_chars_pool = get_safe_latin_chars(limit=len(sorted_cues) + 10)

        for i, cue in enumerate(sorted_cues):
            safe_char = safe_chars_pool[i]
            self.forward_map[cue] = safe_char
            self.backward_map[safe_char] = cue

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


class CuePrefab2(Preprocessor):
    """
    Combines the robust splitting logic of Prefab2 with the safe CueMapping.
    This ensures we handle contractions, digits, and capitalization robustly (like Prefab2),
    while also mapping language cues to safe Latin characters (like CuePreprocessor)
    to prevent tokenizer crashes.
    """
    def __init__(self, marker: BoundaryMarker, truncate_text_after_chars: int = 1_000_000):
        super().__init__(
            uninvertible_mapping=TruncateAndNormalise(truncate_text_after_chars),
            invertible_mapping=CueMapping(),  # Use CueMapping instead of RegisterASCII
            splitter=PretokeniserSequence([
                IsolatePunctuation(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
                WhitespacePretokeniser(destructive=True),
                IsolateEnglishContractions(do_nt=True),
                PolariseApostrophes(tiebreak_left=True),
                AddWordBoundary(marker),
                GroupDigits(n=3),
                IsolateConnectingHyphens(),
                AddCapitalMarker(ignore_marker=marker)
            ])
        )
