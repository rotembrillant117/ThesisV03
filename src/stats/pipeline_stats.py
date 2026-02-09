from src.utils.training_data_utils import get_crosslingual_homographs
from src.utils.unicode import get_language_map


def get_pipeline_inspection(text_input, preprocessor, tokenizer):
    """
    Generates a printable string showing the 5 stages of the tokenizer pipeline for a given input.
    """
    output = []
    output.append(f"Pipeline for: '{text_input}'")

    # Stage 1: Raw
    output.append(f"  1. Raw: '{text_input}'")

    # Stage 2: Normalization (Preprocessing)
    preprocessed = preprocessor.reversible.convert(preprocessor.irreversible.convert(text_input))
    output.append(f"  2. Preprocessed: '{preprocessed}'")

    # Stage 3: Splitting & Boundaries
    pretokens = preprocessor.do(text_input)
    output.append(f"  3. Pretokens: {pretokens}")

    # Stage 4: Segmentation
    segments = []
    for pt in pretokens:
        # Wrappers return list[str]
        segments.extend(tokenizer.tokenise(pt))
    output.append(f"  4. Final Segments: {segments}")

    output.append("-" * 20)
    return "\n".join(output)


def run_cued_pipeline_inspection(tokenizers):
    """
    Runs pipeline inspection for cued tokenizers (index 3) across all languages and algorithms.
    Generates output files with the results.
    """
    cues_map_all = get_language_map()

    for language in tokenizers:
        homographs = get_crosslingual_homographs("en", language)
        cur_cues = cues_map_all.get(language, {})

        for algo in tokenizers[language]:
            # Index 3 is the Cued tokenizer (en_L2_cued)
            tok_list = tokenizers[language][algo]

            cued_tok = tok_list[3]
            preprocessor = cued_tok.preprocessor

            file_name = f"{algo}_{language}_pipeline_results.txt"

            with open(file_name, "w", encoding="utf-8") as f:
                f.write(f"Pipeline Inspection for {algo} en_{language} (Cued)\n")
                f.write(f"Total Homographs: {len(homographs)}\n")
                f.write("=" * 50 + "\n\n")

                for word in sorted(homographs):
                    # Plain Word
                    f.write(f"Word: {word}\n")
                    f.write(get_pipeline_inspection(word, preprocessor, cued_tok) + "\n")

                    # Cued Word
                    if word:
                        first_char = word[0]
                        cued_char = cur_cues.get(first_char, first_char)
                        cued_word = cued_char + word[1:]

                        f.write(f"Cued Word: {cued_word}\n")
                        f.write(get_pipeline_inspection(cued_word, preprocessor, cued_tok) + "\n")
                        f.write("=" * 50 + "\n")
