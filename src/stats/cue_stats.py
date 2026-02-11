from pathlib import Path
from src.utils.unicode import get_language_map, get_inverse_language_map
from src.preprocessors.cue_preprocessor import CueMapping


def _get_safe_cues_map(lang_code, safe_mapper):
    """
    Helper to get a map of Safe Latin -> Unicode Cue for a given language.
    """
    lang_map = get_language_map()
    c_map = lang_map.get(lang_code, {})
    safe_cues = {}
    for ascii_char, unicode_cue in c_map.items():
        safe_latin = safe_mapper.convert(unicode_cue)
        safe_cues[safe_latin] = unicode_cue
    return safe_cues


def _write_token_distribution(lang_name, tokens, decoding_map, file_handle):
    """
    Helper to write distribution of tokens by length.
    Includes both the actual survived tokens (Safe Latin) and their readable forms (Decued).
    """
    file_handle.write(f"### Distribution of {lang_name} Cued Tokens by Length\n")
    by_len = {}
    for t in tokens:
        l = len(t)
        if l not in by_len: by_len[l] = []
        by_len[l].append(t)

    for l in sorted(by_len.keys()):
        ts = sorted(by_len[l])

        # Create readable versions
        readable_ts = []
        for t in ts:
            r = ""
            for char in t:
                r += decoding_map.get(char, char)
            readable_ts.append(r)

        file_handle.write(f"Length {l}: {len(ts)} tokens\n")
        file_handle.write(f"  Survived (Safe Latin): {ts}\n")
        file_handle.write(f"  Readable (Decued):     {readable_ts}\n")
    file_handle.write("\n")


def analyze_cue_survival(vocab, l2_lang, file_handle):
    """
    Analyzes which language cues survived in the vocabulary and their distribution across token lengths.
    """
    # Instantiate mapper to get the actual characters used in the vocab
    safe_mapper = CueMapping()

    l2_safe_map = _get_safe_cues_map(l2_lang, safe_mapper)
    en_safe_map = _get_safe_cues_map("en", safe_mapper)

    # Build Decoding Maps for Readability (Safe Latin -> ASCII)
    # We need: Safe Latin -> Unicode Cue -> ASCII
    inv_lang_map = get_inverse_language_map()

    l2_decoding_map = {}
    for safe_char, unicode_cue in l2_safe_map.items():
        # Unicode Cue -> ASCII
        ascii_char = inv_lang_map.get(l2_lang, {}).get(unicode_cue, unicode_cue)
        l2_decoding_map[safe_char] = ascii_char

    en_decoding_map = {}
    for safe_char, unicode_cue in en_safe_map.items():
        ascii_char = inv_lang_map.get("en", {}).get(unicode_cue, unicode_cue)
        en_decoding_map[safe_char] = ascii_char

    # Track survival
    l2_surviving = set()
    en_surviving = set()

    l2_tokens = []
    en_tokens = []

    for token in vocab:
        has_l2 = False
        has_en = False

        for char in token:
            if char in l2_safe_map:
                has_l2 = True
                l2_surviving.add(l2_safe_map[char])
            if char in en_safe_map:
                has_en = True
                en_surviving.add(en_safe_map[char])

        if has_l2:
            l2_tokens.append(token)
        if has_en:
            en_tokens.append(token)

    # 1. Comparison Table
    file_handle.write(f"### Cue Survival Comparison (English vs {l2_lang})\n")
    header = f"{'Metric':<20} | {'English':<10} | {l2_lang:<10}"
    file_handle.write(header + "\n")
    file_handle.write("-" * len(header) + "\n")

    # Totals
    total_en = len(en_safe_map)
    total_l2 = len(l2_safe_map)
    file_handle.write(f"{'Total Cues':<20} | {total_en:<10} | {total_l2:<10}\n")

    # Survived
    surv_en = len(en_surviving)
    surv_l2 = len(l2_surviving)
    file_handle.write(f"{'Survived':<20} | {surv_en:<10} | {surv_l2:<10}\n")

    # Missing Count
    miss_en = total_en - surv_en
    miss_l2 = total_l2 - surv_l2
    file_handle.write(f"{'Missing':<20} | {miss_en:<10} | {miss_l2:<10}\n\n")

    # 2. Detailed Missing Lists
    file_handle.write(f"### Missing Cues Details\n")

    en_missing = sorted(list(set(en_safe_map.values()) - en_surviving))
    l2_missing = sorted(list(set(l2_safe_map.values()) - l2_surviving))

    file_handle.write(f"English Missing: {en_missing}\n")
    file_handle.write(f"{l2_lang} Missing: {l2_missing}\n\n")

    # 3. Distributions
    _write_token_distribution("English", en_tokens, en_decoding_map, file_handle)
    _write_token_distribution(l2_lang, l2_tokens, l2_decoding_map, file_handle)


def document_cue_mappings(l2_lang, file_handle):
    """
    Writes the mapping for each language cue: ASCII -> Unicode Cue -> Safe Latin Mapping.
    Includes both English and L2 for comparison.
    """
    file_handle.write(f"### Cue Mappings (English vs {l2_lang})\n")

    # Header
    # ASCII | EN Unicode Cue | EN Safe Latin || L2 Unicode Cue | L2 Safe Latin
    header = f"{'ASCII':<5} | {'EN Unicode Cue':<16} | {'EN Safe Latin':<14} || {f'{l2_lang} Unicode Cue':<16} | {f'{l2_lang} Safe Latin':<14}"
    file_handle.write(header + "\n")
    file_handle.write("-" * len(header) + "\n")

    lang_map = get_language_map()
    l2_cue_map = lang_map.get(l2_lang, {})
    en_cue_map = lang_map.get("en", {})

    # Instantiate CueMapping to get the safe chars
    safe_mapper = CueMapping()

    # Sort by ascii char (common keys)
    # Assumes both languages map 'a'-'z'
    sorted_ascii = sorted(l2_cue_map.keys())

    for char in sorted_ascii:
        # English Data
        en_uni = en_cue_map.get(char, "N/A")
        en_safe = safe_mapper.convert(en_uni) if en_uni != "N/A" else "N/A"

        # L2 Data
        l2_uni = l2_cue_map.get(char, "N/A")
        l2_safe = safe_mapper.convert(l2_uni) if l2_uni != "N/A" else "N/A"

        # Formatting
        # Only show hex if it's a single char to avoid crashing on N/A
        en_uni_fmt = f"{en_uni} ({ord(en_uni):04X})" if len(en_uni) == 1 else en_uni
        en_safe_fmt = f"{en_safe} ({ord(en_safe):04X})" if len(en_safe) == 1 else en_safe

        l2_uni_fmt = f"{l2_uni} ({ord(l2_uni):04X})" if len(l2_uni) == 1 else l2_uni
        l2_safe_fmt = f"{l2_safe} ({ord(l2_safe):04X})" if len(l2_safe) == 1 else l2_safe

        # Use simple string interpolation for first column to guarantee width
        char_fmt = f"'{char}'"
        row = f"{char_fmt:<5} | {en_uni_fmt:<16} | {en_safe_fmt:<14} || {l2_uni_fmt:<16} | {l2_safe_fmt:<14}"
        file_handle.write(row + "\n")
    file_handle.write("\n")


def analyze_false_friend_survival(ff_words, l2_lang, cued_tok, en_tok, l2_tok, file_handle):
    """
    Analyzes how False Friend words are tokenized with and without language cues.
    Compares Cued Tokenizer (on Cued Word) vs Base Tokenizers (on Uncued Word).
    """
    file_handle.write(f"### False Friend Tokenization Analysis\n")

    # Header
    header = f"| {'Word':<15} | {'En Cued Tok':<40} | {'En Base Tok':<40} | {'L2 Cued Tok':<40} | {'L2 Base Tok':<40} |"
    file_handle.write(header + "\n")
    file_handle.write("-" * len(header) + "\n")

    lang_map = get_language_map()
    en_cue_map = lang_map.get("en", {})
    l2_cue_map = lang_map.get(l2_lang, {})

    for word in sorted(list(ff_words)):
        # Construct Cued Words (Unicode Cues)
        # We only cue the first character
        first = word[0]

        en_cued_word = en_cue_map.get(first, first) + word[1:]
        l2_cued_word = l2_cue_map.get(first, first) + word[1:]

        # Tokenize using the respective tokenizers
        # cued_tok handles the mapping from Unicode Cue -> Safe Latin internally
        en_cued_toks = cued_tok.prepareAndTokenise(en_cued_word)
        l2_cued_toks = cued_tok.prepareAndTokenise(l2_cued_word)

        en_base_toks = en_tok.prepareAndTokenise(word)
        l2_base_toks = l2_tok.prepareAndTokenise(word)

        row = f"| {word:<15} | {str(en_cued_toks):<40} | {str(en_base_toks):<40} | {str(l2_cued_toks):<40} | {str(l2_base_toks):<40} |"
        file_handle.write(row + "\n")

    file_handle.write("\n")


def do_cue_stats(trial, vocab_size):
    """
    Main function to run cue-specific statistics.
    Writes results to 'cue_stats.txt' in the trial's stats directory.
    """
    stats_dir = trial.get_stats_directory()
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    stats_path = stats_dir / "cue_stats.txt"

    # Identify Cued Vocabulariser
    # In the trial structure (list of 4 tuples), index 3 corresponds to the Cued Vocabulariser
    vocab_info = trial.get_vocabularisers()

    if len(vocab_info) <= 3:
        print(f"Warning: No Cued Vocabulariser found for {trial.algo} {trial.l2}. Skipping cue stats.")
        return

    # Get Artifact and Vocab
    cued_artifact = vocab_info[3][0]
    cued_vocab = cued_artifact.getVocabulary()  # List[str]

    # Get False Friends
    ff_words = trial.get_ff()

    # Get Tokenizers
    tokenizers = trial.get_tokenizers()
    if len(tokenizers) <= 3:
        print(f"Warning: Missing tokenizers for {trial.algo} {trial.l2}.")
        return

    en_tok = tokenizers[0]
    l2_tok = tokenizers[1]
    cued_tok = tokenizers[3]

    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Cue Stats for Algo: {trial.algo}, Language: {trial.l2}\n")
        f.write("=" * 50 + "\n\n")

        # 1. Survival & Distribution
        analyze_cue_survival(cued_vocab, trial.l2, f)

        f.write("=" * 50 + "\n\n")

        # 2. False Friend Analysis
        analyze_false_friend_survival(ff_words, trial.l2, cued_tok, en_tok, l2_tok, f)

        f.write("=" * 50 + "\n\n")

        # 3. Mappings
        document_cue_mappings(trial.l2, f)

