import unicodedata as ud
import string

def is_stable(ch):
    # Survive common normalizations
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        if ud.normalize(form, ch) != ch:
            return False
    return True

def pick_26_from_ranges(ranges):
    picked = []

    for start, end in ranges:
        for cp in range(start, end + 1):
            ch = chr(cp)

            # Must be a lowercase letter
            if ud.category(ch) != "Ll":
                continue
            # Must survive Unicode normalization
            stable = True
            for form in ("NFC", "NFD", "NFKC", "NFKD"):
                if ud.normalize(form, ch) != ch:
                    stable = False
                    break
            if not stable:
                continue
            picked.append(ch)
            if len(picked) == 26:
                return picked

    raise ValueError(f"Not enough stable lowercase letters (found {len(picked)})")

# Suggested blocks (chosen because they contain many true letters and tend to be NFKC-stable)
BLOCKS = {
    # Cyrillic lowercase lives across multiple chunks; this range covers many lowercase Cyrillic letters.
    "en":  [(0xAB70, 0xABBF)],

    # Armenian
    "de":   [(0x0561, 0x0587)],

    # Georgian
    "es":  [(0x10D0, 0x10FF)],

    # Greek + Greek Extended
    "it":  [(0x03B1, 0x03FF), (0x1F00, 0x1FFF)],

    # Coptic
    "se":  [(0x2C81, 0x2CE3)],

    # Glagolitic has lowercase letters in this range
    "ro": [(0x2C30, 0x2C5F)],

    # Deseret
    "fr":   [(0x10428, 0x1044F)],
}

def build_language_maps():
    out = {}
    for lang, ranges in BLOCKS.items():
        alphabet = pick_26_from_ranges(ranges)
        out[lang] = {a: cue for a, cue in zip(string.ascii_lowercase, alphabet)}
    return out

def get_language_map():
    return build_language_maps()

def get_inverse_language_map():

    l_map = get_language_map()
    inv_map = {}

    for lang, mapping in l_map.items():
        inv = {}
        for ascii_letter, cue_char in mapping.items():
            inv[cue_char] = ascii_letter
        inv_map[lang] = inv

    return inv_map

# LANG_CUE = build_language_maps()

# for a in "abcdefghijklmnopqrstuvwxyz":
#     print(a, "->", LANG_CUE["English"][a])
