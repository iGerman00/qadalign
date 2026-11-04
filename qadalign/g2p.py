"""Grapheme to phoneme: CMUdict, then morphology, then espeak-ng, then rules.

Output uses the 39-phone inventory the model expects.

1. compounds split on hyphen, slash, ampersand or plus, part by part;
2. digits read as they are sung, with any letters around them resolved;
3. CMUdict, exact, then with surrounding quotes stripped, then the part before
   an internal apostrophe;
4. contractions, suffixes and prefixes, then two-word compounds, each resolved
   against the dictionary;
5. espeak-ng, when installed;
6. a letter-to-sound pass.

`app.js` runs the same chain minus step 5, which the browser does not have. A
word no step can pronounce (non-Latin script) has no phones, and the lyric line
keeps it so its neighbours can time it.
"""
from __future__ import annotations

import gzip
import re
import shutil
import subprocess
from pathlib import Path

from .phones import PHONE2ID

_DATA = Path(__file__).parent / "data" / "cmudict.txt.gz"


def _load_bundled():
    """CMUdict data shipped with the package, BSD-2-Clause. See CMUDICT-LICENSE."""
    if not _DATA.exists():
        return None
    out = {}
    with gzip.open(_DATA, "rt", encoding="utf-8") as f:
        for line in f:
            word, _, phones = line.partition("\t")
            phones = phones.split()
            if word and phones:
                out[word] = phones
    return out


def _load_fallback():
    """Only used when the bundled data is missing; the PyPI package is GPL-3."""
    try:
        import cmudict
    except ImportError:
        return None
    return {w: [re.sub(r"\d", "", p) for p in prons[0]] for w, prons in cmudict.dict().items()}


_D = _load_bundled() or _load_fallback() or {}

# longest first so digraphs and diphthongs win
_IPA2ARP = [
    ("tʃ", "CH"), ("dʒ", "JH"), ("aɪ", "AY"), ("aʊ", "AW"), ("eɪ", "EY"),
    ("oʊ", "OW"), ("ɔɪ", "OY"), ("əʊ", "OW"),
    ("iː", "IY"), ("uː", "UW"), ("ɔː", "AO"), ("ɑː", "AA"), ("ɜː", "ER"), ("ɛː", "EH"),
    ("p", "P"), ("b", "B"), ("t", "T"), ("d", "D"), ("k", "K"), ("ɡ", "G"), ("g", "G"),
    ("f", "F"), ("v", "V"), ("θ", "TH"), ("ð", "DH"), ("s", "S"), ("z", "Z"),
    ("ʃ", "SH"), ("ʒ", "ZH"), ("h", "HH"), ("m", "M"), ("n", "N"), ("ŋ", "NG"),
    ("l", "L"), ("ɹ", "R"), ("r", "R"), ("w", "W"), ("j", "Y"), ("ɾ", "T"), ("ʔ", "T"),
    ("i", "IY"), ("ɪ", "IH"), ("e", "EY"), ("ɛ", "EH"), ("æ", "AE"), ("a", "AE"),
    ("ɑ", "AA"), ("ɒ", "AA"), ("ɔ", "AO"), ("o", "OW"), ("ʊ", "UH"), ("u", "UW"),
    ("ʌ", "AH"), ("ə", "AH"), ("ɚ", "ER"), ("ɝ", "ER"), ("ɜ", "ER"),
]
_STRIP = set("ˈˌːˑ˥˦˧˨˩ ‿|‖")   # stress, length, tone, linking marks
_WORD_CLEAN = re.compile(r"[^a-z'’]")
_NONLEXICAL = re.compile(r"^[^a-z]+$")

_ESPEAK = shutil.which("espeak-ng")

# phones after which a plural or past-tense suffix changes shape
_SIBILANT = {"S", "Z", "SH", "ZH", "CH", "JH"}
_VOICELESS = {"T", "K", "P", "S", "SH", "CH", "F", "TH"}
_VOICED = {"B", "D", "G", "V", "DH", "Z", "L", "M", "N", "NG", "R", "W", "Y",
           "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
           "OW", "OY", "UH", "UW"}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
         "ninety"]
_ORD = {
    "zero": "zeroth", "one": "first", "two": "second", "three": "third",
    "four": "fourth", "five": "fifth", "six": "sixth", "seven": "seventh",
    "eight": "eighth", "nine": "ninth", "ten": "tenth", "eleven": "eleventh",
    "twelve": "twelfth", "thirteen": "thirteenth", "fourteen": "fourteenth",
    "fifteen": "fifteenth", "sixteen": "sixteenth", "seventeen": "seventeenth",
    "eighteen": "eighteenth", "nineteen": "nineteenth", "twenty": "twentieth",
    "thirty": "thirtieth", "forty": "fortieth", "fifty": "fiftieth",
    "sixty": "sixtieth", "seventy": "seventieth", "eighty": "eightieth",
    "ninety": "ninetieth", "hundred": "hundredth", "thousand": "thousandth",
}

# Letter-to-sound rules, longest grapheme first, consulted at each position.
# This is pretty draft and the rules are not at all guaranteed to work, but better than skipping words.
_LTS = [
    ("ough", ["AO"]), ("augh", ["AO"]), ("eigh", ["EY"]), ("tion", ["SH", "AH", "N"]),
    ("sion", ["ZH", "AH", "N"]), ("ight", ["AY", "T"]), ("cial", ["SH", "AH", "L"]),
    ("tial", ["SH", "AH", "L"]), ("ture", ["CH", "ER"]), ("sure", ["ZH", "ER"]),
    ("kn", ["N"]), ("wr", ["R"]),
    ("igh", ["AY"]), ("air", ["EH", "R"]), ("eer", ["IH", "R"]), ("ear", ["IH", "R"]),
    ("oor", ["AO", "R"]), ("our", ["AW", "ER"]), ("ore", ["AO", "R"]),
    ("are", ["EH", "R"]), ("ere", ["IH", "R"]), ("ire", ["AY", "R"]),
    ("ure", ["Y", "UH", "R"]), ("dge", ["JH"]), ("tch", ["CH"]), ("sch", ["S", "K"]),
    ("ous", ["AH", "S"]), ("ble", ["B", "AH", "L"]), ("tle", ["T", "AH", "L"]),
    ("cle", ["K", "AH", "L"]), ("gle", ["G", "AH", "L"]), ("ple", ["P", "AH", "L"]),
    ("dle", ["D", "AH", "L"]),
    ("th", ["TH"]), ("sh", ["SH"]), ("ch", ["CH"]), ("ph", ["F"]), ("wh", ["W"]),
    ("ck", ["K"]), ("ng", ["NG"]), ("qu", ["K", "W"]), ("gh", []),
    ("ee", ["IY"]), ("ea", ["IY"]), ("ei", ["EY"]), ("ey", ["EY"]), ("ie", ["IY"]),
    ("ai", ["EY"]), ("ay", ["EY"]), ("oa", ["OW"]), ("oe", ["OW"]), ("oi", ["OY"]),
    ("oy", ["OY"]), ("oo", ["UW"]), ("ou", ["AW"]), ("ow", ["OW"]), ("au", ["AO"]),
    ("aw", ["AO"]), ("ue", ["UW"]), ("ui", ["UW"]), ("ew", ["UW"]), ("eu", ["UW"]),
    ("ar", ["AA", "R"]), ("or", ["AO", "R"]), ("er", ["ER"]), ("ir", ["ER"]),
    ("ur", ["ER"]),
    ("ss", ["S"]), ("ll", ["L"]), ("tt", ["T"]), ("pp", ["P"]), ("mm", ["M"]),
    ("nn", ["N"]), ("dd", ["D"]), ("ff", ["F"]), ("rr", ["R"]), ("bb", ["B"]),
    ("gg", ["G"]), ("cc", ["K"]), ("zz", ["Z"]),
    ("a", ["AE"]), ("e", ["EH"]), ("i", ["IH"]), ("o", ["AA"]), ("u", ["AH"]),
    ("y", ["IY"]),
    ("b", ["B"]), ("c", ["K"]), ("d", ["D"]), ("f", ["F"]), ("g", ["G"]),
    ("h", ["HH"]), ("j", ["JH"]), ("k", ["K"]), ("l", ["L"]), ("m", ["M"]),
    ("n", ["N"]), ("p", ["P"]), ("q", ["K"]), ("r", ["R"]), ("s", ["S"]),
    ("t", ["T"]), ("v", ["V"]), ("w", ["W"]), ("x", ["K", "S"]), ("z", ["Z"]),
]
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
           "OW", "OY", "UH", "UW"}
_C_SOFT = ("e", "i", "y")
_MAGIC = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}

_ADD = [
    ("n't", ["AH", "N", "T"]), ("'ll", ["L"]), ("'re", ["R"]), ("'ve", ["V"]),
    ("'m", ["M"]), ("'s", "plural"), ("'d", "past"), ("'", []),
    ("ings", ["IH", "NG", "Z"]), ("less", ["L", "AH", "S"]), ("ness", ["N", "AH", "S"]),
    ("ment", ["M", "AH", "N", "T"]), ("able", ["AH", "B", "AH", "L"]),
    ("ible", ["AH", "B", "AH", "L"]), ("est", ["IH", "S", "T"]),
    ("ing", ["IH", "NG"]), ("in'", ["IH", "NG"]), ("ed", "past"), ("es", "plural"),
    ("er", ["ER"]), ("ly", ["L", "IY"]), ("s", "plural"), ("in", ["IH", "NG"]),
]
_PREFIX = ["un", "re", "dis", "mis", "pre", "non", "over", "under", "out", "sub",
           "super", "semi", "anti", "multi", "micro", "post", "co"]


def _clean(word: str) -> str:
    return _WORD_CLEAN.sub("", word.lower().replace("’", "'"))


def _ipa_to_arpa(ipa: str) -> list[str]:
    ipa = "".join(c for c in ipa if c not in _STRIP)
    out, i = [], 0
    while i < len(ipa):
        for src, dst in _IPA2ARP:
            if ipa.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            i += 1
    return out


def _espeak(word: str) -> list[str]:
    """Optional: only used for words the dictionary and morphology miss."""
    if not _ESPEAK:
        return []
    try:
        r = subprocess.run([_ESPEAK, "-v", "en-us", "-q", "--ipa", word],
                           capture_output=True, text=True, timeout=5)
        return _ipa_to_arpa(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return []


def _dict_lookup(w: str) -> list[str] | None:
    entry = _D.get(w)
    if not entry:
        return None
    ph = [p for p in (re.sub(r"\d", "", p) for p in entry) if p in PHONE2ID]
    return ph or None


def _plural(ph: list[str]) -> list[str]:
    last = ph[-1] if ph else "S"
    if last in _SIBILANT:
        return ph + ["IH", "Z"]
    return ph + (["Z"] if last in _VOICED else ["S"])


def _past(ph: list[str]) -> list[str]:
    last = ph[-1] if ph else "T"
    if last in ("T", "D"):
        return ph + ["IH", "D"]
    return ph + (["T"] if last in _VOICELESS else ["D"])


def _apply(base: list[str], add) -> list[str]:
    if add == "plural":
        return _plural(base)
    if add == "past":
        return _past(base)
    return base + list(add or [])


def _stem_forms(stem: str) -> list[str]:
    """Spellings a stem may take before a suffix: hop(e), run(n), plan(n)."""
    out = [stem + "e", stem]
    if len(stem) > 2:
        if stem[-1] == stem[-2]:
            out.append(stem[:-1])          # runnin -> run
        out.append(stem[:-1])
    return out


def _number_words(digits: str) -> list[str]:
    """Digits as sung: 7 -> seven, 808 -> eight oh eight, 1999 -> nineteen ninety nine."""
    if not digits:
        return []
    n = len(digits)
    if n <= 2:
        v = int(digits)
        if v < 20:
            return [_ONES[v]]
        t, u = divmod(v, 10)
        return [_TENS[t]] if u == 0 else [_TENS[t], _ONES[u]]
    if n == 4 and digits[2:] == "00":
        return [_ONES[int(digits[0])], "thousand"]
    if n == 4 and digits[:2] in ("19", "20"):
        first = _number_words(digits[:2])
        rest = digits[2:]
        if rest[0] == "0":
            return first + ["oh", _ONES[int(rest[1])]]
        return first + _number_words(rest)
    return ["oh" if c == "0" else _ONES[int(c)] for c in digits]


def _ordinal(words: list[str]) -> list[str]:
    if not words or words[-1] not in _ORD:
        return words
    return words[:-1] + [_ORD[words[-1]]]


def _lts(w: str) -> list[str]:
    """Letter to sound: greedy longest match, then a vowel is guaranteed."""
    out, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]
        if ch == "'":
            i += 1
            continue
        # a final e is silent, and it lengthens the vowel two letters back
        if ch == "e" and i == n - 1 and n > 2:
            i += 1
            continue
        if ch in _MAGIC and i + 2 == n - 1 and w[i + 2] == "e" and w[i + 1] not in "hrw":
            out.append(_MAGIC[ch])
            i += 1
            continue
        for src, ph in _LTS:
            if not w.startswith(src, i):
                continue
            emit = ph
            if src == "c" and i + 1 < n and w[i + 1] in _C_SOFT:
                emit = ["S"]
            elif src == "g" and i + 1 < n and w[i + 1] in _C_SOFT:
                emit = ["JH"]
            out += emit
            i += len(src)
            break
        else:
            i += 1
    if out and not any(p in _VOWELS for p in out):
        out.append("AH")
    return out


def _resolve(w: str, depth: int = 0) -> list[str] | None:
    """Dictionary, then a suffix or prefix, each resolved recursively."""
    ph = _dict_lookup(w)
    if ph:
        return ph
    if depth >= 3 or len(w) < 4:
        return None
    for suf, add in _ADD:
        if not w.endswith(suf) or len(w) - len(suf) < 2:
            continue
        for cand in _stem_forms(w[: len(w) - len(suf)]):
            base = _resolve(cand, depth + 1)
            if base:
                return _apply(base, add)
    for pre in _PREFIX:
        rest = w[len(pre):]
        if not w.startswith(pre) or len(rest) < 3:
            continue
        base = _resolve(rest, depth + 1)
        if base:
            return base
    return None


def _glued(w: str) -> list[str] | None:
    """Two dictionary words written as one: lovestruck, raindrop, homesick."""
    if len(w) < 6:
        return None
    for cut in range(len(w) - 3, 2, -1):
        a, b = _dict_lookup(w[:cut]), _dict_lookup(w[cut:])
        if a and b:
            return a + b
    return None


def _digit_phones(tok: str) -> list[str] | None:
    """Phones for a token whose pronunciation comes from digits."""
    digits = re.sub(r"[^0-9A-Za-z]", "", tok)
    if digits.isdigit():
        words = _number_words(digits)
    else:
        m = re.fullmatch(r"(\d+)(st|nd|rd|th)", digits.lower())
        if not m:
            return None
        words = _ordinal(_number_words(m.group(1)))
    out = []
    for word in words:
        ph = _dict_lookup(word) or _lts(word)
        if not ph:
            return None
        out += ph
    return out


def phones_for(word: str, oov: dict | None = None) -> list[str] | None:
    """ARPAbet phones for one word, or None if nothing usable."""
    raw = word.strip()
    # hyphenated, slashed and ampersand compounds
    if re.search(r"[-/&+]", raw):
        parts = [p for p in re.split(r"[-/&+]", raw) if re.search(r"[A-Za-z0-9]", p)]
        if len(parts) > 1:
            out = []
            for part in parts:
                sub = phones_for(part, oov)
                if not sub:
                    out = None
                    break
                out += sub
            if out:
                return out
    if re.search(r"\d", raw):
        ph = _digit_phones(raw)
        if ph:
            return ph
        # letters and digits mixed
        if re.search(r"[A-Za-z]", raw):
            out = []
            for part in re.findall(r"\d+|[A-Za-z']+", raw):
                sub = _digit_phones(part) if part[0].isdigit() else phones_for(part, oov)
                if not sub:
                    return None
                out += sub
            return out or None
    w = _clean(raw)
    if not w or _NONLEXICAL.match(w):
        return None
    if oov is not None and w in oov:
        return oov[w] or None
    # exact first, so leading-apostrophe entries ('em, 'til) still match; then
    # without surrounding quotes, for words written as 'good'; then the part
    # before an internal apostrophe, for rock'n'roll
    bare = w.strip("'")
    cands = [w] + ([bare] if bare != w else [])
    if "'" in bare:
        cands.append(bare.split("'")[0])
    for cand in cands:
        ph = _dict_lookup(cand)
        if ph:
            return ph
    ph = _resolve(w) or (_resolve(w.replace("'", "")) if "'" in w else None) or _glued(w)
    if not ph:
        ph = [p for p in _espeak(w) if p in PHONE2ID]
    ph = ph or _lts(w)
    if oov is not None:
        oov[w] = ph
    return ph or None


_VOWEL_LETTERS = set("aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿаеёиоуыэюяіїє"
                     "αεηιουωάέήίόύώ")

# Cyrillic read letter by letter. Not Russian phonology, but the consonants
# delimit words for the CTC path and the vowels give it length.
_CYRILLIC = {
    "а": ["AA"], "б": ["B"], "в": ["V"], "г": ["G"], "д": ["D"], "е": ["EH"],
    "ё": ["OW"], "ж": ["ZH"], "з": ["Z"], "и": ["IY"], "й": ["Y"], "к": ["K"],
    "л": ["L"], "м": ["M"], "н": ["N"], "о": ["AA"], "п": ["P"], "р": ["R"],
    "с": ["S"], "т": ["T"], "у": ["UW"], "ф": ["F"], "х": ["HH"], "ц": ["T", "S"],
    "ч": ["CH"], "ш": ["SH"], "щ": ["SH"], "ъ": [], "ы": ["IH"], "ь": [],
    "э": ["EH"], "ю": ["UW"], "я": ["AA"],
    "і": ["IY"], "ї": ["IY"], "є": ["EH"], "ґ": ["G"],
}
_CYR_IOTATED = set("еёюяєї")
_CYR_VOWELS = set("аеёиоуыэюяіїє")


def _cyrillic(w: str) -> list[str] | None:
    """Phones for a Cyrillic word, or None if it holds none."""
    if not any(c in _CYRILLIC for c in w):
        return None
    out = []
    for i, ch in enumerate(w):
        ph = _CYRILLIC.get(ch)
        if ph is None:
            continue                         # ь, ъ and anything else silent
        if ch in _CYR_IOTATED and (i == 0 or w[i - 1] in _CYR_VOWELS
                                   or w[i - 1] in "ьъй"):
            out.append("Y")
        out += ph
    if out and not any(p in _VOWELS for p in out):
        out.append("AH")
    return out or None


def skeleton(word: str) -> list[str]:
    """Phones for a word no dictionary or letter-to-sound step can reach.

    Cyrillic is read letter by letter. Any other script gets one placeholder
    vowel per syllable, which is enough for the CTC path to stretch over the
    sung syllables and place the word; a word with no time at all would be
    pushed into its neighbour's silence.
    """
    w = word.lower()
    ph = _cyrillic(w)
    if ph:
        return ph
    letters = [c for c in w if c.isalpha()]
    k = sum(1 for c in letters if c in _VOWEL_LETTERS)
    if not k:
        k = (len(letters) + 2) // 3          # no vowel letters we know: 3 letters a syllable
    return ["AH"] * max(1, min(k, 6))


def tokenize(line: str) -> list[str]:
    """Words of one lyric line.

    `[Section]` tags are structure and are dropped; parenthesised text is kept.
    """
    line = re.sub(r"\[[^\]]*\]", " ", line)
    line = re.sub(r"[()]", " ", line)
    line = re.sub(r"[♪♫🎵]", " ", line)
    toks = []
    for t in re.split(r"\s+", line.strip()):
        if not t or not re.search(r"[A-Za-z0-9]", t):
            continue
        if re.fullmatch(r"\*+", t):
            continue
        toks.append(t)
    return toks
