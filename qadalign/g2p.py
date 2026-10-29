"""Grapheme to phoneme: CMUdict, with espeak-ng as an optional fallback.

Output uses the 39-phone inventory the model expects.
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
    """Optional: only used for words CMUdict lacks. Absent binary is not an error."""
    if not _ESPEAK:
        return []
    try:
        r = subprocess.run([_ESPEAK, "-v", "en-us", "-q", "--ipa", word],
                           capture_output=True, text=True, timeout=5)
        return _ipa_to_arpa(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return []


def phones_for(word: str, oov: dict | None = None) -> list[str] | None:
    """ARPAbet phones for one word, or None if nothing usable."""
    w = _clean(word)
    if not w or _NONLEXICAL.match(w):
        return None
    entry = _D.get(w)
    if entry:
        out = [re.sub(r"\d", "", p) for p in entry]
        return [p for p in out if p in PHONE2ID] or None
    if oov is not None and w in oov:
        return oov[w] or None
    phones = [p for p in _espeak(w) if p in PHONE2ID]
    if oov is not None:
        oov[w] = phones
    return phones or None


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
