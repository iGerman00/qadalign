"""Build the pronunciation dictionary: qadalign/data/cmudict.txt.gz.

CMUdict plus phones for forms it lacks but the Python pipeline covers through
its espeak-ng fallback, mostly contractions. Written as `word<TAB>PHONES`, one
per line, gzip compressed.
"""
import gzip
import sys
from pathlib import Path

import cmudict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qadalign.g2p import phones_for

EXTRA = """i'm i've i'll i'd you're you've you'll you'd he's he'll he'd she's she'll she'd
it's it'll we're we've we'll we'd they're they've they'll they'd that's there's what's
who's where's how's isn't aren't wasn't weren't doesn't didn't don't won't wouldn't
can't couldn't shouldn't hasn't haven't hadn't mustn't needn't let's ain't y'all cause
gonna wanna gotta kinda sorta outta til ooh oohs yeah yea uh ah mm mmm hmm oh whoa woah
la na hey ay eh ohh shoulda coulda woulda musta oughta dunno gimme lemme tryna hafta
ya cha em 'em nothin somethin lovin makin""".split()

OUT = Path(__file__).resolve().parent.parent / "qadalign" / "data" / "cmudict.txt.gz"


def main():
    rows = {}
    for w, prons in cmudict.dict().items():
        if w.isalpha():
            rows[w] = [p.rstrip("012") for p in prons[0]]
    added = 0
    for w in EXTRA + [w for w in cmudict.dict() if not w.isalpha()]:
        if w in rows:
            continue
        ph = phones_for(w)
        if ph:
            rows[w] = ph
            added += 1
    # -in' variants: drop the final G from the -ing form, as sung
    for w, ph in list(rows.items()):
        if w.endswith("ing") and len(ph) > 1 and ph[-1] == "NG":
            rows.setdefault(w[:-3] + "in'", ph[:-1] + ["IH", "N"])
            rows.setdefault(w[:-3] + "in", ph[:-1] + ["IH", "N"])
    body = "\n".join(f"{w}\t{' '.join(p)}" for w, p in sorted(rows.items()) if p).encode()
    # mtime=0 keeps rebuilds byte-identical
    with open(OUT, "wb") as f:
        f.write(gzip.compress(body, compresslevel=9, mtime=0))
    print(f"{len(rows)} words ({added} added), {OUT.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
