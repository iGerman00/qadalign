"""Phone inventory: 39 ARPAbet phones, stress stripped, blank at 0.

The CTC head emits `NUM_PHONES` classes: index 0 is the CTC blank, the phones
occupy 1..39. This must match the shipped checkpoint exactly.
"""

PHONES = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
          "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
          "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]

PHONE2ID = {p: i + 1 for i, p in enumerate(PHONES)}
BLANK = 0
NUM_PHONES = len(PHONES) + 1
