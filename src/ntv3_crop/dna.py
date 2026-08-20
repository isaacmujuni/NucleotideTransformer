from __future__ import annotations

import random
import re

DNA_RE = re.compile(r"[^ACGTN]")
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

MOTIFS: dict[str, str] = {
    "drought": "CACGTG",  # ABRE-like
    "salt": "ACGTGTC",  # DRE/CRT-like
    "heat": "GAANNTTC",  # HSE-like (N kept as A for simplicity below)
    "cold": "CCGAC",  # CRT/DRE core
    "flooding": "GCAGC",  # anaerobic response-like
    "pathogen": "TTGACC",  # W-box
    "insect": "GGTCA",  # JA-responsive-like
}


def sanitize_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return DNA_RE.sub("N", seq)


def reverse_complement(seq: str) -> str:
    return sanitize_dna(seq).translate(COMPLEMENT)[::-1]


def pad_or_trim(seq: str, length: int, rng: random.Random | None = None) -> str:
    """Center-crop or right-pad with N so length is exact."""
    seq = sanitize_dna(seq)
    if len(seq) == length:
        return seq
    if len(seq) > length:
        start = (len(seq) - length) // 2
        return seq[start : start + length]
    pad = length - len(seq)
    left = pad // 2
    right = pad - left
    return ("N" * left) + seq + ("N" * right)


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def plant_motif(seq: str, motif: str, rng: random.Random, copies: int = 3) -> str:
    motif = motif.replace("N", "A")
    chars = list(seq)
    for _ in range(copies):
        if len(motif) >= len(chars):
            break
        pos = rng.randint(0, len(chars) - len(motif))
        chars[pos : pos + len(motif)] = list(motif)
    return "".join(chars)
