#!/usr/bin/env python3
"""Build a tiny synthetic dataset so the training loop can run without Ensembl."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from ntv3_crop.config import DEFAULT_LABELS
from ntv3_crop.dna import MOTIFS, plant_motif, random_dna


def build_frame(n: int, seq_len: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        seq = random_dna(seq_len, rng)
        labels = {name: 0 for name in DEFAULT_LABELS}
        # ~40% negatives, rest get 1-2 planted resistance motifs.
        if rng.random() > 0.4:
            chosen = rng.sample(list(DEFAULT_LABELS), k=rng.choice((1, 2)))
            for name in chosen:
                seq = plant_motif(seq, MOTIFS[name], rng, copies=rng.randint(2, 5))
                labels[name] = 1
        rows.append(
            {
                "gene_id": f"SYNTH_{i:04d}",
                "species": "synthetic",
                "sequence": seq,
                **labels,
            }
        )
    return pd.DataFrame(rows)


def write_candidates(df: pd.DataFrame, path: Path) -> None:
    records = [
        SeqRecord(Seq(row.sequence), id=row.gene_id, description="synthetic candidate")
        for row in df.itertuples(index=False)
    ]
    SeqIO.write(records, path, "fasta")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/sample")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-train", type=int, default=80)
    parser.add_argument("--n-val", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = build_frame(args.n_train, args.seq_len, seed=0)
    val = build_frame(args.n_val, args.seq_len, seed=1)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    write_candidates(val, out / "candidates.fa")
    print(f"Wrote {len(train)} train and {len(val)} val rows to {out}")


if __name__ == "__main__":
    main()
