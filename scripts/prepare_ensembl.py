#!/usr/bin/env python3
"""Harvest GO-labeled crop genes from Ensembl Plants and fetch promoter+gene windows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ntv3_crop.config import DEFAULT_LABELS, ENSEMBL_DATASETS
from ntv3_crop.dna import sanitize_dna
from ntv3_crop.ensembl import (
    biomart_all_genes,
    fetch_gene_window,
    labeled_gene_table,
)


def add_negatives(positives: pd.DataFrame, species: str, ratio: float, seed: int) -> pd.DataFrame:
    all_genes = biomart_all_genes(species).rename(columns={"ensembl_gene_id": "gene_id"})
    positive_ids = set(positives["gene_id"])
    candidates = all_genes.loc[~all_genes["gene_id"].isin(positive_ids), ["gene_id"]].drop_duplicates()
    n_neg = min(len(candidates), int(len(positives) * ratio))
    neg = candidates.sample(n=n_neg, random_state=seed).copy()
    for label in DEFAULT_LABELS:
        neg[label] = 0
    neg["species"] = species
    return pd.concat([positives, neg], ignore_index=True)


def attach_sequences(df: pd.DataFrame, promoter_bp: int, terminator_bp: int) -> pd.DataFrame:
    seqs = []
    keep = []
    for gene_id in tqdm(df["gene_id"], desc="fetch sequences"):
        try:
            seq = sanitize_dna(fetch_gene_window(gene_id, promoter_bp, terminator_bp))
            seqs.append(seq)
            keep.append(True)
        except Exception as exc:
            print(f"[skip] {gene_id}: {exc}")
            seqs.append("")
            keep.append(False)
        time.sleep(0.12)  # stay under Ensembl REST rate limits
    df = df.copy()
    df["sequence"] = seqs
    return df.loc[keep].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species",
        nargs="+",
        default=["arabidopsis_thaliana"],
        help=f"Ensembl Plants species. Known: {', '.join(ENSEMBL_DATASETS)}",
    )
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--neg-ratio", type=float, default=1.5)
    parser.add_argument("--promoter-bp", type=int, default=2000)
    parser.add_argument("--terminator-bp", type=int, default=500)
    parser.add_argument("--max-genes", type=int, default=400, help="Cap per species for a first run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = []
    for species in args.species:
        if species not in ENSEMBL_DATASETS:
            raise SystemExit(f"Unknown species '{species}'. Add it to ENSEMBL_DATASETS.")
        print(f"==> {species}")
        labeled = labeled_gene_table(species)
        labeled = add_negatives(labeled, species, args.neg_ratio, args.seed)
        if args.max_genes and len(labeled) > args.max_genes:
            labeled = labeled.sample(n=args.max_genes, random_state=args.seed)
        labeled = attach_sequences(labeled, args.promoter_bp, args.terminator_bp)
        frames.append(labeled)

    data = pd.concat(frames, ignore_index=True)
    train, val = train_test_split(
        data,
        test_size=0.15,
        random_state=args.seed,
        stratify=(data[list(DEFAULT_LABELS)].sum(axis=1) > 0).astype(int),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    print(f"Wrote {len(train)} train / {len(val)} val genes to {out}")


if __name__ == "__main__":
    main()
