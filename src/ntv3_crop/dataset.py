from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from ntv3_crop.dna import pad_or_trim, sanitize_dna


class GeneLabelDataset(Dataset):
    """CSV with columns: gene_id, species, sequence, plus one 0/1 column per label."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        labels: list[str],
        seq_len: int,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        missing = [c for c in ("gene_id", "sequence") if c not in self.df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        for label in labels:
            if label not in self.df.columns:
                raise ValueError(f"{csv_path} is missing label column '{label}'")
        self.tokenizer = tokenizer
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        seq = pad_or_trim(sanitize_dna(str(row["sequence"])), self.seq_len)
        encoded = self.tokenizer(
            seq,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            pad_to_multiple_of=128,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [float(row[name]) for name in self.labels], dtype=torch.float32
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = labels
        item["gene_id"] = str(row["gene_id"])
        return item
