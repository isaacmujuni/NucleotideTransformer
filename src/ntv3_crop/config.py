from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model_id: str = "InstaDeepAI/NTv3_8M_pre"
    seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 8
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    freeze_backbone: bool = True
    fp16: bool = False
    bf16: bool = False
    num_workers: int = 0
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    train_csv: str = "data/sample/train.csv"
    val_csv: str = "data/sample/val.csv"
    output_dir: str = "outputs/run"
    seed: int = 42
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    max_steps: int = -1
    promoter_bp: int = 2000
    terminator_bp: int = 500

    def __post_init__(self) -> None:
        if self.seq_len % 128 != 0:
            raise ValueError(
                f"NTv3 requires seq_len to be a multiple of 128, got {self.seq_len}"
            )


DEFAULT_LABELS = (
    "drought",
    "salt",
    "heat",
    "cold",
    "flooding",
    "pathogen",
    "insect",
)

# GO terms used to harvest labeled genes from Ensembl Plants / BioMart.
GO_TERMS: dict[str, tuple[str, ...]] = {
    "drought": ("GO:0009414",),  # response to water deprivation
    "salt": ("GO:0009651",),  # response to salt stress
    "heat": ("GO:0009408",),  # response to heat
    "cold": ("GO:0009409", "GO:0050826"),  # cold / freezing
    "flooding": ("GO:0009413",),  # response to flooding
    "pathogen": (
        "GO:0006952",  # defense response
        "GO:0009626",  # plant-type hypersensitive response
        "GO:0009617",  # response to bacterium
        "GO:0009615",  # response to virus
        "GO:0050832",  # defense response to fungus
    ),
    "insect": ("GO:0009625",),  # response to insect
}

# BioMart dataset names on plants.ensembl.org (may need updating per Ensembl release).
ENSEMBL_DATASETS: dict[str, str] = {
    "arabidopsis_thaliana": "athaliana_eg_gene",
    "oryza_sativa": "osativa_eg_gene",
    "zea_mays": "zmays_eg_gene",
    "glycine_max": "gmax_eg_gene",
    "solanum_lycopersicum": "slycopersicum_eg_gene",
    "triticum_aestivum": "taestivum_eg_gene",
    "hordeum_vulgare": "hvulgare_eg_gene",
    "sorghum_bicolor": "sbicolor_eg_gene",
    "vitis_vinifera": "vvinifera_eg_gene",
}


def load_config(path: str | Path) -> TrainConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    known = {k: raw[k] for k in TrainConfig.__dataclass_fields__ if k in raw}
    return TrainConfig(**known)
