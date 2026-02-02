from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm

from ntv3_crop.config import TrainConfig, load_config
from ntv3_crop.dna import pad_or_trim, sanitize_dna
from ntv3_crop.model import NTv3GeneClassifier, load_backbone, load_tokenizer


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def score_sequences(
    model: NTv3GeneClassifier,
    tokenizer,
    records: list[tuple[str, str]],
    labels: list[str],
    seq_len: int,
    batch_size: int = 4,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    model.eval()
    rows = []
    for start in tqdm(range(0, len(records), batch_size), desc="scoring"):
        chunk = records[start : start + batch_size]
        seqs = [pad_or_trim(sanitize_dna(seq), seq_len) for _, seq in chunk]
        encoded = tokenizer(
            seqs,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=seq_len,
            pad_to_multiple_of=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded)["logits"]
        probs = torch.sigmoid(logits).cpu().numpy()
        for (gene_id, _), prob in zip(chunk, probs):
            row = {"gene_id": gene_id, "max_score": float(prob.max())}
            row.update({label: float(p) for label, p in zip(labels, prob)})
            rows.append(row)
    return pd.DataFrame(rows).sort_values("max_score", ascending=False)


def load_trained(
    checkpoint: Path, cfg: TrainConfig
) -> tuple[NTv3GeneClassifier, object, list[str]]:
    label_file = checkpoint / "label_names.json"
    if not label_file.exists():
        label_file = checkpoint.parent / "label_names.json"
    labels = json.loads(label_file.read_text())
    tokenizer_src = checkpoint if (checkpoint / "tokenizer_config.json").exists() else cfg.model_id
    tokenizer = load_tokenizer(str(tokenizer_src))
    backbone, model_cfg = load_backbone(cfg.model_id, freeze=True)
    model = NTv3GeneClassifier(backbone, model_cfg, num_labels=len(labels))
    state_path = checkpoint / "classifier.pt"
    if not state_path.exists():
        state_path = (
            checkpoint / "model.safetensors"
            if (checkpoint / "model.safetensors").exists()
            else checkpoint / "pytorch_model.bin"
        )
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(state_path)
    else:
        state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(_device())
    return model, tokenizer, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score candidate crop genes for stress/disease resistance"
    )
    parser.add_argument("--config", default="configs/mac_8m.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fasta", required=True, help="FASTA of candidate gene windows")
    parser.add_argument("--out", default="outputs/discoveries.tsv")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer, labels = load_trained(Path(args.checkpoint), cfg)
    records = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(args.fasta, "fasta")]
    if not records:
        raise SystemExit(f"No FASTA records in {args.fasta}")
    table = score_sequences(model, tokenizer, records, labels, cfg.seq_len)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, sep="\t", index=False)
    print(table.head(args.top_k).to_string(index=False))
    print(f"\nWrote {len(table)} scored genes to {out}")


if __name__ == "__main__":
    main()
