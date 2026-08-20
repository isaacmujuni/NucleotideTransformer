# Fine-tune NTv3 for crop stress and disease-resistance gene discovery

Nucleotide Transformer v3 ([NTv3](https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-v3)) is InstaDeep’s long-context DNA foundation model. This repo fine-tunes it as a **multi-label gene classifier**, then ranks unlabeled crop genes by the probability they are involved in abiotic stress or disease resistance.

## Why this task shape

“Discovering resistant genes” is not BigWig track prediction. The useful output is a **ranked list of genes** with scores for drought, salt, heat, cold, flooding, pathogen, and insect. That maps cleanly onto:

1. A gene-centric DNA window (promoter + gene + terminator)
2. A mean-pooled NTv3 embedding
3. A multi-label head trained on known resistance genes vs matched negatives
4. A genome-wide scoring pass over candidate loci

Use the **pre-trained** checkpoints (`*_pre`), not the post-trained track models. Post-trained NTv3 is built for functional-track / annotation heads. The pretrained U-Net backbone is the right starting point for a new gene-level task.

| Checkpoint | Params | Use |
|---|---|---|
| `InstaDeepAI/NTv3_8M_pre` | 8M | Local smoke tests on this Mac |
| `InstaDeepAI/NTv3_100M_pre` | 100M | First real GPU run |
| `InstaDeepAI/NTv3_650M_pre` | 650M | Best accuracy (A100/H100) |

This machine is an **Apple M2 with 8 GB RAM**. The 650M model will not train here. Prototype on 8M, then move `configs/gpu_100m.yaml` / `configs/gpu_650m.yaml` to a Colab or cloud GPU.

NTv3 input length must be a **multiple of 128**. Load every checkpoint with `trust_remote_code=True`. `transformers>=4.55` and **Python 3.10+** are required (system Python on this Mac is 3.9).

## Pipeline

```
known stress / R-genes  -->  promoter+gene windows  -->  LoRA / head fine-tune
        Ensembl Plants GO + PRGdb                         NTv3 backbone
                                                              |
                                                              v
unlabeled crop genes / GFF  -->  score  -->  ranked TSV of candidate resistance genes
```

Positive labels come from Gene Ontology via Ensembl Plants BioMart:

| Label | GO evidence |
|---|---|
| drought | GO:0009414 water deprivation |
| salt | GO:0009651 salt stress |
| heat | GO:0009408 heat |
| cold | GO:0009409 / GO:0050826 |
| flooding | GO:0009413 |
| pathogen | defense / bacterium / virus / fungus |
| insect | GO:0009625 |

Negatives are genes from the same species that lack those terms. Sequence windows are genomic DNA with 2 kb upstream and 0.5 kb downstream of the gene.

Optional later upgrades: PRGdb 4.0 curated R-genes, stress RNA-seq (treat as extra labels or as a track head), and QTL / GWAS hits as weak positives.

## Google Colab (recommended)

This Mac cannot train NTv3-100M/650M. Use the Colab notebook:

1. Open [`notebooks/NTv3_crop_Colab.ipynb`](notebooks/NTv3_crop_Colab.ipynb) in [Google Colab](https://colab.research.google.com/).
2. `Runtime → Change runtime type → T4 GPU` (A100 on Colab Pro for 650M).
3. Zip this folder and upload it in the notebook (or put it on Drive):

```bash
cd /Users/mac
zip -r NucleotideTransformer.zip NucleotideTransformer \
  -x "NucleotideTransformer/.venv/*" \
  -x "NucleotideTransformer/outputs/*"
```

4. Run all cells. First pass uses synthetic **sample** data so you can confirm the loop. Then set `DATA_MODE` to `ensembl` for real crop genes.

T4 uses `configs/colab_t4.yaml` (`NTv3_100M_pre` + LoRA + fp16). A100 uses `configs/colab_a100.yaml` (`NTv3_650M_pre` + LoRA + bf16).

## Local setup

```bash
# Python 3.10+ (Homebrew: brew install python@3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Hugging Face may require a token for some NTv3 files:

```bash
huggingface-cli login
```

## Run locally (synthetic data, 8M model)

```bash
python scripts/prepare_sample.py
python scripts/check_env.py --download InstaDeepAI/NTv3_8M_pre
python scripts/train.py --config configs/mac_8m.yaml
python scripts/discover.py \
  --config configs/mac_8m.yaml \
  --checkpoint outputs/mac_8m/best \
  --fasta data/sample/candidates.fa \
  --out outputs/discoveries.tsv
```

The sample set plants known cis-motifs (ABRE, W-box, CRT/DRE) into random DNA so the loop can overfit. It is **not** biological signal.

## Real crop data (needs network + GPU for training)

```bash
python scripts/prepare_ensembl.py \
  --species arabidopsis_thaliana oryza_sativa zea_mays \
  --out-dir data/processed \
  --max-genes 800
```

Then train on a GPU:

```bash
python scripts/train.py --config configs/gpu_100m.yaml
python scripts/discover.py \
  --config configs/gpu_100m.yaml \
  --checkpoint outputs/gpu_100m/best \
  --fasta path/to/candidate_genes.fa \
  --out outputs/rice_candidates.tsv
```

Candidate FASTA records should be the same window style as training (promoter + gene). Rank by `max_score` or a specific column such as `pathogen`.

## Recommended next steps

1. Confirm target crops (rice, maize, wheat, soybean, tomato, …).
2. Install Python 3.11 and run the 8M smoke test on this Mac.
3. Pull Ensembl Plants GO labels for those crops.
4. Fine-tune `NTv3_100M_pre` with LoRA on a 16–24 GB GPU.
5. Hold out one species (e.g. train Arabidopsis+rice, test maize) to measure cross-species discovery.
6. Only then scale to `NTv3_650M_pre` and add PRGdb / RNA-seq.

AgroNT (`InstaDeepAI/agro-nucleotide-transformer-1b`) is a plant-only alternative, but its context is ~6 kb. NTv3’s 1 Mb window is the better long-range regulatory model if a GPU is available.
