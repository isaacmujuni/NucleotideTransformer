#!/usr/bin/env python3
"""Sanity-check Python version, memory, and optional NTv3 download."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            f"Python 3.10+ is required (NTv3 needs transformers>=4.55). Found {sys.version}"
        )
    print(f"Python {sys.version.split()[0]}")

    import torch

    print(f"torch {torch.__version__}")
    print(f"cuda={torch.cuda.is_available()}  mps={torch.backends.mps.is_available()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", default="", help="HF repo to prefetch, e.g. InstaDeepAI/NTv3_8M_pre")
    args = parser.parse_args()
    if not args.download:
        return

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"Downloading {args.download} ...")
    tok = AutoTokenizer.from_pretrained(args.download, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.download, trust_remote_code=True)
    batch = tok(
        ["ACGTACGT"],
        add_special_tokens=False,
        padding=True,
        pad_to_multiple_of=128,
        return_tensors="pt",
    )
    out = model(**batch)
    print("tokenizer + forward pass ok", tuple(out.logits.shape))


if __name__ == "__main__":
    main()
