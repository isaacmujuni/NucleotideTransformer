from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import Trainer, TrainingArguments, set_seed

from ntv3_crop.config import TrainConfig, load_config
from ntv3_crop.dataset import GeneLabelDataset
from ntv3_crop.model import (
    NTv3GeneClassifier,
    infer_lora_targets,
    load_backbone,
    load_tokenizer,
)


def maybe_apply_lora(model: NTv3GeneClassifier, cfg: TrainConfig) -> NTv3GeneClassifier:
    if not cfg.use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    targets = infer_lora_targets(model.backbone)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=targets,
    )
    model.backbone = get_peft_model(model.backbone, lora)
    return model


def collate(features: list[dict]) -> dict[str, torch.Tensor]:
    ignore = {"gene_id"}
    keys = [k for k in features[0] if k not in ignore]
    batch = {k: torch.stack([f[k] for f in features]) for k in keys}
    return batch


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


def compute_metrics(eval_pred, labels: list[str]) -> dict[str, float]:
    logits, y_true = eval_pred
    y_score = 1.0 / (1.0 + np.exp(-logits))
    metrics = {
        "auroc_macro": _safe_auc(y_true, y_score),
        "auprc_macro": float(average_precision_score(y_true, y_score, average="macro")),
    }
    for i, name in enumerate(labels):
        if len(np.unique(y_true[:, i])) < 2:
            metrics[f"auprc_{name}"] = float("nan")
            continue
        metrics[f"auprc_{name}"] = float(
            average_precision_score(y_true[:, i], y_score[:, i])
        )
    return metrics


def _training_arguments(cfg: TrainConfig) -> TrainingArguments:
    kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        seed=cfg.seed,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.num_workers,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="auprc_macro",
        greater_is_better=True,
    )
    if cfg.max_steps and cfg.max_steps > 0:
        kwargs["max_steps"] = cfg.max_steps
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["save_strategy"] = "steps"
    else:
        kwargs["eval_strategy"] = "steps"
        kwargs["save_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg.model_id)
    backbone, model_cfg = load_backbone(cfg.model_id, freeze=cfg.freeze_backbone)
    model = NTv3GeneClassifier(
        backbone,
        model_cfg,
        num_labels=len(cfg.labels),
        freeze_backbone=cfg.freeze_backbone,
    )
    model = maybe_apply_lora(model, cfg)

    train_ds = GeneLabelDataset(cfg.train_csv, tokenizer, cfg.labels, cfg.seq_len)
    val_ds = GeneLabelDataset(cfg.val_csv, tokenizer, cfg.labels, cfg.seq_len)

    trainer = Trainer(
        model=model,
        args=_training_arguments(cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=lambda pred: compute_metrics(pred, cfg.labels),
    )
    trainer.train()
    metrics = trainer.evaluate()
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    best_dir = output / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    torch.save(trainer.model.state_dict(), best_dir / "classifier.pt")
    tokenizer.save_pretrained(str(best_dir))
    labels_json = json.dumps(cfg.labels, indent=2)
    (output / "label_names.json").write_text(labels_json)
    (best_dir / "label_names.json").write_text(labels_json)
    (output / "train_config.json").write_text(
        json.dumps(cfg.__dict__, indent=2, default=str)
    )
    return best_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune NTv3 on crop resistance genes")
    parser.add_argument("--config", default="configs/mac_8m.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    best = train(cfg)
    print(f"Saved best checkpoint to {best}")


if __name__ == "__main__":
    main()
