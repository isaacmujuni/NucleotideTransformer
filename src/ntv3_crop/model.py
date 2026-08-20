from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


def resolve_embed_dim(config) -> int:
    for key in ("embed_dim", "hidden_size", "model_dim", "d_model"):
        if hasattr(config, key):
            value = getattr(config, key)
            if isinstance(value, int) and value > 0:
                return value
    raise AttributeError("Could not resolve NTv3 embedding dimension from config")


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_backbone(model_id: str, freeze: bool = False):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "deconv_layers_to_save"):
        config.deconv_layers_to_save = (1,)
    backbone = AutoModelForMaskedLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
    )
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone, config


def infer_lora_targets(model: nn.Module) -> list[str]:
    keywords = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query",
        "key",
        "value",
        "out_proj",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
    )
    found: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if any(key in name.lower() or key == leaf for key in keywords):
            found.add(leaf)
    if not found:
        raise RuntimeError(
            "Could not infer LoRA target modules. Inspect model.named_modules() "
            "and set lora target_modules explicitly."
        )
    return sorted(found)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(x)))


class NTv3GeneClassifier(nn.Module):
    """NTv3 backbone + mean-pooled multi-label head for stress/disease genes."""

    def __init__(
        self,
        backbone,
        config,
        num_labels: int,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = ClassificationHead(resolve_embed_dim(config), num_labels)

    def _hidden(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        if isinstance(outputs, dict) and outputs.get("hidden_states"):
            return outputs["hidden_states"][-1]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        raise RuntimeError("Backbone did not return hidden states")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = self._hidden(outputs)
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.head(pooled)
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
        return result
