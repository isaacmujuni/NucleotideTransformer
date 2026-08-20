from __future__ import annotations


def recommend_config() -> str:
    """Return a YAML config path that fits the current accelerator."""
    try:
        import torch
    except ImportError:
        return "configs/mac_8m.yaml"

    if not torch.cuda.is_available():
        return "configs/mac_8m.yaml"
    name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if "a100" in name or "h100" in name or vram_gb >= 36:
        return "configs/colab_a100.yaml"
    return "configs/colab_t4.yaml"
