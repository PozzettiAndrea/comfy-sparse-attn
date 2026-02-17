"""Attention dispatch for ComfyUI custom nodes.

GPU-aware auto-detection with ComfyUI-native backends.

Usage:
    from comfy_attn import set_backend, dispatch_attention

    label = set_backend("auto")  # picks fastest for your GPU
    # label = set_backend("sage")  # force sage attention
    # label = set_backend("flash_attn")  # force flash attention
    # label = set_backend("sdpa")  # force PyTorch SDPA

    # In your model's attention forward:
    out = dispatch_attention(q, k, v)  # q/k/v shape: (B, H, N, D)
"""

from .dispatch import dispatch_attention, set_backend, get_backend, get_backend_label
from .detect import auto_detect_precision, auto_select

__all__ = [
    "dispatch_attention",
    "set_backend",
    "get_backend",
    "get_backend_label",
    "auto_detect_precision",
    "auto_select",
]
