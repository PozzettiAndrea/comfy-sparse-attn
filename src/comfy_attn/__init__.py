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

    # For variable-length / sparse sequences:
    from comfy_attn import dispatch_attention_varlen
    out = dispatch_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q, max_kv)
"""

from .dispatch import dispatch_attention, set_backend, get_backend, get_backend_label
from .detect import auto_detect_precision, auto_select
from .varlen import dispatch_attention_varlen, get_varlen_backend

__all__ = [
    "dispatch_attention",
    "dispatch_attention_varlen",
    "set_backend",
    "get_backend",
    "get_backend_label",
    "get_varlen_backend",
    "auto_detect_precision",
    "auto_select",
]
