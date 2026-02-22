"""comfy_attn — sparse / variable-length attention dispatch for ComfyUI custom nodes.

For dense attention use ComfyUI's native optimized_attention_for_device directly.
This package handles the varlen/sparse case that ComfyUI doesn't cover natively.

Usage:
    from comfy_attn import dispatch_varlen_attention

    # q, k, v: [T, H, D]  (total tokens, packed across batch)
    # cu_seqlens: [B+1] int32 cumulative sequence lengths
    out = dispatch_varlen_attention(
        q, k, v,
        cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv,
    )
"""

from .varlen import dispatch_varlen_attention, get_varlen_backend

__all__ = [
    "dispatch_varlen_attention",
    "get_varlen_backend",
]
