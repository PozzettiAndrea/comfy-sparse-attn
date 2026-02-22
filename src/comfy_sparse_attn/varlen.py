"""Variable-length / sparse attention dispatch.

Routes to the best available varlen backend:
- sageattn_varlen  (SageAttention INT8, Ampere+ SM >= 8)
- flash_attn_varlen_func  (FlashAttention 2)
- xFormers memory_efficient_attention with BlockDiagonalMask
- PyTorch SDPA with block-diagonal additive mask (always available)

All inputs use packed format: tokens from all batch items concatenated
along the sequence dimension, with cu_seqlens tracking boundaries.

Usage:
    from comfy_attn import dispatch_varlen_attention

    # q, k, v: [T, H, D]  (total tokens across batch, packed)
    # cu_seqlens_q/kv: [B+1] int32 cumulative sequence lengths
    out = dispatch_varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv,
    )
"""

import logging

import torch
import torch.nn.functional as F

from . import detect

log = logging.getLogger("comfy_attn")

_varlen_fn = None
_varlen_backend: str = "sdpa"


def _resolve_varlen_backend() -> tuple:
    """Pick the best varlen backend. Returns (fn, name).

    Priority: sage2 > flash > xformers > sdpa
    """
    major, _ = detect._get_gpu_arch()

    if major >= 8 and detect._can_import("sageattention"):
        try:
            from sageattention import sageattn_varlen
            return sageattn_varlen, "sage2"
        except ImportError:
            pass

    if detect._can_import("flash_attn"):
        try:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func, "flash"
        except ImportError:
            pass

    if detect._can_import("xformers"):
        return _xformers_varlen, "xformers"

    return _sdpa_varlen, "sdpa"


def _xformers_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                     max_seqlen_q, max_seqlen_kv, **kwargs):
    """xFormers varlen via BlockDiagonalMask."""
    import xformers.ops as xops

    B = cu_seqlens_q.shape[0] - 1
    q_seqlen = [(cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item() for i in range(B)]
    kv_seqlen = [(cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]).item() for i in range(B)]
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
    return xops.memory_efficient_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask,
    )[0]


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                 max_seqlen_q, max_seqlen_kv, **kwargs):
    """SDPA fallback with block-diagonal additive mask."""
    B = cu_seqlens_q.shape[0] - 1
    T_q, H, D = q.shape
    T_kv = k.shape[0]

    mask = torch.full((T_q, T_kv), float("-inf"), device=q.device, dtype=q.dtype)
    for i in range(B):
        qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        ks, ke = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        mask[qs:qe, ks:ke] = 0.0

    q = q.unsqueeze(0).permute(0, 2, 1, 3)   # [1, H, T_q, D]
    k = k.unsqueeze(0).permute(0, 2, 1, 3)
    v = v.unsqueeze(0).permute(0, 2, 1, 3)
    mask = mask.unsqueeze(0).unsqueeze(0)     # [1, 1, T_q, T_kv]

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.permute(0, 2, 1, 3).squeeze(0)  # [T_q, H, D]


def get_varlen_backend() -> str:
    """Return the active varlen backend name (resolved on first call)."""
    return _varlen_backend


def dispatch_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> torch.Tensor:
    """Route variable-length attention to the best available backend.

    Args:
        q:             [T_q, H, D]  query tensor, packed across batch.
        k:             [T_kv, H, D] key tensor.
        v:             [T_kv, H, D_v] value tensor.
        cu_seqlens_q:  [B+1] int32 cumulative query sequence lengths.
        cu_seqlens_kv: [B+1] int32 cumulative key/value sequence lengths.
        max_seqlen_q:  Scalar maximum query sequence length.
        max_seqlen_kv: Scalar maximum key/value sequence length.

    Returns:
        [T_q, H, D_v] output tensor.
    """
    global _varlen_fn, _varlen_backend
    if _varlen_fn is None:
        _varlen_fn, _varlen_backend = _resolve_varlen_backend()
        import sys
        print(f"\033[33m[comfy-attn] Varlen attention: {_varlen_backend}\033[0m",
              file=sys.stderr, flush=True)

    return _varlen_fn(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
    )
