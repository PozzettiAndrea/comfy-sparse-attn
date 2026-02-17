"""Variable-length attention dispatch for sparse/packed sequences.

Routes to the best available varlen backend:
- sageattn_varlen (SageAttention INT8, Ampere+)
- flash_attn_varlen_func (FlashAttention 2)
- xFormers memory_efficient_attention with BlockDiagonalMask
- PyTorch SDPA with block-diagonal additive mask (fallback)

Usage:
    from comfy_attn import dispatch_attention_varlen

    # q, k, v: [T, H, D] (total tokens across batch)
    # cu_seqlens_q/kv: [B+1] cumulative sequence lengths
    out = dispatch_attention_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv,
    )
"""

import logging

import torch

from . import detect

log = logging.getLogger("comfy_attn")

_varlen_fn = None
_varlen_backend: str = "sdpa"


def _resolve_varlen_backend() -> tuple:
    """Pick the best varlen backend. Returns (fn, name).

    Priority: sage2 > flash > xformers > sdpa
    (sage varlen uses Triton INT8 — faster than flash on Ampere/Ada)
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
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    return xops.memory_efficient_attention(q, k, v, mask)[0]


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                 max_seqlen_q, max_seqlen_kv, **kwargs):
    """SDPA fallback with block-diagonal additive mask."""
    B = cu_seqlens_q.shape[0] - 1
    T_q, H, D = q.shape
    T_kv = k.shape[0]

    # Build block-diagonal mask: 0 where allowed, -inf where masked
    mask = torch.full((T_q, T_kv), float("-inf"), device=q.device, dtype=q.dtype)
    for i in range(B):
        qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        ks, ke = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        mask[qs:qe, ks:ke] = 0.0

    q = q.unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, T_q, D]
    k = k.unsqueeze(0).permute(0, 2, 1, 3)
    v = v.unsqueeze(0).permute(0, 2, 1, 3)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T_q, T_kv]

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.permute(0, 2, 1, 3).squeeze(0)  # [T_q, H, D]


def get_varlen_backend() -> str:
    """Return the current varlen backend name."""
    return _varlen_backend


def dispatch_attention_varlen(
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
        q: [T_q, H, D] query tensor (packed across batch).
        k: [T_kv, H, D] key tensor.
        v: [T_kv, H, D_v] value tensor.
        cu_seqlens_q: [B+1] cumulative query sequence lengths.
        cu_seqlens_kv: [B+1] cumulative key/value sequence lengths.
        max_seqlen_q: Maximum query sequence length in batch.
        max_seqlen_kv: Maximum key/value sequence length in batch.

    Returns:
        Output tensor of shape [T_q, H, D_v].
    """
    global _varlen_fn, _varlen_backend
    if _varlen_fn is None:
        _varlen_fn, _varlen_backend = _resolve_varlen_backend()
        log.info("Varlen attention: %s", _varlen_backend)

    return _varlen_fn(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
    )
