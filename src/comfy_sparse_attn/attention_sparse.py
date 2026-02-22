"""
comfy/attention_sparse.py — attention dispatch for sparse/varlen tensors.

Dense attention: wraps ComfyUI's optimized_attention_for_device with layout
conversion (N,L,H,C <-> B,H,N,D) for models that use the [N,L,H,C] convention.

Varlen attention: delegated to comfy_sparse_attn.dispatch_varlen_attention.
Priority: sage2 > flash > xformers > sdpa.

Usage:
    # Dense attention ([N,L,H,C] layout):
    from comfy.attention_sparse import scaled_dot_product_attention
    out = scaled_dot_product_attention(q, k, v)  # [N, L, H, C] tensors

    # Sparse varlen attention (VarLenTensor):
    from comfy.attention_sparse import sparse_scaled_dot_product_attention
    out = sparse_scaled_dot_product_attention(q, k, v)  # VarLenTensor / dense mixes
"""

import logging

import torch

from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy_sparse_attn import dispatch_varlen_attention  # noqa: F401 — re-exported

log = logging.getLogger("comfy_sparse_attn")

_dense_printed = False

__all__ = [
    'scaled_dot_product_attention',
    'sparse_scaled_dot_product_attention',
    'dispatch_varlen_attention',
]


# ---------------------------------------------------------------------------
# Dense attention dispatch (TRELLIS2 layout)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(*args, **kwargs):
    """
    Scaled dot product attention for TRELLIS2 dense tensors.

    Supports 1, 2, or 3 argument forms:
        scaled_dot_product_attention(qkv)       # qkv: [N, L, 3, H, C]
        scaled_dot_product_attention(q, kv)      # q: [N, L, H, C], kv: [N, L, 2, H, C]
        scaled_dot_product_attention(q, k, v)    # each: [N, L, H, C]

    Returns: [N, L, H, C] tensor.
    """
    transformer_options = kwargs.pop('transformer_options', {})

    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        q, k, v = qkv.unbind(dim=2)

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        k, v = kv.unbind(dim=2)

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"

    # TRELLIS2 [N, L, H, C] -> ComfyUI [N, H, L, C]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    heads = q.shape[1]
    attn_fn = optimized_attention_for_device(q.device)
    global _dense_printed
    if not _dense_printed:
        import sys
        print(f"[comfy.attention_sparse] Dense attention: {attn_fn.__name__}", file=sys.stderr)
        _dense_printed = True
    out = attn_fn(q, k, v, heads=heads, skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options)

    # ComfyUI [N, H, L, C] -> TRELLIS2 [N, L, H, C]
    return out.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Sparse attention dispatch (VarLenTensor)
# ---------------------------------------------------------------------------

def sparse_scaled_dot_product_attention(*args, **kwargs):
    """
    Scaled dot product attention for sparse/variable-length tensors.

    Supports combinations of VarLenTensor and dense torch.Tensor inputs:
        sparse_scaled_dot_product_attention(qkv)       # qkv: VarLenTensor [N, *, 3, H, C]
        sparse_scaled_dot_product_attention(q, kv)      # mixed VarLenTensor/Tensor
        sparse_scaled_dot_product_attention(q, k, v)    # mixed VarLenTensor/Tensor

    Returns VarLenTensor if q is VarLenTensor, else dense [N, L, H, C] tensor.
    """
    from .sparse import VarLenTensor

    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv_feats = qkv.feats       # [T, 3, H, C]
        q, k, v = qkv_feats.unbind(dim=1)  # each [T, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv_feats = kv.feats
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv_feats = kv.reshape(N * L, 2, H, C)
        k, v = kv_feats.unbind(dim=1)

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, VarLenTensor) and isinstance(k, (VarLenTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, VarLenTensor) and isinstance(v, VarLenTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats
            v = v.feats
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)
            v = v.reshape(N * L, H, CO)

    # Build cumulative sequence lengths
    cu_seqlens_q = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(torch.tensor(q_seqlen, device=device), dim=0)
    ]).int()
    cu_seqlens_kv = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(torch.tensor(kv_seqlen, device=device), dim=0)
    ]).int()

    out = dispatch_varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen),
    )

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
