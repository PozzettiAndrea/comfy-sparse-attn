"""comfy_sparse_attn — sparse data types, ops, and attention dispatch for ComfyUI custom nodes.

This package IS the canonical implementation of comfy/sparse.py, comfy/ops_sparse.py,
and comfy/attention_sparse.py, staged ahead of their upstream ComfyUI merge.

For dense attention use ComfyUI's native optimized_attention_for_device directly.

Usage — varlen attention:
    from comfy_sparse_attn import dispatch_varlen_attention

    # q, k, v: [T, H, D]  (total tokens, packed across batch)
    # cu_seqlens: [B+1] int32 cumulative sequence lengths
    out = dispatch_varlen_attention(
        q, k, v,
        cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv,
    )

Namespace injection (call from your node's __init__.py before any model imports):
    import pathlib
    import comfy_sparse_attn
    from comfy_sparse_attn import setup_link
    _PKG = pathlib.Path(comfy_sparse_attn.__file__).parent
    setup_link(_PKG / "sparse.py",           "sparse.py")
    setup_link(_PKG / "ops_sparse.py",       "ops_sparse.py")
    setup_link(_PKG / "attention_sparse.py", "attention_sparse.py")
"""

from .varlen import dispatch_varlen_attention, get_varlen_backend
from ._links import setup_link
from .detect import get_conv_backend, set_conv_backend
from .sparse import (
    VarLenTensor, SparseTensor,
    varlen_cat, varlen_unbind,
    sparse_cat, sparse_unbind,
    get_debug, set_debug,
)

__all__ = [
    # varlen attention
    "dispatch_varlen_attention",
    "get_varlen_backend",
    # namespace injection
    "setup_link",
    # conv backend
    "get_conv_backend",
    "set_conv_backend",
    # sparse data types
    "VarLenTensor",
    "SparseTensor",
    "varlen_cat",
    "varlen_unbind",
    "sparse_cat",
    "sparse_unbind",
    "get_debug",
    "set_debug",
]
