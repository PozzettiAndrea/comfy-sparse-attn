# comfy-sparse-attn

Sparse / variable-length attention dispatch for ComfyUI custom nodes, plus a
filesystem-link utility for staging sparse primitives under the `comfy` namespace
ahead of their upstream merge.

## Quick Start

```bash
pip install comfy-sparse-attn
```

## Varlen Attention

Handles the varlen case that ComfyUI's built-in attention doesn't cover: packed
sequences of different lengths across a batch, as produced by sparse 3D models
(point clouds, voxels, VarLenTensors). For ordinary dense attention use
`optimized_attention_for_device` from ComfyUI directly.

```python
from comfy_sparse_attn import dispatch_varlen_attention

# q, k, v: [T, H, D]  — total tokens packed across the batch
# cu_seqlens_q / cu_seqlens_kv: [B+1] int32 cumulative lengths
# max_seqlen_q / max_seqlen_kv: int, longest sequence in the batch
out = dispatch_varlen_attention(
    q, k, v,
    cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_kv,
)
# out: [T, H, D]
```

### Backend Priority

| GPU | Priority |
|---|---|
| SM ≥ 8.0 (Ampere / Ada / Hopper / Blackwell) | sage2 > flash > xformers > sdpa |
| Older / CPU | xformers > sdpa |

## comfy Namespace Injection

Sparse primitives (`comfy.sparse`, `comfy.ops_sparse`, `comfy.attention_sparse`)
are written as if already merged into ComfyUI main. `setup_link` creates symlinks
at those paths at node-load time so the imports work today. When ComfyUI ships the
real files, `setup_link` detects them and becomes a no-op — no code changes needed.

Call it from your node's `__init__.py` **before** any model imports:

```python
import pathlib
from comfy_sparse_attn import setup_link

_HERE = pathlib.Path(__file__).parent / "my_model"
setup_link(_HERE / "sparse.py",           "sparse.py")            # → comfy/sparse.py
setup_link(_HERE / "ops_sparse.py",       "ops_sparse.py")        # → comfy/ops_sparse.py
setup_link(_HERE / "attention_sparse.py", "attention_sparse.py")  # → comfy/attention_sparse.py
```

`setup_link` skips silently if:
- The target is a **real file** (ComfyUI shipped it natively).
- The target is already a symlink pointing to the same source.

## What This Package Is Not

- **Not a dense attention dispatcher.** Use `comfy.ldm.modules.attention.optimized_attention_for_device`.
- **Not a kernel implementation.** Routes to ComfyUI's built-in backends or xformers/SDPA.

## License

MIT
