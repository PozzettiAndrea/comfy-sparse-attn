# comfy-attn

Sparse / variable-length attention dispatch for ComfyUI custom nodes.

Handles the varlen case that ComfyUI's built-in attention doesn't cover: packed sequences of different lengths across a batch, as produced by sparse 3D models (point clouds, voxels, VarLenTensors). For ordinary dense attention, use `optimized_attention_for_device` from ComfyUI directly.

## Quick Start

```bash
pip install comfy-attn
```

## Usage

```python
from comfy_attn import dispatch_varlen_attention

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

## Backend Priority

Backends are selected at call time based on GPU compute capability and installed packages:

| GPU Generation | Priority |
|---|---|
| Ada / Ampere / Hopper / Blackwell (SM ≥ 8.0) | sage2 > flash > xformers > sdpa |
| Older / CPU | xformers > sdpa |

Only installed backends are tried. The first that succeeds is used; failures fall through to the next.

## What This Package Is Not

- **Not a dense attention dispatcher.** Use `comfy.ldm.modules.attention.optimized_attention_for_device` for `[B, H, N, D]` tensors.
- **Not a kernel implementation.** Routes to ComfyUI's built-in `attention_sage`, `attention_flash`, `attention_pytorch`, etc., or to xformers/SDPA directly for the varlen case.

## License

MIT
