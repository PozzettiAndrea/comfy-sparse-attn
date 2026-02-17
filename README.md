# comfy-attn

Attention dispatch for ComfyUI custom nodes — GPU-aware auto-detection with ComfyUI-native backends.

Replaces per-node attention dispatch code with a shared package that:
- Auto-detects the fastest attention backend for your GPU and installed packages
- Routes through ComfyUI's native attention implementations (no duplicated backend code)
- Caches dimension incompatibilities to avoid repeated failures

## Quick Start

```bash
pip install comfy-attn
```

## Usage

```python
from comfy_attn import set_backend, dispatch_attention

# In your model loader — pick a backend
label = set_backend("auto")  # GPU-aware auto-detection (recommended)
print(f"Attention: {label}")

# Or let users choose via a node dropdown
label = set_backend("sage")       # force SageAttention
label = set_backend("flash_attn") # force FlashAttention
label = set_backend("sdpa")       # force PyTorch SDPA

# In your model's attention forward pass
# q, k, v shape: (B, H, N, D)
out = dispatch_attention(q, k, v)
out = dispatch_attention(q, k, v, attn_mask=mask)  # mask forces SDPA
```

## Auto-Detection Priority

`set_backend("auto")` picks the fastest available backend based on GPU architecture:

| GPU Generation | Priority |
|---|---|
| Blackwell (SM 10.x) | sage3 > flash_fp8 > sage2 > flash > sdpa |
| Hopper (SM 9.0) | flash_fp8 > sage2 > flash > sdpa |
| Ada / Ampere (SM 8.x) | sage2 > flash > sdpa |
| Older / CPU | sdpa |

Backends are only selected if the required package is installed. Install attention packages via [comfy-env](https://github.com/PozzettiAndrea/comfy-env) CUDA config or pip.

## How It Works

`comfy-attn` does **not** implement attention kernels. It routes to ComfyUI's built-in attention functions (`attention_sage`, `attention_flash`, `attention_pytorch`, etc.) which handle dtype casting, tensor layout, and error fallbacks.

The package provides:
1. **GPU-tiered auto-detection** — checks compute capability + installed packages
2. **Thin dispatch layer** — maps `(B, H, N, D)` tensors to ComfyUI functions with `skip_reshape=True`
3. **Dimension tracking** — caches head dimensions that fail for a backend, skips retries

## License

MIT
