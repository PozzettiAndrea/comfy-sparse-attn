"""Attention dispatch — routes to ComfyUI's native attention functions.

Usage:
    from comfy_attn import set_backend, dispatch_attention

    label = set_backend("auto")  # or "sdpa", "flash_attn", "sage"
    print(f"Using: {label}")

    # In your model's attention forward:
    out = dispatch_attention(q, k, v)  # q/k/v shape: (B, H, N, D)
"""

import logging
import sys
from typing import Callable, Optional

from . import detect

log = logging.getLogger("comfy_attn")

# Active attention function (ComfyUI's native implementation)
_attn_fn: Optional[Callable] = None
_sdpa_fn: Optional[Callable] = None

# Current backend state
_backend_name: str = "sdpa"
_backend_label: str = "sdpa (PyTorch native)"

# {backend_name: set(head_dims)} — skip known failures
_unsupported_dims: dict[str, set[int]] = {}

# Human-readable labels for logging
_LABELS = {
    "sdpa": "sdpa (PyTorch native)",
    "flash": "FlashAttention 2 (fp16/bf16)",
    "flash_fp8": "FlashAttention 3/4 (FP8)",
    "sage2": "SageAttention v2 (INT8)",
    "sage3": "SageAttention v3 (Blackwell FP4)",
}


def _get_comfy_attention_fn(name: str) -> Optional[Callable]:
    """Look up a ComfyUI attention function by registry name.

    Args:
        name: ComfyUI registry name ("pytorch", "sage", "sage3", "flash").

    Returns:
        The attention function, or None if not found.
    """
    try:
        from comfy.ldm.modules.attention import get_attention_function
        return get_attention_function(name, default=None)
    except ImportError:
        return None


def _get_optimized_attention() -> Callable:
    """Get ComfyUI's default optimized attention (CLI-flag driven)."""
    try:
        from comfy.ldm.modules.attention import optimized_attention
        return optimized_attention
    except ImportError:
        import torch
        return torch.nn.functional.scaled_dot_product_attention


def _ensure_sdpa_fn() -> Callable:
    """Get the SDPA/pytorch attention function (used as fallback)."""
    global _sdpa_fn
    if _sdpa_fn is None:
        fn = _get_comfy_attention_fn("pytorch")
        _sdpa_fn = fn if fn is not None else _get_optimized_attention()
    return _sdpa_fn


def _resolve_backend(internal_name: str) -> tuple[Callable, str]:
    """Resolve an internal backend name to a ComfyUI function + label.

    Returns:
        (attention_function, human_readable_label)
    """
    entry = detect._BACKEND_REGISTRY.get(internal_name)
    if entry is None:
        fn = _ensure_sdpa_fn()
        return fn, _LABELS["sdpa"]

    comfy_name = entry[0]
    fn = _get_comfy_attention_fn(comfy_name)
    if fn is not None:
        return fn, _LABELS.get(internal_name, internal_name)

    # Registered function not found — fall back to optimized_attention
    log.warning(f"Backend '{internal_name}' not available in ComfyUI, falling back to default")
    fn = _get_optimized_attention()
    return fn, f"default (wanted {internal_name})"


def set_backend(name: str) -> str:
    """Set the active attention backend.

    Args:
        name: One of "auto", "sdpa", "flash_attn", "sage".

    Returns:
        Human-readable label describing the resolved backend.
    """
    global _attn_fn, _backend_name, _backend_label

    if name == "auto":
        resolved = detect.auto_select()
        _attn_fn, _backend_label = _resolve_backend(resolved)
        _backend_name = resolved
        return _backend_label

    if name == "sdpa":
        _attn_fn = _ensure_sdpa_fn()
        _backend_name = "sdpa"
        _backend_label = _LABELS["sdpa"]
        return _backend_label

    if name == "flash_attn":
        major, _ = detect._get_gpu_arch()
        # Try FP8 (FA3/FA4) on Hopper+
        if major >= 9 and detect._is_available("flash_fp8"):
            _attn_fn, _backend_label = _resolve_backend("flash_fp8")
            _backend_name = "flash_fp8"
            return _backend_label
        # Fall back to FA2
        if detect._is_available("flash"):
            _attn_fn, _backend_label = _resolve_backend("flash")
            _backend_name = "flash"
            return _backend_label
        log.warning("flash-attn not available, falling back to sdpa")
        _attn_fn = _ensure_sdpa_fn()
        _backend_name = "sdpa"
        _backend_label = _LABELS["sdpa"]
        return _backend_label

    if name == "sage":
        major, _ = detect._get_gpu_arch()
        # Try sage3 on Blackwell
        if major >= 10 and detect._is_available("sage3"):
            _attn_fn, _backend_label = _resolve_backend("sage3")
            _backend_name = "sage3"
            return _backend_label
        # Fall back to sage2
        if detect._is_available("sage2"):
            _attn_fn, _backend_label = _resolve_backend("sage2")
            _backend_name = "sage2"
            return _backend_label
        log.warning("sageattention not available, falling back to sdpa")
        _attn_fn = _ensure_sdpa_fn()
        _backend_name = "sdpa"
        _backend_label = _LABELS["sdpa"]
        return _backend_label

    log.warning(f"Unknown backend '{name}', falling back to sdpa")
    _attn_fn = _ensure_sdpa_fn()
    _backend_name = "sdpa"
    _backend_label = _LABELS["sdpa"]
    return _backend_label


def get_backend() -> str:
    """Return the current internal backend name."""
    return _backend_name


def get_backend_label() -> str:
    """Return the current human-readable backend label."""
    return _backend_label


def dispatch_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p: float = 0.0,
):
    """Route attention to the active ComfyUI backend.

    Args:
        q, k, v: Tensors of shape (B, H, N, D).
        attn_mask: Optional attention mask. Forces SDPA when present.
        dropout_p: Dropout probability (accepted for API compat, not used by ComfyUI backends).

    Returns:
        Output tensor of shape (B, H, N, D).
    """
    global _attn_fn
    if _attn_fn is None:
        label = set_backend("auto")
        print(f"\033[33m[comfy-attn] Dense attention: {label}\033[0m", file=sys.stderr, flush=True)

    heads = q.shape[1]
    head_dim = q.shape[-1]

    # Dimension tracking: skip backends known to fail on this head_dim
    if head_dim in _unsupported_dims.get(_backend_name, set()):
        return _ensure_sdpa_fn()(
            q, k, v, heads, mask=attn_mask, skip_reshape=True, skip_output_reshape=True,
        )

    # Mask forces SDPA (sage/flash don't support arbitrary masks well)
    if attn_mask is not None:
        return _ensure_sdpa_fn()(
            q, k, v, heads, mask=attn_mask, skip_reshape=True, skip_output_reshape=True,
        )

    try:
        return _attn_fn(
            q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True,
        )
    except Exception as e:
        # Cache the failure and fall back
        _unsupported_dims.setdefault(_backend_name, set()).add(head_dim)
        log.warning(
            f"{_backend_name} failed for head_dim={head_dim}: {e}, "
            f"using sdpa for these layers"
        )
        return _ensure_sdpa_fn()(
            q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True,
        )
