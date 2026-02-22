"""GPU and sparse conv backend detection for comfy_sparse_attn."""

import logging
import os
from typing import Optional

log = logging.getLogger("comfy_sparse_attn")

# ---------------------------------------------------------------------------
# GPU arch
# ---------------------------------------------------------------------------

_gpu_arch: Optional[tuple[int, int]] = None


def _get_gpu_arch() -> tuple[int, int]:
    """Return (major, minor) compute capability, cached after first call."""
    global _gpu_arch
    if _gpu_arch is not None:
        return _gpu_arch

    import torch

    try:
        import comfy.model_management
        if comfy.model_management.get_torch_device().type == "cuda":
            _gpu_arch = torch.cuda.get_device_capability()
        else:
            _gpu_arch = (0, 0)
    except ImportError:
        if torch.cuda.is_available():
            _gpu_arch = torch.cuda.get_device_capability()
        else:
            _gpu_arch = (0, 0)
    return _gpu_arch


def _can_import(module_name: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Sparse conv backend
# ---------------------------------------------------------------------------

SPCONV_ALGO = 'auto'
FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0

_CONV = None  # Lazy — detected on first use


def _detect_available_conv_backend() -> str:
    """Try to import conv backends in priority order, return first available."""
    env_backend = os.environ.get('SPARSE_CONV_BACKEND')
    if env_backend:
        valid_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
        if env_backend in valid_backends:
            log.info(f"Using conv backend from SPARSE_CONV_BACKEND env var: {env_backend}")
            return env_backend
        else:
            log.warning(f"Invalid SPARSE_CONV_BACKEND '{env_backend}', must be one of {valid_backends}")

    backends = ['flex_gemm', 'spconv', 'torchsparse']
    for backend in backends:
        try:
            if backend == 'flex_gemm':
                import flex_gemm  # noqa: F401
                log.info("Auto-detected conv backend: flex_gemm")
                return backend
            elif backend == 'spconv':
                import spconv  # noqa: F401
                log.info("Auto-detected conv backend: spconv")
                return backend
            elif backend == 'torchsparse':
                import torchsparse  # noqa: F401
                log.info("Auto-detected conv backend: torchsparse")
                return backend
        except ImportError:
            continue
        except Exception as e:
            log.warning(f"{backend} import failed: {e}")
            continue

    log.info("No sparse conv backend available, using none")
    return 'none'


def get_conv_backend() -> str:
    """Get current conv backend, detecting on first call."""
    global _CONV
    if _CONV is None:
        _CONV = _detect_available_conv_backend()
    return _CONV


def set_conv_backend(backend: str) -> None:
    """Set conv backend explicitly."""
    global _CONV
    valid_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
    if backend not in valid_backends:
        raise ValueError(f"Invalid conv backend '{backend}', must be one of {valid_backends}")
    if _CONV is not None and _CONV != backend:
        log.info(f"Changing conv backend from {_CONV} to {backend}")
    _CONV = backend
    log.info(f"Conv backend set to: {backend}")
