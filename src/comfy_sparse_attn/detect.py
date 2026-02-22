"""GPU detection helpers for comfy_attn."""

import logging
from typing import Optional

log = logging.getLogger("comfy_attn")

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
