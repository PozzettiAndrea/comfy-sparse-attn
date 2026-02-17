"""GPU detection and attention backend auto-selection.

Speed tiers by GPU generation:
- Blackwell (SM 10.x): sage3 > flash > sage2 > sdpa
- Hopper   (SM 9.0):   flash_fp8 > sage2 > flash > sdpa
- Ada      (SM 8.9):   sage2 > flash > sdpa
- Ampere   (SM 8.x):   sage2 ~ flash > sdpa
- Older:                sdpa only
"""

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


def _is_backend_registered(name: str) -> bool:
    """Check if a backend is registered in ComfyUI's attention system."""
    try:
        from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS
        return name in REGISTERED_ATTENTION_FUNCTIONS
    except ImportError:
        return False


# Maps internal backend names to (comfyui_function_name, import_module_to_check)
# comfyui_function_name is used with get_attention_function()
# import_module_to_check is the fallback availability check when not in registry
_BACKEND_REGISTRY = {
    "sage3": ("sage3", "sageattn3"),
    "sage2": ("sage", "sageattention"),
    "flash": ("flash", "flash_attn"),
    "flash_fp8": ("flash", "flash_attn_interface"),
    "sdpa": ("pytorch", None),
}


def _is_available(backend: str) -> bool:
    """Check if a backend is available (registered in ComfyUI or importable)."""
    entry = _BACKEND_REGISTRY.get(backend)
    if entry is None:
        return False
    comfy_name, import_module = entry
    if _is_backend_registered(comfy_name):
        return True
    if import_module is not None:
        return _can_import(import_module)
    return True  # sdpa is always available


def auto_select() -> str:
    """Pick the fastest available backend for the current GPU.

    Returns:
        Internal backend name: "sage3", "sage2", "flash", "flash_fp8", or "sdpa".
    """
    major, _ = _get_gpu_arch()

    if major >= 10:  # Blackwell (SM 10.0+)
        candidates = ["sage3", "flash_fp8", "sage2", "flash"]
    elif major == 9:  # Hopper (SM 9.0)
        candidates = ["flash_fp8", "sage2", "flash"]
    elif major >= 8:  # Ampere / Ada (SM 8.0-8.9)
        candidates = ["sage2", "flash"]
    else:
        candidates = []

    for backend in candidates:
        if _is_available(backend):
            return backend

    return "sdpa"


def auto_detect_precision():
    """Return the best inference dtype for the current GPU.

    Returns:
        torch.bfloat16 on Ampere+ (SM 8.0+)
        torch.float16  on Volta/Turing (SM 7.x)
        torch.float32  on older / CPU
    """
    import torch

    major, _ = _get_gpu_arch()
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return torch.float32
