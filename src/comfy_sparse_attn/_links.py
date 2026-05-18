"""
Namespace injection: make `from comfy.sparse import ...` (and ops_sparse,
attention_sparse) resolve to the modules shipped by this package, so consumer
code can be written as if the upstream ComfyUI PR were already merged.

Implementation is purely in-memory: `sys.modules['comfy.<name>']` is pointed
at our own module, and the attribute is set on the `comfy` package. No
filesystem operations — earlier versions used `pathlib.Path.symlink_to`,
which fails on Windows containers without `SeCreateSymbolicLinkPrivilege`
(`OSError: [WinError 1314] A required privilege is not held by the client`).

When ComfyUI ships these files natively, `setup_link` detects the real file
under `comfy/` and refuses to override — the native module wins.
"""

import importlib
import logging
import pathlib
import sys

log = logging.getLogger("comfy_sparse_attn")


def setup_link(source: pathlib.Path, comfy_relative: str) -> bool:
    """
    Alias `comfy.<name>` to `comfy_sparse_attn.<name>` via `sys.modules`.

    `name` is derived from `comfy_relative` by stripping a `.py` suffix.
    The `source` argument is accepted for backward compatibility with the
    previous symlink-based API; it is not read.

    Skips (returns False) when:
      - ComfyUI is not importable from this interpreter.
      - A real file exists at `comfy/<comfy_relative>` (ComfyUI shipped
        this module natively — native wins on import).
      - `sys.modules['comfy.<name>']` is already populated (someone else
        aliased first, or the module was already imported).
      - `comfy_sparse_attn.<name>` itself is not importable.

    Returns True iff a fresh alias was installed.
    """
    try:
        import comfy
    except ImportError:
        log.warning("comfy not found on sys.path, skipping sparse alias setup")
        return False

    name = comfy_relative.removesuffix(".py")
    fq = f"comfy.{name}"

    comfy_dir = pathlib.Path(comfy.__path__[0])
    if (comfy_dir / comfy_relative).is_file():
        log.debug("comfy/%s is a native file, skipping alias", comfy_relative)
        return False

    if fq in sys.modules:
        return False

    try:
        mod = importlib.import_module(f"comfy_sparse_attn.{name}")
    except ImportError as e:
        log.warning("comfy_sparse_attn.%s not importable: %s", name, e)
        return False

    sys.modules[fq] = mod
    setattr(comfy, name, mod)
    log.info("Aliased %s -> comfy_sparse_attn.%s", fq, name)
    return True
