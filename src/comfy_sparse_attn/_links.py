"""
Filesystem link setup: inject sparse primitives into the comfy namespace.

Creates symlinks at comfy/{target} → source so that code written as
    from comfy.ops_sparse import ...
    from comfy.attention_sparse import ...
    from comfy.sparse import VarLenTensor
works NOW, before the PR lands in ComfyUI main.

When ComfyUI ships the real files, the links become no-ops: a real file
(non-symlink) at the target path is left untouched.
"""

import logging
import pathlib

log = logging.getLogger("comfy_sparse_attn")


def setup_link(source: pathlib.Path, comfy_relative: str) -> bool:
    """
    Create a symlink at comfy/{comfy_relative} → source.

    Skips if:
      - The target is a real file (ComfyUI shipped it natively — nothing to do).
      - The target is already a symlink pointing to source.

    Removes and recreates if the target is a stale symlink pointing elsewhere.

    Returns True if a new link was created.
    """
    try:
        import comfy
    except ImportError:
        log.warning("comfy not found on sys.path, skipping sparse link setup")
        return False

    comfy_dir = pathlib.Path(comfy.__path__[0])
    source = source.resolve()
    target = comfy_dir / comfy_relative

    if not source.exists():
        log.warning(f"Source does not exist, skipping: {source}")
        return False

    if target.exists() and not target.is_symlink():
        # Real file — ComfyUI has shipped this natively.
        log.debug(f"comfy/{comfy_relative} is a real file (native), skipping junction")
        return False

    if target.is_symlink():
        if target.resolve() == source:
            return False  # Already correctly linked.
        log.debug(f"comfy/{comfy_relative}: stale symlink, relinking")
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    log.info(f"Linked comfy/{comfy_relative} -> {source}")
    return True
