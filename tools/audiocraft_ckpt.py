"""Read an audiocraft checkpoint (`state_dict.bin` / `compression_state_dict.bin`).

Both MelodyFlow and every `thepatch/*` MusicGen finetune ship raw audiocraft pickles
rather than safetensors, and the config the model was built from is embedded as an
OmegaConf blob under `xp.cfg`. This module is the one place that knows how to open them,
so the converters stay declarative.

torch and omegaconf are imported lazily and with a pointed error message, because they are
the only heavy dependencies in this repo and a user converting a checkpoint may be running
from a gary4local service venv rather than ours.
"""

from __future__ import annotations

import typing as tp
from collections import defaultdict
from pathlib import Path


class CheckpointError(RuntimeError):
    pass


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CheckpointError(
            "reading an audiocraft checkpoint needs torch. Either install it here "
            "(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) or run "
            "this script with one of gary4local's service venvs, e.g. "
            "services/melodyflow/env/Scripts/python.exe"
        ) from exc
    return torch


def _safe_globals(torch):
    """The allowlist `xp.cfg` needs under torch>=2.6's weights_only default.

    Mirrors audiocraft's own `_CHECKPOINT_SAFE_GLOBALS` (models/loaders.py). Loading with
    weights_only=True matters here: these files are downloaded from the Hub, and a plain
    `torch.load` on a pickle is arbitrary code execution.
    """
    try:
        from omegaconf import DictConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CheckpointError(
            "reading an audiocraft checkpoint needs omegaconf (`pip install omegaconf`)"
        ) from exc
    return [DictConfig, ListConfig, AnyNode, Metadata, ContainerMetadata,
            defaultdict, dict, list, int, tp.Any]


def load(path: str | Path) -> tuple[dict, dict]:
    """Return (state_dict, config) from an audiocraft checkpoint.

    The state dict is `best_state` (audiocraft's exported inference weights) and the config
    is `xp.cfg` resolved to plain Python containers, so callers never touch OmegaConf.
    """
    torch = _load_torch()
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no such checkpoint: {path}")

    with torch.serialization.safe_globals(_safe_globals(torch)):
        pkg = torch.load(str(path), map_location="cpu", weights_only=True)

    if "best_state" not in pkg:
        raise CheckpointError(
            f"{path.name} has no 'best_state' (keys: {sorted(pkg)[:8]}); this does not look "
            "like an audiocraft checkpoint")
    if "xp.cfg" not in pkg:
        raise CheckpointError(f"{path.name} has no 'xp.cfg'")

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.create(pkg["xp.cfg"]), resolve=False)
    state = {k: v.detach().cpu().numpy() for k, v in pkg["best_state"].items()}
    return state, cfg


def require(cfg: dict, *path: str):
    """Fetch a nested config value, failing with the full dotted path if it is absent."""
    node = cfg
    for i, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            raise CheckpointError("checkpoint config is missing " + ".".join(path[:i + 1]))
        node = node[key]
    return node


def optional(cfg: dict, *path: str, default=None):
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
