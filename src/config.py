"""
Muhafiz.sol - central configuration loader.

Every script reads paths, seeds and hyper-parameters from ``config.yaml`` through
this module so that no value is hard-coded in the code (blueprint P1.1).
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = parent of the ``src`` package. All relative paths in
# config.yaml are resolved against it, so scripts work from any working dir.
ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load ``config.yaml`` once and cache it."""
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def variant() -> str:
    """Model variant: '' for model-v1, or a key of the ``variants`` section (set MUHAFIZ_VARIANT=v2)."""
    return os.environ.get("MUHAFIZ_VARIANT", "").strip()


def path(key: str) -> Path:
    """Return an absolute path for an entry of the ``paths`` section (redirected for a model variant)."""
    cfg = load_config()
    v = variant()
    if v:
        over = cfg.get("variants", {}).get(v, {}).get("paths", {})
        if key in over:
            return ROOT / over[key]
    return ROOT / cfg["paths"][key]


def freeze_tag() -> str:
    v = variant()
    return load_config().get("variants", {}).get(v, {}).get("tag", "model-v1") if v else "model-v1"


def seed() -> int:
    return int(load_config()["project"]["random_seed"])


def set_global_seed(value: int | None = None) -> int:
    """Seed python, numpy (and torch if installed) for reproducibility."""
    value = seed() if value is None else value
    random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    try:
        import numpy as np

        np.random.seed(value)
    except ImportError:  # pragma: no cover
        pass
    try:  # torch is only needed for MLP training
        import torch

        torch.manual_seed(value)
    except ImportError:
        pass
    return value


def read_json(p: os.PathLike | str) -> Any:
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj: Any, p: os.PathLike | str, indent: int = 2) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
