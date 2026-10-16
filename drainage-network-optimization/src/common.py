"""Shared helpers: config loading and project paths.

Every other module imports from here so the whole pipeline reads one config.yaml
and the three algorithms run on an identical setup (fair comparison).
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# Project root is the parent of this src/ directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Parse config.yaml once and cache it."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Resolve a configured directory (under ``paths:``) to an absolute Path,
    creating it if needed. ``key`` is e.g. 'data_dir', 'tables_dir'."""
    cfg = load_config()
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_inp() -> Path:
    """Locate the EPA SWMM .inp model inside the data directory."""
    cfg = load_config()
    data_dir = ROOT / cfg["paths"]["data_dir"]
    matches = sorted(data_dir.rglob(cfg["dataset"]["inp_glob"]))
    if not matches:
        raise FileNotFoundError(
            f"No SWMM .inp found under {data_dir}. Run `python -m src.download_data` first."
        )
    # Prefer the largest .inp (the full distributed model, not a rain/snippet file).
    return max(matches, key=lambda f: f.stat().st_size)
