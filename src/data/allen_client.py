"""Shared connection utility for the Allen Institute BrainObservatoryCache.

This is infrastructure, not a scientific analysis thread: it just resolves
where the AllenSDK manifest lives and hands back a cached cache handle so
every Allen-derived analysis thread opens the same underlying HDF5-backed
cache once instead of repeatedly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_MANIFEST_PATH_ENV_VARS = ("ALLEN_MANIFEST_PATH", "ALLEN_DATA")
_dotenv_loaded = False


def _load_dotenv_once() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    from dotenv import load_dotenv

    load_dotenv()
    _dotenv_loaded = True


def resolve_manifest_path(manifest_path: Path | str | None = None) -> Path:
    """Resolve the BrainObservatoryCache manifest JSON path.

    Precedence: explicit `manifest_path` argument > ALLEN_MANIFEST_PATH env
    var > ALLEN_DATA env var > raise.
    """
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()

    _load_dotenv_once()
    for var in _MANIFEST_PATH_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser().resolve()

    raise RuntimeError(
        "Allen manifest path is unset. Pass manifest_path explicitly, or set "
        "ALLEN_MANIFEST_PATH (preferred) or ALLEN_DATA in the environment / .env."
    )


@lru_cache(maxsize=4)
def _cached_brain_observatory_cache(manifest_path_str: str) -> Any:
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError as exc:
        raise ImportError(
            "allensdk is not installed. Install it (see requirements.txt) to "
            "load Allen Institute calcium imaging data."
        ) from exc
    return BrainObservatoryCache(manifest_file=manifest_path_str)


def get_brain_observatory_cache(manifest_path: Path) -> Any:
    """Return a cached BrainObservatoryCache for the given manifest path."""
    return _cached_brain_observatory_cache(str(manifest_path))
