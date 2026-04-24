"""
CuPy/NumPy compatibility shim.

Wordle's original pipeline relies on CuPy for GPU acceleration. This module
makes that dependency optional: if CuPy is not installed (e.g. on a machine
without a CUDA toolkit), we transparently fall back to NumPy so that the
entire CPU path still runs. The extended Mastermind and Zoo games are
CPU-only by design and do not require CuPy.

Usage:
    from utils.xp_utils import cp, HAS_CUPY

If HAS_CUPY is False, ``cp`` is an alias of ``numpy``; callers must still
respect ``configs['GPU']`` to avoid attempting a GPU-only code path.
"""
import numpy as np

try:
    import cupy as _cp
    cp = _cp
    HAS_CUPY = True
except Exception:
    cp = np
    HAS_CUPY = False
