"""F2P (fail-to-pass) partial-credit transforms. All map pass fraction in [0,1] -> credit in [0,1]."""
from __future__ import annotations

import math


def transform(frac: float, mode: str = "pow", gamma: float = 1.5, ladder=((0.34, 0.2), (0.67, 0.5), (1.0, 1.0))) -> float:
    f = min(1.0, max(0.0, float(frac)))
    if mode == "binary":
        return 1.0 if f >= 1.0 else 0.0
    if mode == "linear":
        return f
    if mode == "sqrt":
        return math.sqrt(f)
    if mode == "pow":
        return f ** gamma
    if mode == "ladder":          # threshold scaffold: partial / mostly / complete
        credit = 0.0
        for thr, c in sorted(ladder):
            if f >= thr - 1e-9:
                credit = c
        return credit
    raise ValueError(f"unknown f2p mode {mode}")
