from __future__ import annotations

SYCOPHANCY_DEFLATOR: float = 0.70


def deflate(rate: float) -> float:
    return round(rate * SYCOPHANCY_DEFLATOR, 4)
