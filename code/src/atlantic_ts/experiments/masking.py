"""Deterministic, model-shared random masking interface."""

from __future__ import annotations

import hashlib


def deterministic_mask(record: dict[str, str], seed: int = 20260729, rate: float = 0.2) -> bool:
    """Return the common 20% mask decision from immutable M1 lineage fields."""
    if not 0.0 < rate < 1.0:
        raise ValueError("mask rate must be between zero and one")
    key = "|".join(record[field] for field in ("file_id", "source_depth_index_0", "source_time_index_0", "value_token_raw"))
    value = int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "big") / 2**64
    return value < rate
