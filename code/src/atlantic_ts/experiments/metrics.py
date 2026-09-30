"""Dependency-free M2 metric definitions."""

from __future__ import annotations

import math


def rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("metric inputs must be equal non-empty lengths")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def r2(actual: list[float], predicted: list[float]) -> float:
    mean = sum(actual) / len(actual)
    total = sum((a - mean) ** 2 for a in actual)
    return float("nan") if total == 0 else 1 - sum((a - p) ** 2 for a, p in zip(actual, predicted)) / total


def anomaly_acc(actual: list[float], predicted: list[float], climatology: list[float]) -> float:
    a = [x - c for x, c in zip(actual, climatology)]
    p = [x - c for x, c in zip(predicted, climatology)]
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in p))
    return float("nan") if denominator == 0 else sum(x * y for x, y in zip(a, p)) / denominator
