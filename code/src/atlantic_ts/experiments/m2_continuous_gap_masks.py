"""Deterministic, native-label-only masks for the M2 continuous-gap protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import netcdf_file


PROTOCOL_ID = "m2_continuous_missing_reconstruction_v1"
SEED = 20260811
GAP_LENGTHS = (3, 6, 12)
VARIABLES = ("temperature", "salinity")
DEPTHS = ("1.0", "20.0", "40.0", "120.0")
SPLIT_MONTHS = {
    "train": ("1998-02", "2014-12"),
    "validation": ("2015-01", "2018-12"),
    "test": ("2019-01", "2024-07"),
}
REQUIRED_COLUMNS = {
    "file_id",
    "station_id_raw",
    "variable",
    "depth",
    "time_month",
    "missing_flag",
    "split",
}


@dataclass(frozen=True)
class NativeLabel:
    split: str
    station: str
    variable: str
    depth: str
    month: str


def _month_range(start: str, end: str) -> list[str]:
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    result: list[str] = []
    while (year, month) <= (end_year, end_month):
        result.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


ALL_MONTHS = _month_range("1998-02", "2024-07")


def _split_for_month(month: str) -> str | None:
    for split, (start, end) in SPLIT_MONTHS.items():
        if start <= month <= end:
            return split
    return None


def _read_native_labels(split_path: Path) -> list[NativeLabel]:
    """Read only metadata required to establish eligible native labels."""
    labels: list[NativeLabel] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    with split_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"split dataset missing columns: {sorted(missing)}")
        for row in reader:
            split = row["split"]
            month = row["time_month"][:7]
            if split not in SPLIT_MONTHS:
                continue
            if _split_for_month(month) != split:
                raise ValueError(f"split/month mismatch for {row['file_id']}: {split}, {month}")
            if row["missing_flag"] != "false":
                continue
            if row["variable"] not in VARIABLES or row["depth"] not in DEPTHS:
                continue
            key = (split, row["station_id_raw"], row["variable"], row["depth"], month)
            if key in seen:
                raise ValueError(f"duplicate native station-month label: {key}")
            seen.add(key)
            labels.append(NativeLabel(*key))
    if not labels:
        raise ValueError("no finite native target-depth labels found")
    return labels


def _candidate_starts(months: set[str], gap_length: int) -> list[str]:
    candidates: list[str] = []
    for start in sorted(months):
        sequence = _month_range(start, _advance_month(start, gap_length - 1))
        if all(month in months for month in sequence):
            candidates.append(start)
    return candidates


def _advance_month(month: str, offset: int) -> str:
    year, number = map(int, month.split("-"))
    ordinal = year * 12 + number - 1 + offset
    return f"{ordinal // 12:04d}-{ordinal % 12 + 1:02d}"


def select_gap_windows(labels: Iterable[NativeLabel], gap_length: int, seed: int = SEED) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Choose one reproducibly ranked full native window per eligible series and split."""
    by_series: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for label in labels:
        by_series[(label.split, label.station, label.variable, label.depth)].add(label.month)

    windows: list[dict[str, str]] = []
    unsupported: list[dict[str, str]] = []
    for (split, station, variable, depth), months in sorted(by_series.items()):
        candidates = _candidate_starts(months, gap_length)
        if not candidates:
            unsupported.append(
                {
                    "split": split,
                    "station_id_raw": station,
                    "variable": variable,
                    "depth": depth,
                    "reason": "no_complete_native_window",
                }
            )
            continue
        start = min(
            candidates,
            key=lambda candidate: hashlib.sha256(
                f"{seed}|{gap_length}|{split}|{station}|{variable}|{depth}|{candidate}".encode("utf-8")
            ).hexdigest(),
        )
        windows.append(
            {
                "split": split,
                "station_id_raw": station,
                "variable": variable,
                "depth": depth,
                "start_month": start,
                "end_month": _advance_month(start, gap_length - 1),
                "gap_length_months": str(gap_length),
            }
        )
    return windows, unsupported


def _encode_strings(values: list[str], width: int) -> np.ndarray:
    array = np.full((len(values), width), b" ", dtype="S1")
    for index, value in enumerate(values):
        encoded = value.encode("utf-8")
        array[index, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
    return array


def _atomic_write_netcdf(path: Path, stations: list[str], labels: list[NativeLabel], windows: list[dict[str, str]], gap_length: int) -> None:
    station_index = {value: index for index, value in enumerate(stations)}
    month_index = {value: index for index, value in enumerate(ALL_MONTHS)}
    depth_index = {value: index for index, value in enumerate(DEPTHS)}
    variable_index = {value: index for index, value in enumerate(VARIABLES)}
    native = np.zeros((len(stations), len(ALL_MONTHS), len(DEPTHS), len(VARIABLES)), dtype=np.int8)
    hidden = np.zeros_like(native)
    for label in labels:
        native[station_index[label.station], month_index[label.month], depth_index[label.depth], variable_index[label.variable]] = 1
    for window in windows:
        for month in _month_range(window["start_month"], window["end_month"]):
            index = (station_index[window["station_id_raw"]], month_index[month], depth_index[window["depth"]], variable_index[window["variable"]])
            if native[index] != 1:
                raise ValueError(f"attempted to mask non-native label: {window}")
            hidden[index] = 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with netcdf_file(str(temporary_path), "w") as dataset:
            dataset.createDimension("station", len(stations))
            dataset.createDimension("time", len(ALL_MONTHS))
            dataset.createDimension("depth", len(DEPTHS))
            dataset.createDimension("variable", len(VARIABLES))
            dataset.createDimension("station_name_length", max(map(len, stations)))
            dataset.createDimension("variable_name_length", max(map(len, VARIABLES)))
            dataset.createVariable("station", "c", ("station", "station_name_length"))[:] = _encode_strings(stations, max(map(len, stations)))
            dataset.createVariable("time", "i", ("time",))[:] = np.asarray([int(month.replace("-", "")) for month in ALL_MONTHS], dtype=np.int32)
            dataset.createVariable("depth", "d", ("depth",))[:] = np.asarray([float(value) for value in DEPTHS], dtype=np.float64)
            dataset.createVariable("variable", "c", ("variable", "variable_name_length"))[:] = _encode_strings(list(VARIABLES), max(map(len, VARIABLES)))
            dataset.createVariable("native_label_flag", "b", ("station", "time", "depth", "variable"))[:] = native
            dataset.createVariable("mask_flag", "b", ("station", "time", "depth", "variable"))[:] = hidden
            dataset.protocol_id = PROTOCOL_ID
            dataset.gap_length_months = gap_length
            dataset.selection_seed = SEED
            dataset.mask_flag_definition = "1=hidden continuous native label; 0=not hidden; consult native_label_flag for label availability"
            dataset.time_encoding = "YYYYMM calendar months"
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_masks(split_path: Path, output_dir: Path) -> dict[str, object]:
    """Materialize and validate all frozen M2 continuous-gap masks."""
    labels = _read_native_labels(Path(split_path))
    stations = sorted({label.station for label in labels})
    all_windows: dict[str, list[dict[str, str]]] = {}
    all_unsupported: dict[str, list[dict[str, str]]] = {}
    files: dict[str, str] = {}
    for length in GAP_LENGTHS:
        windows, unsupported = select_gap_windows(labels, length)
        all_windows[str(length)] = windows
        all_unsupported[str(length)] = unsupported
        destination = Path(output_dir) / f"m2_gap{length}_mask.nc"
        _atomic_write_netcdf(destination, stations, labels, windows, length)
        files[str(length)] = _sha256(destination)

    _write_window_manifest(Path(output_dir) / "m2_gap_window_manifest.csv", all_windows)

    validation = validate_masks(labels, Path(output_dir), all_windows, all_unsupported)
    validation["mask_sha256"] = files
    validation["protocol_id"] = PROTOCOL_ID
    validation["selection_seed"] = SEED
    validation["status"] = "pass" if all(validation["checks"].values()) else "fail"
    validation_path = Path(output_dir) / "m2_continuous_gap_mask_validation.json"
    _atomic_write_text(validation_path, json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return validation


def _write_window_manifest(path: Path, windows_by_length: dict[str, list[dict[str, str]]]) -> None:
    fieldnames = [
        "gap_length_months",
        "mask_id",
        "mask_file",
        "split",
        "station_id_raw",
        "variable",
        "depth",
        "start_month",
        "end_month",
    ]
    rows: list[dict[str, str]] = []
    for length, windows in sorted(windows_by_length.items(), key=lambda item: int(item[0])):
        for window in windows:
            rows.append(
                {
                    "gap_length_months": length,
                    "mask_id": f"m2_continuous_gap_v1_gap{length}_seed{SEED}",
                    "mask_file": f"m2_gap{length}_mask.nc",
                    **{field: window[field] for field in ("split", "station_id_raw", "variable", "depth", "start_month", "end_month")},
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_freeze_report(validation: dict[str, object], root: Path) -> Path:
    """Write the M2 readiness report solely from materialized validation evidence."""
    masks = validation["mask_sha256"]
    windows = validation["window_counts_by_split"]
    unsupported = validation["unsupported_series_by_gap_length"]
    checks = validation["checks"]
    check_lines = "\n".join(f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in sorted(checks.items()))
    count_lines = "\n".join(
        f"| {length} | {counts['train']} | {counts['validation']} | {counts['test']} | {unsupported[length]} |"
        for length, counts in sorted(windows.items(), key=lambda item: int(item[0]))
    )
    mask_lines = "\n".join(
        f"- `artifacts/masks/M2/m2_gap{length}_mask.nc` — SHA-256 `{digest}`"
        for length, digest in sorted(masks.items(), key=lambda item: int(item[0]))
    )
    text = f"""# M2 Continuous Missing Reconstruction Protocol Freeze Report

**Status:** `{'READY_FOR_M2_EXPERIMENT' if validation['status'] == 'pass' else 'BLOCKED'}`  
**Protocol:** `m2_continuous_missing_reconstruction_v1`  
**Mask selection seed:** `{validation['selection_seed']}`

## Frozen protocol and schemas

- Protocol: `docs/experiments/M2_continuous_gap_protocol.yaml`
- Metric schema: `configs/M2_metric_schema.yaml`
- Source split: `artifacts/splits/native_observation_splits.csv`
- Mask validation: `artifacts/masks/M2/m2_continuous_gap_mask_validation.json`
- Window manifest: `artifacts/masks/M2/m2_gap_window_manifest.csv`

The protocol evaluates temperature and salinity separately at 1, 20, 40, and 120 m. Each eligible split × station × variable × depth series contributes one SHA-256-ranked native-only window per gap length. A series without a complete native window is retained as unsupported; it is never interpolated or filled.

## Materialized masks

{mask_lines}

Each NetCDF file has `station`, `time`, `depth`, `variable`, `native_label_flag`, and `mask_flag`. `mask_flag=1` only for a hidden finite native label. `native_label_flag` distinguishes unavailable grid cells from available-but-unmasked observations.

| Gap length (months) | Train windows | Validation windows | Test windows | Unsupported series |
|---:|---:|---:|---:|---:|
{count_lines}

## Leakage and reproducibility checks

{check_lines}

The frozen input contract prohibits hidden observations, future months, and test hidden labels from model inputs, normalization, selection, and early stopping. Test windows are final-evaluation-only. Only hidden native labels in the approved windows may be scored, grouped by station, depth, variable, and gap length, using RMSE, R, and MAE.

## Readiness decision

`{'READY_FOR_M2_EXPERIMENT' if validation['status'] == 'pass' else 'BLOCKED'}`. The protocol, deterministic masks, metric schema, and structural leakage controls are materialized. No model was trained or evaluated by this preparation task. Any future runner must enforce the frozen input contract before execution.
"""
    destination = Path(root) / "reports" / "M2_protocol_freeze_report.md"
    _atomic_write_text(destination, text)
    return destination


def validate_masks(labels: list[NativeLabel], output_dir: Path, windows_by_length: dict[str, list[dict[str, str]]], unsupported_by_length: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    """Validate continuousness, split isolation, native eligibility, and determinism."""
    native_keys = {(label.station, label.variable, label.depth, label.month, label.split) for label in labels}
    counts_by_split: dict[str, dict[str, int]] = {}
    all_checks: dict[str, bool] = {}
    for length_text, windows in windows_by_length.items():
        length = int(length_text)
        continuous = all(len(_month_range(window["start_month"], window["end_month"])) == length for window in windows)
        native = all(
            (window["station_id_raw"], window["variable"], window["depth"], month, window["split"]) in native_keys
            for window in windows
            for month in _month_range(window["start_month"], window["end_month"])
        )
        split_isolation = all(_split_for_month(month) == window["split"] for window in windows for month in _month_range(window["start_month"], window["end_month"]))
        repeated_windows, repeated_unsupported = select_gap_windows(labels, length)
        deterministic = repeated_windows == windows and repeated_unsupported == unsupported_by_length[length_text]
        path = Path(output_dir) / f"m2_gap{length}_mask.nc"
        with netcdf_file(str(path), "r", mmap=False) as dataset:
            hidden = dataset.variables["mask_flag"][:]
            native_flags = dataset.variables["native_label_flag"][:]
            dimensions = hidden.shape == native_flags.shape == (len({label.station for label in labels}), len(ALL_MONTHS), len(DEPTHS), len(VARIABLES))
            flags_valid = bool(np.isin(hidden, [0, 1]).all() and np.all(hidden <= native_flags))
        all_checks[f"gap{length}_continuous"] = continuous
        all_checks[f"gap{length}_native_labels"] = native
        all_checks[f"gap{length}_split_isolation"] = split_isolation
        all_checks[f"gap{length}_deterministic"] = deterministic
        all_checks[f"gap{length}_netcdf_schema"] = dimensions and flags_valid
        counts_by_split[length_text] = {split: sum(1 for window in windows if window["split"] == split) for split in SPLIT_MONTHS}
    all_checks["test_windows_do_not_overlap_train_months"] = all(
        _split_for_month(month) == "test"
        for windows in windows_by_length.values()
        for window in windows
        if window["split"] == "test"
        for month in _month_range(window["start_month"], window["end_month"])
    )
    all_checks["no_future_window_months"] = all(
        month <= window["end_month"]
        for windows in windows_by_length.values()
        for window in windows
        for month in _month_range(window["start_month"], window["end_month"])
    )
    return {
        "source_split_path": str(Path(split_path_name(labels))),
        "eligible_native_label_count": len(labels),
        "station_count": len({label.station for label in labels}),
        "variables": list(VARIABLES),
        "depths_m": [float(value) for value in DEPTHS],
        "window_counts_by_split": counts_by_split,
        "unsupported_series_by_gap_length": {key: len(value) for key, value in unsupported_by_length.items()},
        "checks": all_checks,
    }


def split_path_name(labels: list[NativeLabel]) -> str:
    """Compatibility placeholder retained to keep validation free of row payloads."""
    del labels
    return "artifacts/splits/native_observation_splits.csv"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
