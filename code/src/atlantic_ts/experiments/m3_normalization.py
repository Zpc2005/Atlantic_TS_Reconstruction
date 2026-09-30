"""Generate shared, train-only normalizers for frozen M3 LOSO folds."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from atlantic_ts.experiments.m3_runner import M3Fold, M3Manifest, load_m3_loso_manifest
from atlantic_ts.io.atomic import atomic_write_text


TARGET_DEPTHS = (1.0, 20.0, 40.0, 120.0)
VARIABLES = ("temperature", "salinity")
TRAIN_START = "1998-02"
TRAIN_END = "2014-12"


class M3NormalizationError(ValueError):
    """Raised when a fold cannot produce a leakage-safe normalizer."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_std(values: list[float], label: str) -> dict[str, float | int]:
    if not values:
        raise M3NormalizationError(f"no train values for {label}")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if not math.isfinite(mean) or not math.isfinite(std) or std == 0.0:
        raise M3NormalizationError(f"invalid train-only statistic for {label}")
    return {"mean": mean, "std": std, "n": len(values)}


def _read_geometry(path: Path) -> dict[str, tuple[float, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"station_id_raw", "latitude", "longitude"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise M3NormalizationError("station geometry schema mismatch")
        geometry = {
            row["station_id_raw"]: (float(row["latitude"]), float(row["longitude"]))
            for row in reader
        }
    return geometry


def _is_train_member(row: dict[str, str], fold: M3Fold) -> bool:
    return (
        row["split"] == "train"
        and row["station_id_raw"] in fold.training_stations
        and row["station_id_raw"] != fold.held_out_station
        and TRAIN_START <= row["time_month"] <= TRAIN_END
        and row["variable"] in VARIABLES
        and float(row["depth"]) in TARGET_DEPTHS
        and row["missing_flag"].strip().lower() in {"false", "0"}
        and row["value_native"].strip() != ""
    )


def build_fold_normalizer(
    project_root: Path,
    manifest: M3Manifest,
    fold: M3Fold,
    woa23_prior: Any | None = None,
) -> dict[str, Any]:
    """Fit a fold normalizer from native training records only."""
    root = Path(project_root)
    labels = root / manifest.label_path
    geometry_path = root / "artifacts/native/station_geometry_registry.csv"
    if _sha256(labels) != manifest.label_sha256:
        raise M3NormalizationError("native label source SHA-256 mismatch")
    geometry = _read_geometry(geometry_path)
    missing_geometry = set(fold.training_stations) - set(geometry)
    if missing_geometry:
        raise M3NormalizationError(f"training station geometry missing: {sorted(missing_geometry)}")

    targets: dict[tuple[str, float], list[float]] = defaultdict(list)
    months: list[float] = []
    depths: list[float] = []
    woa_values: list[float] = []
    train_station_ids: set[str] = set()
    prior = woa23_prior
    own_prior = False
    sampling_rule = root / "artifacts/algorithm/m3_spatial_extrapolation/m3_woa23_horizontal_sampling.yaml"
    if sampling_rule.is_file():
        from atlantic_ts.experiments.m3_woa23_prior import Woa23NativeGridPrior

        prior = Woa23NativeGridPrior(root)
        own_prior = True
    try:
        with labels.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if not _is_train_member(row, fold):
                    continue
                value = float(row["value_native"])
                key = (row["variable"], float(row["depth"]))
                targets[key].append(value)
                months.append(float(row["month_raw"]))
                depths.append(float(row["depth"]))
                station = row["station_id_raw"]
                train_station_ids.add(station)
                if prior is not None:
                    latitude, longitude = geometry[station]
                    woa_values.append(prior.value(row["variable"], int(row["month_raw"]), float(row["depth"]), latitude, longitude))
    finally:
        if own_prior and prior is not None:
            prior.close()
    if fold.held_out_station in train_station_ids:
        raise M3NormalizationError("held-out station entered normalizer fitting")
    if train_station_ids - set(fold.training_stations):
        raise M3NormalizationError("non-training station entered normalizer fitting")
    target_stats = {
        f"{variable}:{int(depth)}": _mean_std(targets[(variable, depth)], f"{variable}:{depth}")
        for variable in VARIABLES
        for depth in TARGET_DEPTHS
    }
    latitudes = [geometry[station][0] for station in fold.training_stations]
    longitudes = [geometry[station][1] for station in fold.training_stations]
    return {
        "normalizer_id": f"m3_train_only_{fold.fold_id}_v1",
        "status": "PASS_SHARED_BASE_NORMALIZER",
        "fold_id": fold.fold_id,
        "held_out_station": fold.held_out_station,
        "training_stations": list(fold.training_stations),
        "manifest_sha256": manifest.sha256,
        "label_source": manifest.label_path,
        "label_source_sha256": manifest.label_sha256,
        "fit_partition": "train",
        "fit_time_window": {"start": TRAIN_START, "end": TRAIN_END},
        "validation_contributors": 0,
        "test_contributors": 0,
        "held_out_station_contributors": 0,
        "train_station_count_with_records": len(train_station_ids),
        "shared_between_modes": True,
        "feature_statistics": {
            "latitude": _mean_std(latitudes, "latitude"),
            "longitude": _mean_std(longitudes, "longitude"),
            "month": {
                "normalization": "not_fitted_cyclic_feature",
                "n": len(months),
            },
            "depth_m": _mean_std(depths, "depth_m"),
            "woa23_background_prior": (
                _mean_std(woa_values, "woa23_background_prior")
                if prior is not None
                else {"status": "UNBOUND_HORIZONTAL_SAMPLING_RULE_REQUIRED", "n": 0}
            ),
        },
        "target_statistics": target_stats,
        "prohibitions": {
            "full_development_normalizer": True,
            "validation_or_test_fitting": True,
            "held_out_station_fitting": True,
        },
    }


def materialize_fold_normalizers(project_root: Path, output_dir: Path) -> list[Path]:
    """Create exactly one shared base normalizer for every frozen M3 fold."""
    root = Path(project_root)
    manifest = load_m3_loso_manifest(root / "artifacts/algorithm/m3_spatial_extrapolation/m3_loso_manifest.yaml")
    output_dir = Path(output_dir)
    paths: list[Path] = []
    sampling_rule = root / "artifacts/algorithm/m3_spatial_extrapolation/m3_woa23_horizontal_sampling.yaml"
    prior = None
    if sampling_rule.is_file():
        from atlantic_ts.experiments.m3_woa23_prior import Woa23NativeGridPrior

        prior = Woa23NativeGridPrior(root)
    try:
        for fold in manifest.folds:
            normalizer = build_fold_normalizer(root, manifest, fold, woa23_prior=prior)
            path = output_dir / f"{fold.fold_id}.json"
            atomic_write_text(path, json.dumps(normalizer, indent=2, sort_keys=True) + "\n")
            paths.append(path)
    finally:
        if prior is not None:
            prior.close()
    if len(paths) != 17:
        raise M3NormalizationError("M3 normalizer count mismatch")
    return paths


def write_normalization_report(path: Path, normalizer_paths: list[Path], project_root: Path) -> None:
    """Write a concise provenance report without exposing numeric statistics."""
    lines = (
        "# M3 normalization generation report",
        "",
        "**Status: PASS for 17 shared train-only base normalizers.**",
        "",
        "- One normalizer was generated for each frozen LOSO fold.",
        "- Each was fit only from that fold's training stations and 1998-02 through 2014-12 train records.",
        "- Validation/test contributors and held-out-station contributors are zero by construction and recorded in every file.",
        "- The normalizers are shared between primary and sensitivity.",
        "- The optional WOA23 prior statistic, when bound, uses only the same fold training support and the frozen nearest-native-grid rule.",
        "",
        "## Files",
        "",
        *[f"- `{path.relative_to(project_root).as_posix()}`" for path in normalizer_paths],
    )
    atomic_write_text(Path(path), "\n".join(lines) + "\n")
