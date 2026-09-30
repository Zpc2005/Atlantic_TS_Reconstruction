"""Fail-closed M3 training-record adapter with no model execution path."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from atlantic_ts.experiments.m3_normalization import TARGET_DEPTHS, TRAIN_END, TRAIN_START, VARIABLES
from atlantic_ts.experiments.m3_runner import M3Fold, M3Manifest


Mode = Literal["primary", "sensitivity"]


class M3DatasetAdapterError(ValueError):
    """Raised when M3 records would violate the frozen feature contract."""


@dataclass(frozen=True)
class M3FeatureScan:
    fold_id: str
    mode: Mode
    feature_count: int
    record_count: int
    held_out_record_count: int
    woa23_prior_count: int


FEATURES: dict[Mode, tuple[str, ...]] = {
    "primary": ("latitude", "longitude", "month", "depth_m", "woa23_background_prior"),
    "sensitivity": ("latitude", "longitude", "month", "depth_m"),
}


def _geometry(root: Path) -> dict[str, tuple[float, float]]:
    path = root / "artifacts/native/station_geometry_registry.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"station_id_raw", "latitude", "longitude"} <= set(reader.fieldnames):
            raise M3DatasetAdapterError("station geometry schema mismatch")
        return {row["station_id_raw"]: (float(row["latitude"]), float(row["longitude"])) for row in reader}


def _validate_normalizer(path: Path, manifest: M3Manifest, fold: M3Fold) -> dict[str, object]:
    if not path.is_file():
        raise M3DatasetAdapterError(f"normalizer missing: {path}")
    normalizer = json.loads(path.read_text(encoding="utf-8"))
    if normalizer.get("fold_id") != fold.fold_id or normalizer.get("manifest_sha256") != manifest.sha256:
        raise M3DatasetAdapterError("normalizer does not bind the frozen fold")
    if normalizer.get("fit_partition") != "train":
        raise M3DatasetAdapterError("normalizer is not train-only")
    if normalizer.get("validation_contributors") != 0 or normalizer.get("test_contributors") != 0:
        raise M3DatasetAdapterError("validation or test entered normalizer")
    if normalizer.get("held_out_station_contributors") != 0:
        raise M3DatasetAdapterError("held-out station entered normalizer")
    return normalizer


def _require_primary_lookup(root: Path) -> None:
    path = root / "artifacts/algorithm/m3_spatial_extrapolation/m3_woa23_horizontal_sampling.yaml"
    if not path.is_file():
        raise M3DatasetAdapterError(
            "BLOCKED: frozen WOA23 horizontal sampling rule is missing; primary features cannot be constructed"
        )
    text = path.read_text(encoding="utf-8")
    required = ("horizontal_rule: nearest_native_grid", "interpolation_allowed: false", "vertical_interpolation_allowed: false")
    if any(item not in text for item in required):
        raise M3DatasetAdapterError("WOA23 horizontal sampling rule permits interpolation")


def scan_training_features(
    project_root: Path,
    manifest: M3Manifest,
    fold: M3Fold,
    mode: Mode,
    normalizer_path: Path,
    woa23_prior: Any | None = None,
) -> M3FeatureScan:
    """Stream training metadata and validate the feature contract without training."""
    if mode not in FEATURES:
        raise M3DatasetAdapterError(f"unsupported M3 mode: {mode}")
    root = Path(project_root)
    normalizer = _validate_normalizer(Path(normalizer_path), manifest, fold)
    if mode == "primary":
        _require_primary_lookup(root)
        woa_stats = normalizer.get("feature_statistics", {}).get("woa23_background_prior", {})
        if not isinstance(woa_stats, dict) or not isinstance(woa_stats.get("n"), int) or woa_stats["n"] <= 0:
            raise M3DatasetAdapterError("primary normalizer has no train-only WOA23 prior binding")
    geometry = _geometry(root)
    count = 0
    woa23_prior_count = 0
    held_out_feature_records = 0
    labels = root / manifest.label_path
    prior = woa23_prior
    own_prior = False
    if mode == "primary":
        if prior is None:
            from atlantic_ts.experiments.m3_woa23_prior import Woa23NativeGridPrior

            prior = Woa23NativeGridPrior(root)
            own_prior = True
    try:
        with labels.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if row["split"] != "train" or not (TRAIN_START <= row["time_month"] <= TRAIN_END):
                    continue
                station = row["station_id_raw"]
                if station == fold.held_out_station:
                    continue
                if station not in fold.training_stations:
                    continue
                if row["variable"] not in VARIABLES or float(row["depth"]) not in TARGET_DEPTHS:
                    continue
                if station not in geometry:
                    raise M3DatasetAdapterError(f"missing geometry for training station: {station}")
                if prior is not None:
                    latitude, longitude = geometry[station]
                    prior.value(row["variable"], int(row["month_raw"]), float(row["depth"]), latitude, longitude)
                    woa23_prior_count += 1
                count += 1
    finally:
        if own_prior and prior is not None:
            prior.close()
    if count == 0:
        raise M3DatasetAdapterError("no training records available for fold")
    return M3FeatureScan(
        fold.fold_id,
        mode,
        len(FEATURES[mode]),
        count,
        held_out_feature_records,
        woa23_prior_count,
    )
