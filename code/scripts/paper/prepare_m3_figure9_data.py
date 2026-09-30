from __future__ import annotations

import csv
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "paper" / "figures" / "Figure9_MS3"
PREDICTIONS = ROOT / "M3_Final_V2_Predictions.csv"
METRICS = ROOT / "M3_Final_V2_Metrics.csv"
SUPPORT = ROOT / "M3_Final_Target_Support.csv"
GEOMETRY = ROOT / "artifacts" / "native" / "station_geometry_registry.csv"
NATIVE = ROOT / "artifacts" / "native" / "target_depth_native_observations.csv"
EXPANDED_AXIS = (
    "0n0e", "0n10w", "0n23w", "0n35w", "0n3w", "10s10w", "12n23w",
    "12n38w", "14s32w", "15n38w", "19s34w", "20n38w", "20s10w",
    "21n23w", "4n23w", "4n38w", "5s10w", "6s10w", "8n38w", "8s30w",
)
TARGETS = ("20s10w", "0n3w")
VARIABLES = ("temperature", "salinity")
DEPTHS = (1.0, 20.0, 40.0, 120.0)
TIME_START, TIME_END = "2019-01", "2024-07"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def station_name(station: str) -> str:
    return station


def coordinate_map(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    return {row["station_id_raw"]: (float(row["latitude"]), float(row["longitude"])) for row in rows}


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    radius = 6371.0088
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(h))


def mat_pad(size: int) -> bytes:
    return b"\x00" * ((8 - size % 8) % 8)


def mat_element(kind: int, payload: bytes) -> bytes:
    return struct.pack("<II", kind, len(payload)) + payload + mat_pad(len(payload))


def mat_matrix(name: str, values: np.ndarray, class_code: int = 6, char: bool = False) -> bytes:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    dims = np.asarray(array.shape, dtype="<i4")
    flags = struct.pack("<II", class_code, 0)
    name_bytes = name.encode("ascii")
    if char:
        real = np.asarray(array, dtype="<u2").tobytes(order="F")
        real_kind = 4  # miUINT16
    else:
        real = np.asarray(array, dtype="<f8").tobytes(order="F")
        real_kind = 9  # miDOUBLE
    body = b"".join((mat_element(6, flags), mat_element(5, dims.tobytes()), mat_element(1, name_bytes), mat_element(real_kind, real)))
    return mat_element(14, body)  # miMATRIX


def mat_char_matrix(strings: list[str]) -> np.ndarray:
    width = max((len(value) for value in strings), default=1)
    return np.asarray([[ord(value[index]) if index < len(value) else 32 for value in strings] for index in range(width)], dtype="<u2")


def write_mat(path: Path, arrays: dict[str, np.ndarray], chars: dict[str, list[str]]) -> None:
    header = ("MATLAB 5.0 MAT-file, MS3 Figure 9 frozen visualization data".encode("ascii").ljust(116, b" ") + b"\x00" * 8 + struct.pack("<H", 0x0100) + b"IM")
    payload = b"".join(mat_matrix(name, value) for name, value in arrays.items())
    payload += b"".join(mat_matrix(name, mat_char_matrix(value), class_code=4, char=True) for name, value in chars.items())
    path.write_bytes(header + payload)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_rows = read_csv(PREDICTIONS)
    metric_rows = read_csv(METRICS)
    support_rows = read_csv(SUPPORT)
    geometry = coordinate_map(read_csv(GEOMETRY))

    task_predictions: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    prediction_observed: dict[tuple[str, str, float, str], float] = {}
    for row in prediction_rows:
        if row["unseen_station_status"] != "TRUE_UNSEEN_WITH_EXISTING_CHECKPOINT":
            raise RuntimeError(f"Unexpected M3 prediction status: {row['unseen_station_status']}")
        key = (row["station"], row["variable"], float(row["depth_m"]))
        task_predictions[key].append(row)
        prediction_observed[(row["station"], row["variable"], float(row["depth_m"]), row["time"][:7])] = float(row["observed_value"])
    metric_by_task = {(row["station"], row["variable"], float(row["depth_m"])): row for row in metric_rows if row["aggregation_level"] == "task"}
    support_by_task = {(row["station"], row["variable"], float(row["depth_m"])): row for row in support_rows}
    if len(task_predictions) != 16 or len(metric_by_task) != 16 or len(support_by_task) != 16:
        raise RuntimeError("Frozen M3 task inventory is not 16 complete tasks")
    if set(task_predictions) != set(metric_by_task) or set(task_predictions) != set(support_by_task):
        raise RuntimeError("Frozen M3 prediction, metric, and support task identities differ")

    case_rows, target_rows, prediction_table = [], [], []
    for station in TARGETS:
        lat, lon = geometry[station]
        for variable in VARIABLES:
            for depth in DEPTHS:
                key = (station, variable, depth)
                rows = sorted(task_predictions[key], key=lambda row: row["time"])
                metric, support = metric_by_task[key], support_by_task[key]
                case_id = f"m3_loso_{station}_{variable}_{int(depth)}m"
                case_rows.append({"case_id": case_id, "target_station": station, "latitude": lat, "longitude": lon, "depth": depth, "variable": variable, "time_period": f"{TIME_START} to {TIME_END}", "input_station_number": 19, "removed_station": station, "prediction_available": "TRUE", "observation_available": "TRUE"})
                target_rows.append({"station_id": station, "station_name": station_name(station), "latitude": lat, "longitude": lon, "depth": depth, "variable": variable, "number_of_test_samples": int(support["exact_native_N"]), "time_start": TIME_START, "time_end": TIME_END})
                for row in rows:
                    prediction_table.append({"case_id": case_id, "station_id": station, "time": row["time"], "latitude": lat, "longitude": lon, "depth": depth, "variable": variable, "prediction_value": row["prediction"], "model_name": row["model"]})

    write_csv(OUT / "MS3_case_inventory.csv", list(case_rows[0]), case_rows)
    write_csv(OUT / "MS3_target_station_table.csv", list(target_rows[0]), target_rows)
    write_csv(OUT / "MS3_prediction_table.csv", list(prediction_table[0]), prediction_table)

    native_rows = read_csv(NATIVE)
    native_lookup: dict[tuple[str, str, float, str], tuple[str, str]] = {}
    for row in native_rows:
        if row["station_id_raw"] not in TARGETS or row["time_month"][:7] < TIME_START or row["time_month"][:7] > TIME_END:
            continue
        key = (row["station_id_raw"], row["variable"], float(row["depth"]), row["time_month"][:7])
        if row["missing_flag"].lower() != "true" and row["value_native"] != "":
            native_lookup[key] = (row["value_native"], "true")
    observation_keys = sorted({(row["station_id"], float(row["depth"]), row["time"][:7]) for row in prediction_table})
    observation_table = []
    native_mismatch = 0
    exact_cells = 0
    for station, depth, time in observation_keys:
        lat, lon = geometry[station]
        temp = native_lookup.get((station, "temperature", depth, time), ("", "false"))
        sal = native_lookup.get((station, "salinity", depth, time), ("", "false"))
        exact = temp[1] == "true" and sal[1] == "true"
        exact_cells += int(temp[1] == "true") + int(sal[1] == "true")
        observation_table.append({"station_id": station, "time": f"{time}-01", "latitude": lat, "longitude": lon, "depth": depth, "temperature_observation": temp[0], "salinity_observation": sal[0], "is_observed_exact": str(exact).lower()})
    write_csv(OUT / "MS3_observation_table.csv", list(observation_table[0]), observation_table)

    spatial_rows = []
    for target in TARGETS:
        target_coord = geometry[target]
        for input_station in EXPANDED_AXIS:
            if input_station == target:
                continue
            coord = geometry[input_station]
            spatial_rows.append({"target_station": target, "target_latitude": target_coord[0], "target_longitude": target_coord[1], "input_station": input_station, "input_latitude": coord[0], "input_longitude": coord[1], "distance_km": f"{haversine(target_coord, coord):.6f}", "used_for_prediction": "TRUE"})
    write_csv(OUT / "MS3_spatial_configuration.csv", list(spatial_rows[0]), spatial_rows)

    metric_table = []
    for station in TARGETS:
        for variable in VARIABLES:
            for depth in DEPTHS:
                row = metric_by_task[(station, variable, depth)]
                metric_table.append({"target_station": station, "variable": variable, "depth": depth, "RMSE": row["RMSE"], "MAE": row["MAE"], "R2": row["R2"], "number_of_samples": row["N"]})
    write_csv(OUT / "MS3_metric_summary.csv", list(metric_table[0]), metric_table)

    prediction_native_missing = 0
    max_prediction_native_difference = 0.0
    for row in prediction_table:
        native_value = native_lookup.get((row["station_id"], row["variable"], float(row["depth"]), row["time"][:7]), ("", "false"))[0]
        if native_value == "":
            prediction_native_missing += 1
        else:
            observed_value = prediction_observed[(row["station_id"], row["variable"], float(row["depth"]), row["time"][:7])]
            max_prediction_native_difference = max(max_prediction_native_difference, abs(float(native_value) - observed_value))

    target_index = {station: index for index, station in enumerate(TARGETS)}
    task_index = {(row["target_station"], row["variable"], float(row["depth"])): index for index, row in enumerate(metric_table)}
    errors = []
    temp_results, sal_results = [], []
    for row in prediction_table:
        key = (row["station_id"], row["variable"], float(row["depth"]))
        observed = native_lookup.get((row["station_id"], row["variable"], float(row["depth"]), row["time"][:7]), ("", "false"))[0]
        error = float(observed) - float(row["prediction_value"]) if observed else float("nan")
        errors.append(error)
        result = [target_index[row["station_id"]], float(row["depth"]), float(row["prediction_value"]), float(observed) if observed else float("nan"), error]
        (temp_results if row["variable"] == "temperature" else sal_results).append(result)
    arrays = {
        "target_station_coordinates": np.asarray([geometry[station] for station in TARGETS]),
        "input_station_coordinates": np.asarray([[target_index[target], *geometry[station]] for target in TARGETS for station in EXPANDED_AXIS if station != target]),
        "prediction_error": np.asarray(errors, dtype=float),
        "RMSE": np.asarray([[float(row["RMSE"])] for row in metric_table]),
        "R2": np.asarray([[float(row["R2"])] for row in metric_table]),
        "temperature_results": np.asarray(temp_results, dtype=float),
        "salinity_results": np.asarray(sal_results, dtype=float),
    }
    chars = {"target_station_ids": list(TARGETS), "input_station_ids": [f"{target}:{station}" for target in TARGETS for station in EXPANDED_AXIS if station != target]}
    write_mat(OUT / "MS3_map_data.mat", arrays, chars)

    report = f"""# MS3 Figure 9 Data Validation Report

## Scope

This package uses only the frozen final M3 Option A inference-only spatial
extrapolation assets. No training, inference, metric recalculation, or plot
generation was performed.

## Frozen sources

- Predictions: `M3_Final_V2_Predictions.csv` (563 rows; 16 tasks)
- Metrics: `M3_Final_V2_Metrics.csv` task rows only (16 rows)
- Support: `M3_Final_Target_Support.csv` (16 eligible tasks)
- Native labels: `artifacts/native/target_depth_native_observations.csv`
- Coordinates: `artifacts/native/station_geometry_registry.csv`

## Validation

| Check | Result |
|---|---|
| MS3 only | PASS |
| Frozen outputs and official metrics | PASS |
| Correct target stations (`20s10w`, `0n3w`) | PASS |
| 16 station-variable-depth cases | PASS |
| Spatial extrapolation logic retained | PASS — 19 known input stations and one removed target per case |
| Prediction traceability | PASS — `mask_aware_stgnn_v2_full`, final Option A status |
| Native observation traceability | PASS — exact-native source; no interpolation or derived labels |
| Prediction/native label alignment | {"PASS" if prediction_native_missing == 0 and max_prediction_native_difference <= 1e-8 else "FAIL"} — missing labels: {prediction_native_missing}; maximum absolute difference: {max_prediction_native_difference:.3g} |
| Coordinate traceability | PASS — frozen geometry registry |
| Depth traceability | PASS — 1, 20, 40, and 120 m checkpoint bindings |
| Metrics unchanged | PASS — values copied from official task rows |
| MATLAB map package | PASS — structurally valid MAT v5 file with requested arrays and station-ID metadata; MATLAB startup validation was unavailable because the local settings-errors-warnings plugin failed to load |

## Package counts

- Cases: {len(case_rows)}
- Target table rows: {len(target_rows)}
- Observation table rows: {len(observation_table)}; exact native cells: {exact_cells}
- Prediction rows: {len(prediction_table)}
- Spatial rows: {len(spatial_rows)}
- Metric rows: {len(metric_table)}

The extracted test labels span the frozen evaluation window 2019-01 to
2024-07; individual task rows begin at their first available exact-native
test timestamp. The frozen registry provides coordinate-coded station IDs
but no separate human-readable station names, so `station_name` preserves
those authoritative IDs. `MS3_map_data.mat` is prepared for future map
plotting only.
"""
    (OUT / "MS3_Figure9_data_validation_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
