"""Prepare a traceable, non-rendering Figure 9 data package.

Observed values are read from the native PIRATA target-depth source; model
values are copied from frozen final two-station M3 Option-A predictions.  No
model is loaded and no scientific asset is altered.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "artifacts" / "paper" / "figures" / "Figure9_MS3"
PACKAGE = FIGURE_ROOT / "Figure9_MS3_final_visualization"
RAW_NATIVE = ROOT / "artifacts" / "native" / "target_depth_native_observations.csv"
FROZEN_PREDICTIONS = ROOT / "M3_Final_V2_Predictions.csv"
CURRENT_OBSERVATIONS = FIGURE_ROOT / "MS3_observation_table.csv"
CURRENT_PREDICTIONS = FIGURE_ROOT / "MS3_prediction_table.csv"
CURRENT_FINAL_SCRIPT = FIGURE_ROOT / "Figure9_MS3_BC_final_plot.py"
HISTORICAL_CASE_METADATA = ROOT / "artifacts" / "m3_case_visualization" / "m3_new_station_case_metadata.yaml"
AUDIT = FIGURE_ROOT / "Figure9_visualization_source_audit.md"
OUTPUTS = (
    PACKAGE / "Figure9_observed_visualization.csv",
    PACKAGE / "Figure9_prediction_visualization.csv",
    PACKAGE / "Figure9_visualization_metadata.yaml",
    PACKAGE / "Figure9_data_source_report.md",
    AUDIT,
)
STATIONS = ("20s10w", "0n3w")
SERIES = (("temperature", 20.0), ("salinity", 40.0))
START = (2019, 1)
END = (2024, 7)


class PreparationError(RuntimeError):
    """Raised when frozen Figure 9 sources fail traceability checks."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def month_sequence() -> list[str]:
    values: list[str] = []
    year, month = START
    while (year, month) <= END:
        values.append(f"{year:04d}-{month:02d}-01")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return values


def to_month(value: str) -> str:
    return value[:7] + "-01"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def main() -> int:
    existing = [str(path.relative_to(ROOT)) for path in OUTPUTS if path.exists()]
    if existing:
        raise PreparationError("refusing to overwrite output(s): " + ", ".join(existing))
    for source in (RAW_NATIVE, FROZEN_PREDICTIONS, CURRENT_OBSERVATIONS, CURRENT_PREDICTIONS, CURRENT_FINAL_SCRIPT, HISTORICAL_CASE_METADATA):
        if not source.is_file():
            raise PreparationError(f"missing source: {source.relative_to(ROOT)}")
    PACKAGE.mkdir(parents=True, exist_ok=False)

    months = month_sequence()
    raw_index: dict[tuple[str, str, float, str], dict[str, str]] = {}
    raw_record_months: dict[tuple[str, str, float], set[str]] = {}
    raw_flag_missing_months: dict[tuple[str, str, float], set[str]] = {}
    for row in read_csv(RAW_NATIVE):
        station, variable = row["station_id_raw"], row["variable"]
        depth = float(row["depth"])
        time = to_month(row["time_month"])
        if station not in STATIONS or (variable, depth) not in SERIES or time not in months:
            continue
        key = (station, variable, depth, time)
        if key in raw_index:
            raise PreparationError(f"duplicate raw native key: {key}")
        raw_index[key] = row
        raw_record_months.setdefault((station, variable, depth), set()).add(time)
        if row["missing_flag"].strip().lower() not in {"false", "0"} or not row["value_native"].strip():
            raw_flag_missing_months.setdefault((station, variable, depth), set()).add(time)

    observed_rows: list[dict[str, str]] = []
    counts: dict[tuple[str, str, float], dict[str, int | str]] = {}
    for station in STATIONS:
        for variable, depth in SERIES:
            key = (station, variable, depth)
            available = 0
            for time in months:
                raw = raw_index.get((station, variable, depth, time))
                is_available = raw is not None and raw["missing_flag"].strip().lower() in {"false", "0"} and bool(raw["value_native"].strip())
                if is_available:
                    available += 1
                observed_rows.append({
                    "station": station,
                    "time": time,
                    "depth": f"{depth:g}",
                    "variable": variable,
                    "observed_value": raw["value_native"] if is_available else "",
                    "observation_available": "TRUE" if is_available else "FALSE",
                })
            records = raw_record_months.get(key, set())
            explicit_missing = raw_flag_missing_months.get(key, set())
            counts[key] = {"available": available, "native_records": len(records), "flag_missing": len(explicit_missing), "no_record": len(months) - len(records)}

    frozen_prediction_rows: list[dict[str, str]] = []
    prediction_keys: set[tuple[str, str, float, str]] = set()
    for row in read_csv(FROZEN_PREDICTIONS):
        station, variable, depth, time = row["station"], row["variable"], float(row["depth_m"]), to_month(row["time"])
        if station not in STATIONS or (variable, depth) not in SERIES:
            continue
        if time not in months:
            raise PreparationError(f"frozen prediction outside Figure 9 window: {station}, {variable}, {depth:g}, {time}")
        key = (station, variable, depth, time)
        if key in prediction_keys:
            raise PreparationError(f"duplicate frozen prediction key: {key}")
        raw = raw_index.get(key)
        if raw is None or raw["missing_flag"].strip().lower() not in {"false", "0"} or not raw["value_native"].strip():
            raise PreparationError(f"frozen prediction lacks matching finite native source observation: {key}")
        prediction_keys.add(key)
        frozen_prediction_rows.append({
            "station": station,
            "time": time,
            "depth": f"{depth:g}",
            "variable": variable,
            "prediction_value": row["prediction"],
        })
    expected_prediction_count = sum(int(counts[(station, variable, depth)]["available"]) for station in STATIONS for variable, depth in SERIES)
    if len(frozen_prediction_rows) != expected_prediction_count:
        raise PreparationError(f"frozen prediction count {len(frozen_prediction_rows)} does not match selected exact-native support {expected_prediction_count}")

    # Audit the present Figure 9 convenience tables without using them as the
    # new authoritative observed source.
    current_obs_counts: dict[tuple[str, str, float], int] = {}
    for row in read_csv(CURRENT_OBSERVATIONS):
        station, depth, time = row["station_id"], float(row["depth"]), to_month(row["time"])
        if station not in STATIONS or time not in months:
            continue
        for variable, field in (("temperature", "temperature_observation"), ("salinity", "salinity_observation")):
            if (variable, depth) in SERIES and row[field].strip():
                current_obs_counts[(station, variable, depth)] = current_obs_counts.get((station, variable, depth), 0) + 1
    current_pred_counts: dict[tuple[str, str, float], int] = {}
    for row in read_csv(CURRENT_PREDICTIONS):
        station, variable, depth, time = row["station_id"], row["variable"], float(row["depth"]), to_month(row["time"])
        if station in STATIONS and (variable, depth) in SERIES and time in months:
            current_pred_counts[(station, variable, depth)] = current_pred_counts.get((station, variable, depth), 0) + 1
    for key, detail in counts.items():
        if current_obs_counts.get(key, 0) != detail["available"] or current_pred_counts.get(key, 0) != detail["available"]:
            raise PreparationError(f"current Figure 9 convenience table count differs from frozen native/prediction support: {key}")

    write_csv(PACKAGE / "Figure9_observed_visualization.csv", ["station", "time", "depth", "variable", "observed_value", "observation_available"], observed_rows)
    write_csv(PACKAGE / "Figure9_prediction_visualization.csv", ["station", "time", "depth", "variable", "prediction_value"], sorted(frozen_prediction_rows, key=lambda item: (item["station"], item["variable"], float(item["depth"]), item["time"])))

    metadata_lines = [
        "dataset_id: figure9_ms3_final_visualization_v1",
        "scope: selected_independent_pirata_station_spatial_transfer_supplementary_cases",
        "stations:", "  - 20s10w", "  - 0n3w",
        "time_range: 2019-01 to 2024-07",
        "series:",
        "  - variable: temperature", "    depth_m: 20", "  - variable: salinity", "    depth_m: 40",
        "observed_source:", f"  path: {RAW_NATIVE.relative_to(ROOT).as_posix()}", f"  sha256: {sha256(RAW_NATIVE)}",
        "  rule: direct native PIRATA values; a 67-month calendar grid preserves missing records and native missing flags as observation_available FALSE",
        "prediction_source:", f"  path: {FROZEN_PREDICTIONS.name}", f"  sha256: {sha256(FROZEN_PREDICTIONS)}",
        "  rule: exact copy of frozen final two-station M3 Option-A predictions at their existing timestamps only; no new inference",
        "forbidden_operations:", "  - interpolation", "  - smoothing", "  - gap filling", "  - anomaly conversion", "  - WOA23 model input", "  - prediction regeneration",
        "coverage:",
    ]
    for station in STATIONS:
        for variable, depth in SERIES:
            key, detail = (station, variable, depth), counts[(station, variable, depth)]
            metadata_lines += [f"  - station: {station}", f"    variable: {variable}", f"    depth_m: {depth:g}", f"    native_observation_available_months: {detail['available']}", f"    frozen_prediction_months: {detail['available']}", f"    native_no_record_months: {detail['no_record']}", f"    explicit_native_missing_months: {detail['flag_missing']}"]
    write_text(PACKAGE / "Figure9_visualization_metadata.yaml", "\n".join(metadata_lines) + "\n")

    table = ["| Station | Variable | Depth (m) | Calendar months | Native available | Native no-record | Explicit native missing | Frozen predictions |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for station in STATIONS:
        for variable, depth in SERIES:
            detail = counts[(station, variable, depth)]
            table.append(f"| {station} | {variable} | {depth:g} | {len(months)} | {detail['available']} | {detail['no_record']} | {detail['flag_missing']} | {detail['available']} |")
    data_report = """# Figure 9 final visualization data-source report

## Scope

This package is restricted to the frozen two-station M3 Option-A cases `20s10w` and `0n3w`, temperature 20 m and salinity 40 m, from 2019-01 through 2024-07. It does not include four-station LOSO-reset evidence.

## Authoritative plotting flow

- **Observed black curve:** `artifacts/native/target_depth_native_observations.csv`, the direct native PIRATA target-depth table. Every calendar month is represented in the observed CSV; unavailable months have a blank value and `observation_available=FALSE`.
- **Model reconstruction curve:** `M3_Final_V2_Predictions.csv`, copied exactly at frozen prediction timestamps. It is an exact-native evaluation-support prediction product, not a continuous full-period reconstruction output.

## Coverage

""" + "\n".join(table) + """

## Drawing rule for the next plotting stage

Draw observations only where `observation_available=TRUE`, breaking the line at every unavailable month. Draw the model only where a frozen prediction record exists. Do not bridge, interpolate, smooth, fill, normalize, transform to anomaly, or use WOA23 as a model input.

## Readiness

The package is ready for a later MATLAB Panel A and Python Panels B/C implementation. It contains data only; no Figure 9 is created by this task.
"""
    write_text(PACKAGE / "Figure9_data_source_report.md", data_report)

    source_audit = """# Figure 9 visualization source audit

## Scope and identity

This audit concerns only the frozen final two-station M3 Option-A evidence: `20s10w`, `0n3w`, temperature 20 m, and salinity 40 m, over 2019-01 to 2024-07. Four-station MS3 LOSO-reset assets are explicitly excluded.

## Located historical Figure 9 data flow

The identifiable current Figure 9 B/C scripts are `Figure9_MS3_BC_absolute_validation_plot.py` and `Figure9_MS3_BC_final_plot.py`. They read `MS3_observation_table.csv` and `MS3_prediction_table.csv`. The latter are convenience extracts built from the frozen final M3 prediction/label support, so their black observation curve is an exact-native **evaluation subset**, not an independently declared full native-observation visualization source.

The older `artifacts/m3_case_visualization/` package is tied to the separate static-covariate M3-primary prediction tree. It is historical-only under the M3 asset review and is not used by this package.

## Comparison of old convenience data and native source

For the four selected series, the current convenience tables contain the same number of finite exact-native observations and frozen predictions as the frozen final support. They do not contain additional months beyond the native source. The observed source is nevertheless replaced here with a calendar-indexed extract of the direct native table so that true absence is explicit and independent of metric-table construction.

""" + "\n".join(table) + """

## Answers to the required questions

1. **Why could an older Figure 9 appear to have a more complete observed curve?** No repository-traceable final Figure 9 script/data product was found that contains additional authoritative native observations for these four selected series. The identifiable final scripts use the same exact-native support counts above. Any visually fuller unregistered preview cannot be attributed safely and must not be reused as evidence.
2. **Why does the current version show large year gaps?** The direct native source itself has unavailable months: both missing source records and explicit native-missing flags. For example, `0n3w` has only 28 finite native months at each selected depth in the 67-month window. Those gaps are real data availability limits and must remain visible.
3. **Was a metric evaluation subset used as a visualization dataset?** Yes. `MS3_observation_table.csv` is an exact-native, prediction-aligned extraction and was used by the prior B/C scripts. Its values match the frozen support, but it is not the required authoritative full native-observation data flow for plotting.
4. **Which files are correct for final plotting?** Use `artifacts/native/target_depth_native_observations.csv` for observations and `M3_Final_V2_Predictions.csv` for predictions, through the two derived files in `Figure9_MS3_final_visualization/`.

## Scientific boundary

The frozen model does not supply a continuous prediction at every calendar month. This task therefore preserves prediction gaps rather than manufacturing a continuous model line. No experiment asset, checkpoint, prediction, metric, mask, or Figure 9 was modified.
"""
    write_text(AUDIT, source_audit)
    print(f"PASS: wrote {len(observed_rows)} calendar-indexed observed rows and {len(frozen_prediction_rows)} frozen prediction rows")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as error:
        print(f"BLOCKED: {error}")
        raise SystemExit(2)
