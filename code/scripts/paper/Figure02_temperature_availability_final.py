"""Render manuscript Figure 2: PIRATA temperature observation availability.

This figure is a data-quality figure, not a model-evaluation figure.  The
script reads the authoritative observation-level target-depth CSV only.  It
constructs a complete monthly index from 1998-02 through 2024-07 so absent
months remain NaN; no interpolation, smoothing, reconstruction, or model
dataset is used.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "artifacts" / "native" / "target_depth_native_observations.csv"
GEOMETRY_PATH = ROOT / "artifacts" / "native" / "station_geometry_registry.csv"
MANIFEST_PATH = ROOT / "artifacts" / "algorithm" / "v21_station_extension" / "station_extension_manifest.json"
SELECTION_PATH = ROOT / "artifacts" / "paper" / "figure_data" / "Figure02_station_selection.csv"
PNG_PATH = ROOT / "artifacts" / "paper" / "figures" / "Figure02_temperature_availability_final.png"
PDF_PATH = ROOT / "artifacts" / "paper" / "figures" / "Figure02_temperature_availability_final.pdf"
REPORT_PATH = ROOT / "artifacts" / "paper" / "figure_design" / "Figure02_design_report.md"

START = pd.Timestamp("1998-02-01")
END = pd.Timestamp("2024-07-01")
MONTHS = pd.date_range(START, END, freq="MS")
DEPTHS = (1.0, 20.0, 40.0)
Y_LIMITS = (20.0, 30.0)
Y_TICKS = (20, 22, 24, 26, 28, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_observations() -> pd.DataFrame:
    columns = [
        "variable",
        "station_id_raw",
        "depth",
        "year_raw",
        "month_raw",
        "missing_flag",
        "value_native",
    ]
    data = pd.read_csv(SOURCE_PATH, usecols=columns)
    data = data[data["variable"].eq("temperature") & data["depth"].isin(DEPTHS)].copy()
    data["time"] = pd.to_datetime(
        {"year": data["year_raw"], "month": data["month_raw"], "day": 1},
        errors="raise",
    )
    data = data[data["time"].between(START, END)].copy()
    missing = data["missing_flag"].astype("string").str.lower().isin(["true", "1", "yes"])
    data["available"] = (~missing) & data["value_native"].notna()
    key = ["station_id_raw", "depth", "time"]
    if data.duplicated(key).any():
        raise ValueError("duplicate station/depth/month rows in authoritative observations")
    if data.empty:
        raise ValueError("no temperature observations remain after the required filters")
    return data


def read_station_inventory() -> tuple[list[str], pd.DataFrame]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    intersection = list(manifest["development_axis"]["stations"])
    geometry = pd.read_csv(GEOMETRY_PATH)
    geometry = geometry.rename(
        columns={"station_id_raw": "station_id", "latitude": "latitude", "longitude": "longitude"}
    )
    geometry["station_id"] = geometry["station_id"].astype(str)
    if not set(intersection).issubset(set(geometry["station_id"])):
        raise ValueError("development-axis station is absent from the authoritative geometry registry")
    return intersection, geometry.set_index("station_id")


def availability_table(data: pd.DataFrame, stations: list[str], geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station in stations:
        row: dict[str, object] = {
            "station_id": station,
            "longitude": float(geometry.loc[station, "longitude"]),
            "latitude": float(geometry.loc[station, "latitude"]),
        }
        all_available = 0
        for depth in DEPTHS:
            subset = data[(data["station_id_raw"].eq(station)) & data["depth"].eq(depth)]
            series = subset.set_index("time")["available"].reindex(MONTHS, fill_value=False)
            valid = int(series.sum())
            row[f"valid_months_{int(depth)}m"] = valid
            row[f"missing_months_{int(depth)}m"] = int(len(MONTHS) - valid)
            all_available += valid
        row["missing_rate"] = 1.0 - all_available / (len(MONTHS) * len(DEPTHS))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_stations(table: pd.DataFrame) -> pd.DataFrame:
    valid_columns = [f"valid_months_{int(depth)}m" for depth in DEPTHS]
    eligible = table[table[valid_columns].gt(0).all(axis=1)].copy()
    if len(eligible) < 3:
        raise ValueError("fewer than three intersection stations have observations at all target depths")

    high = eligible.sort_values(["missing_rate", "station_id"], ascending=[True, True]).iloc[0]["station_id"]
    remaining = eligible[eligible["station_id"].ne(high)].copy()
    gap = remaining.sort_values(["missing_rate", "station_id"], ascending=[False, True]).iloc[0]["station_id"]
    remaining = remaining[remaining["station_id"].ne(gap)].copy()
    median_rate = float(remaining["missing_rate"].median())
    typical = (
        remaining.assign(distance=(remaining["missing_rate"] - median_rate).abs())
        .sort_values(["distance", "station_id"], ascending=[True, True])
        .iloc[0]["station_id"]
    )

    selected = table[table["station_id"].isin([high, gap, typical])].copy()
    reason = {
        high: "High availability: lowest aggregate missing rate among intersection stations with valid 1/20/40 m observations.",
        gap: "Obvious gaps: highest aggregate missing rate among the remaining eligible stations; all three target depths remain represented.",
        typical: "Typical station: aggregate missing rate closest to the median of the remaining eligible stations.",
    }
    selected["selection_reason"] = selected["station_id"].map(reason)
    order = {high: 1, gap: 2, typical: 3}
    selected["selection_order"] = selected["station_id"].map(order)
    return selected.sort_values("selection_order").drop(columns=["selection_order"])


def write_selection_table(selected: pd.DataFrame) -> None:
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "station_id",
        "longitude",
        "latitude",
        "valid_months_1m",
        "valid_months_20m",
        "valid_months_40m",
        "missing_rate",
        "selection_reason",
    ]
    selected[columns].to_csv(SELECTION_PATH, index=False, float_format="%.6f")


def station_label(row: pd.Series) -> str:
    lat = abs(float(row["latitude"]))
    lon = abs(float(row["longitude"]))
    # Use the conventional north/east suffix at the equator/prime meridian
    # so labels remain in the requested station-coordinate style (e.g. 0°N).
    lat_suffix = "N" if float(row["latitude"]) >= 0 else "S"
    lon_suffix = "E" if float(row["longitude"]) >= 0 else "W"
    return f"{lat:g}°{lat_suffix}, {lon:g}°{lon_suffix}"


def monthly_series(data: pd.DataFrame, station: str, depth: float) -> pd.Series:
    subset = data[(data["station_id_raw"].eq(station)) & data["depth"].eq(depth)].copy()
    subset["plot_value"] = subset["value_native"].where(subset["available"])
    return subset.set_index("time")["plot_value"].reindex(MONTHS)


def render(selected: pd.DataFrame, data: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True, squeeze=False)
    tick_years = (1998, 2003, 2008, 2013, 2018, 2024)
    tick_dates = [pd.Timestamp(f"{year}-01-01") for year in tick_years]
    for row_index, depth in enumerate(DEPTHS):
        for col_index, (_, station_row) in enumerate(selected.iterrows()):
            station = str(station_row["station_id"])
            ax = axes[row_index, col_index]
            values = monthly_series(data, station, depth)
            ax.plot(
                MONTHS,
                values.to_numpy(dtype=float),
                color="#1559a6",
                linewidth=1.2,
                marker="o",
                markersize=1.8,
                markerfacecolor="#1559a6",
                markeredgewidth=0,
                solid_capstyle="round",
            )
            ax.set_ylim(*Y_LIMITS)
            ax.set_yticks(Y_TICKS)
            ax.set_xlim(START, END)
            ax.set_xticks(tick_dates)
            ax.set_xticklabels([str(year) for year in tick_years])
            ax.grid(axis="y", color="#b8c2cc", linewidth=0.55, alpha=0.65)
            ax.grid(axis="x", color="#d4dbe2", linewidth=0.45, alpha=0.55)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(station_label(station_row), pad=8, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(f"{int(depth)} m\nTemperature (°C)")
            else:
                ax.tick_params(labelleft=False)
            if row_index < len(DEPTHS) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(axis="x", rotation=0)
            for spine in ax.spines.values():
                spine.set_color("#58636e")
                spine.set_linewidth(0.7)

    legend_handle = Line2D(
        [0],
        [0],
        color="#1559a6",
        linewidth=1.2,
        marker="o",
        markersize=3.2,
        markerfacecolor="#1559a6",
        markeredgewidth=0,
        label="Observed temperature",
    )
    fig.legend(handles=[legend_handle], loc="lower center", bbox_to_anchor=(0.5, 0.015), frameon=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.105, top=0.935, hspace=0.18, wspace=0.08)
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(selected: pd.DataFrame, data: pd.DataFrame) -> None:
    selected_lines = []
    for _, row in selected.iterrows():
        selected_lines.append(
            "- **{station}** ({label}; longitude {lon:.1f}°, latitude {lat:.1f}°): "
            "1 m = {v1}, 20 m = {v20}, 40 m = {v40} valid months; aggregate missing rate = {rate:.3%}. {reason}".format(
                station=row["station_id"],
                label=station_label(row),
                lon=row["longitude"],
                lat=row["latitude"],
                v1=row["valid_months_1m"],
                v20=row["valid_months_20m"],
                v40=row["valid_months_40m"],
                rate=row["missing_rate"],
                reason=row["selection_reason"],
            )
        )
    report = f"""# Figure 2 Design Report — PIRATA Temperature Observation Availability

## Scope

Figure 2 is a data-quality figure showing original PIRATA temperature
observation continuity and missing months. It is not a reconstruction,
prediction, or model-evaluation figure.

## Source and observation definition

- Observation-level source: `{SOURCE_PATH}`
- Coordinate source: `{GEOMETRY_PATH}`
- Station-axis source: `{MANIFEST_PATH}`
- Source SHA-256: `{sha256(SOURCE_PATH)}`
- Variable: `temperature` only
- Unit: °C (native observation values)
- Period: 1998-02 through 2024-07 inclusive ({len(MONTHS)} expected months)
- Target depths: exactly 1 m, 20 m, and 40 m; 120 m excluded from the main figure
- Availability rule: a month is available only when an exact target-depth row has
  `missing_flag=False` and a non-missing native value. Months absent from the
  observation product remain missing after reindexing to the complete monthly grid.
- No interpolation, extrapolation, smoothing, gap filling, normalization,
  reconstruction, or model/training/validation/test data was used.

## Station selection

The authoritative 18-station temperature–salinity intersection was read from
the station-extension manifest. Candidate stations were required to have at
least one valid exact-depth observation at each of 1 m, 20 m, and 40 m. The
station `5s10w` was therefore not eligible because its 20 m temperature count
is zero. The selection table was generated before plotting:

`{SELECTION_PATH}`

{chr(10).join(selected_lines)}

Aggregate missing rate is calculated as:

`1 - (valid_months_1m + valid_months_20m + valid_months_40m) / ({len(MONTHS)} × 3)`

## Output

- Python script: `{__file__}`
- 600 dpi PNG: `{PNG_PATH}`
- Vector PDF: `{PDF_PATH}`

## Visual specification

- Layout unchanged: 3 × 3 panels, 12 × 9 inches, shared axes, and station order preserved
- Shared y-axis limits: 20–30 °C
- Shared y-axis ticks: 20, 22, 24, 26, 28, 30 °C
- Observed temperature line: blue, linewidth 1.2; monthly observation markers retained at size 1.8

## Quality-control checklist

- Python only: PASS
- 3 × 3 panels: PASS
- Depths exactly 1/20/40 m: PASS
- Temperature only: PASS
- Observation-level source only: PASS
- No interpolation or reconstruction: PASS
- Identical y-axis range (20–30 °C): PASS
- Y-axis ticks exactly 20, 22, 24, 26, 28, 30: PASS
- Station coordinates checked against authoritative registry: PASS
- Monthly availability calculated on the complete 1998-02 to 2024-07 grid: PASS
- Global legend only: PASS
- 120 m excluded from main figure: PASS

Runtime used: Python {platform.python_version()}, pandas {pd.__version__},
NumPy {np.__version__}, Matplotlib {matplotlib.__version__}.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    data = read_observations()
    stations, geometry = read_station_inventory()
    table = availability_table(data, stations, geometry)
    selected = choose_stations(table)
    write_selection_table(selected)
    render(selected, data)
    write_report(selected, data)
    print(f"Selected stations: {', '.join(selected['station_id'])}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SELECTION_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
