"""Render the paper Figure 2 observation-coverage and missing-pattern panels.

Only the three approved native coverage summaries are read.  The renderer uses
their recorded finite and missing counts directly; it never opens observation
values or experiment outputs.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


MONTHLY_PATH = Path("artifacts/native/monthly_coverage.csv")
DEPTH_PATH = Path("artifacts/native/depth_coverage.csv")
STATION_DEPTH_VARIABLE_PATH = Path(
    "artifacts/native/station_depth_variable_coverage.csv"
)
FIGURE_DIRECTORY = Path("artifacts/paper/figures")
TARGET_DEPTHS = (1.0, 20.0, 40.0, 120.0)
START_MONTH = datetime(1998, 2, 1)
END_MONTH = datetime(2024, 7, 1)

MONTHLY_FIELDS = ["time_month", "record_count", "finite_count", "missing_count"]
DEPTH_FIELDS = ["depth", "record_count", "finite_count", "missing_count"]
STATION_FIELDS = [
    "station_id_raw",
    "depth",
    "temperature_count",
    "salinity_count",
    "temperature_missing_count",
    "salinity_missing_count",
]

TEMPERATURE = colors.HexColor("#C65A28")
SALINITY = colors.HexColor("#2A6F9B")
MISSING = colors.HexColor("#CBD5DC")
INK = colors.HexColor("#1D2B34")
GRID = colors.HexColor("#D9E1E6")
NO_RECORD = colors.HexColor("#F1F3F4")


def _read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Unexpected schema for {path}: {reader.fieldnames}")
        return list(reader)


def _ratio(observed: int, missing: int) -> float | None:
    total = observed + missing
    return observed / total if total else None


def _parse_month(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _load_coverage(repo_root: Path) -> tuple[
    list[dict[str, object]],
    dict[float, dict[str, float | None]],
    list[str],
    dict[tuple[str, float, str], float | None],
]:
    """Load the approved count summaries and convert only count ratios."""
    monthly_rows = _read_csv(repo_root / MONTHLY_PATH, MONTHLY_FIELDS)
    monthly: list[dict[str, object]] = []
    for row in monthly_rows:
        month = _parse_month(row["time_month"])
        if START_MONTH <= month <= END_MONTH:
            observed = int(row["finite_count"])
            missing = int(row["missing_count"])
            monthly.append(
                {
                    "month": month,
                    "coverage": _ratio(observed, missing),
                    "missing": _ratio(missing, observed),
                }
            )
    if not monthly or monthly[0]["month"] != START_MONTH or monthly[-1]["month"] != END_MONTH:
        raise ValueError("Monthly coverage does not span the Figure 2 period")

    depth_rows = _read_csv(repo_root / DEPTH_PATH, DEPTH_FIELDS)
    depth_coverage: dict[float, dict[str, float | None]] = {}
    for row in depth_rows:
        depth = float(row["depth"])
        if depth in TARGET_DEPTHS:
            observed = int(row["finite_count"])
            missing = int(row["missing_count"])
            depth_coverage[depth] = {"all": _ratio(observed, missing)}
    if set(depth_coverage) != set(TARGET_DEPTHS):
        raise ValueError("Depth coverage is missing a target Figure 2 depth")

    station_rows = _read_csv(repo_root / STATION_DEPTH_VARIABLE_PATH, STATION_FIELDS)
    stations = sorted({row["station_id_raw"] for row in station_rows})
    heatmap: dict[tuple[str, float, str], float | None] = {
        (station, depth, variable): None
        for station in stations
        for depth in TARGET_DEPTHS
        for variable in ("Temperature", "Salinity")
    }
    variable_totals = {
        depth: {"Temperature": [0, 0], "Salinity": [0, 0]}
        for depth in TARGET_DEPTHS
    }
    for row in station_rows:
        depth = float(row["depth"])
        if depth not in TARGET_DEPTHS:
            continue
        station = row["station_id_raw"]
        for variable, observed_field, missing_field in (
            ("Temperature", "temperature_count", "temperature_missing_count"),
            ("Salinity", "salinity_count", "salinity_missing_count"),
        ):
            observed = int(row[observed_field])
            missing = int(row[missing_field])
            heatmap[(station, depth, variable)] = _ratio(observed, missing)
            variable_totals[depth][variable][0] += observed
            variable_totals[depth][variable][1] += missing
    for depth in TARGET_DEPTHS:
        depth_coverage[depth]["Temperature"] = _ratio(*variable_totals[depth]["Temperature"])
        depth_coverage[depth]["Salinity"] = _ratio(*variable_totals[depth]["Salinity"])
    return monthly, depth_coverage, stations, heatmap


def _linear_color(low: colors.Color, high: colors.Color, ratio: float) -> colors.Color:
    return colors.Color(
        low.red + (high.red - low.red) * ratio,
        low.green + (high.green - low.green) * ratio,
        low.blue + (high.blue - low.blue) * ratio,
    )


def _temperature_color(ratio: float) -> colors.Color:
    return _linear_color(colors.HexColor("#F9E4D7"), TEMPERATURE, ratio)


def _salinity_color(ratio: float) -> colors.Color:
    return _linear_color(colors.HexColor("#D9EAF4"), SALINITY, ratio)


def _draw_panel_a(
    pdf: canvas.Canvas, monthly: list[dict[str, object]], x: float, y: float, width: float, height: float
) -> None:
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(INK)
    pdf.drawString(x, y + height + 15, "(a) Monthly observation coverage")
    pdf.setFillColor(NO_RECORD)
    pdf.rect(x, y, width, height, stroke=0, fill=1)
    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.45)
    for fraction, label in ((0, "0"), (0.5, "50"), (1, "100")):
        line_y = y + height * fraction
        pdf.line(x, line_y, x + width, line_y)
        pdf.setFont("Helvetica", 7)
        pdf.setFillColor(INK)
        pdf.drawRightString(x - 5, line_y - 2, label)
    bar_width = width / len(monthly)
    for index, row in enumerate(monthly):
        coverage = row["coverage"]
        if not isinstance(coverage, float):
            continue
        bar_height = height * coverage
        pdf.setFillColor(SALINITY)
        pdf.rect(x + index * bar_width, y, max(bar_width, 0.35), bar_height, stroke=0, fill=1)
    for year in (1998, 2002, 2006, 2010, 2014, 2018, 2022, 2024):
        fraction = (year - 1998) / (2024 - 1998 + 7 / 12)
        label_x = x + width * fraction
        pdf.setStrokeColor(GRID)
        pdf.line(label_x, y, label_x, y + height)
        pdf.setFont("Helvetica", 7)
        pdf.setFillColor(INK)
        pdf.drawCentredString(label_x, y - 12, str(year))
    pdf.setStrokeColor(INK)
    pdf.rect(x, y, width, height, stroke=1, fill=0)
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(INK)
    pdf.drawString(x, y - 25, "Month")
    pdf.saveState()
    pdf.translate(x - 34, y + height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Observed coverage (%)")
    pdf.restoreState()


def _draw_panel_b(
    pdf: canvas.Canvas,
    depth_coverage: dict[float, dict[str, float | None]],
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(INK)
    pdf.drawString(x, y + height + 15, "(b) Depth-dependent coverage")
    pdf.setStrokeColor(GRID)
    for fraction, label in ((0, "0"), (0.5, "50"), (1, "100")):
        line_y = y + height * fraction
        pdf.line(x, line_y, x + width, line_y)
        pdf.setFont("Helvetica", 7)
        pdf.setFillColor(INK)
        pdf.drawRightString(x - 5, line_y - 2, label)
    slot = width / len(TARGET_DEPTHS)
    bar_width = slot * 0.26
    for index, depth in enumerate(TARGET_DEPTHS):
        centre = x + slot * (index + 0.5)
        for offset, variable, color in (
            (-bar_width * 0.65, "Temperature", TEMPERATURE),
            (bar_width * 0.65, "Salinity", SALINITY),
        ):
            coverage = depth_coverage[depth][variable]
            if coverage is not None:
                pdf.setFillColor(color)
                pdf.rect(centre + offset - bar_width / 2, y, bar_width, height * coverage, stroke=0, fill=1)
        pdf.setFont("Helvetica", 8)
        pdf.setFillColor(INK)
        pdf.drawCentredString(centre, y - 12, f"{int(depth)}")
    pdf.setStrokeColor(INK)
    pdf.rect(x, y, width, height, stroke=1, fill=0)
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(INK)
    pdf.drawString(x + width / 2 - 23, y - 25, "Depth (m)")
    pdf.saveState()
    pdf.translate(x - 34, y + height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Observed coverage (%)")
    pdf.restoreState()
    legend_y = y + height + 3
    for legend_x, label, color in (
        (x + width - 128, "Temperature", TEMPERATURE),
        (x + width - 52, "Salinity", SALINITY),
    ):
        pdf.setFillColor(color)
        pdf.rect(legend_x, legend_y, 7, 7, stroke=0, fill=1)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica", 7)
        pdf.drawString(legend_x + 10, legend_y, label)


def _draw_panel_c(
    pdf: canvas.Canvas,
    stations: list[str],
    heatmap: dict[tuple[str, float, str], float | None],
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(INK)
    pdf.drawString(x, y + height + 15, "(c) Station-depth-variable coverage")
    columns = [("Temperature", depth) for depth in TARGET_DEPTHS] + [
        ("Salinity", depth) for depth in TARGET_DEPTHS
    ]
    label_width = 40
    grid_x = x + label_width
    grid_width = width - label_width
    cell_width = grid_width / len(columns)
    cell_height = height / len(stations)
    for row_index, station in enumerate(stations):
        cell_y = y + height - (row_index + 1) * cell_height
        pdf.setFont("Helvetica", 6.4)
        pdf.setFillColor(INK)
        pdf.drawRightString(x + label_width - 4, cell_y + cell_height * 0.28, station)
        for column_index, (variable, depth) in enumerate(columns):
            ratio = heatmap[(station, depth, variable)]
            cell_x = grid_x + column_index * cell_width
            if ratio is None:
                color = NO_RECORD
            elif variable == "Temperature":
                color = _temperature_color(ratio)
            else:
                color = _salinity_color(ratio)
            pdf.setFillColor(color)
            pdf.setStrokeColor(colors.white)
            pdf.rect(cell_x, cell_y, cell_width, cell_height, stroke=1, fill=1)
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.6)
    pdf.rect(grid_x, y, grid_width, height, stroke=1, fill=0)
    pdf.line(grid_x + cell_width * 4, y, grid_x + cell_width * 4, y + height)
    for column_index, (variable, depth) in enumerate(columns):
        label_x = grid_x + (column_index + 0.5) * cell_width
        pdf.setFont("Helvetica", 6.4)
        pdf.setFillColor(INK)
        pdf.drawCentredString(label_x, y + height + 3, str(int(depth)))
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.setFillColor(TEMPERATURE)
    pdf.drawCentredString(grid_x + cell_width * 2, y + height + 12, "Temperature")
    pdf.setFillColor(SALINITY)
    pdf.drawCentredString(grid_x + cell_width * 6, y + height + 12, "Salinity")
    legend_y = y - 14
    for legend_x, label, color in (
        (grid_x, "low coverage", _temperature_color(0.15)),
        (grid_x + 86, "high coverage", _temperature_color(0.95)),
        (grid_x + 176, "no record", NO_RECORD),
    ):
        pdf.setFillColor(color)
        pdf.setStrokeColor(colors.HexColor("#A9B3B9"))
        pdf.rect(legend_x, legend_y, 7, 7, stroke=1, fill=1)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica", 6.7)
        pdf.drawString(legend_x + 10, legend_y, label)


def _find_pdf_renderer() -> str:
    bundled_renderer = (
        Path(sys.executable).resolve().parents[1]
        / "native"
        / "poppler"
        / "Library"
        / "bin"
        / "pdftoppm.exe"
    )
    if bundled_renderer.is_file():
        return str(bundled_renderer)
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("pdftoppm is required to render the Figure 2 PNG")
    return renderer


def plot_figure(repo_root: Path) -> tuple[Path, Path]:
    """Create the Figure 2 PDF master and a 300 dpi PNG render."""
    monthly, depth_coverage, stations, heatmap = _load_coverage(repo_root)
    output_dir = repo_root / FIGURE_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "Figure2_missing_pattern.pdf"
    png_path = output_dir / "Figure2_missing_pattern.png"

    page_width, page_height = 11 * inch, 8.5 * inch
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height))
    pdf.setTitle("Figure 2. Observation coverage and missing pattern")
    pdf.setAuthor("Atlantic_TS_Reconstruction_v2_algorithm")
    pdf.setFont("Helvetica-Bold", 14)
    pdf.setFillColor(INK)
    pdf.drawString(0.58 * inch, page_height - 0.43 * inch, "Figure 2. Observation coverage and missing pattern")
    _draw_panel_a(pdf, monthly, 0.78 * inch, 5.48 * inch, 4.36 * inch, 1.55 * inch)
    _draw_panel_b(pdf, depth_coverage, 6.05 * inch, 5.48 * inch, 3.85 * inch, 1.55 * inch)
    _draw_panel_c(pdf, stations, heatmap, 0.78 * inch, 0.88 * inch, 9.12 * inch, 3.65 * inch)
    pdf.setFont("Helvetica-Oblique", 7)
    pdf.setFillColor(INK)
    pdf.drawString(
        0.78 * inch,
        0.38 * inch,
        "Coverage is finite observations divided by recorded observations; pale cells have no station-depth-variable record.",
    )
    pdf.showPage()
    pdf.save()

    renderer = _find_pdf_renderer()
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_prefix = Path(temporary_dir) / "Figure2_missing_pattern"
        subprocess.run(
            [renderer, "-r", "300", "-png", "-singlefile", str(pdf_path), str(output_prefix)],
            check=True,
        )
        rendered_png = output_prefix.with_suffix(".png")
        if not rendered_png.is_file():
            raise RuntimeError("PDF renderer did not produce a PNG")
        png_path.write_bytes(rendered_png.read_bytes())
    return png_path, pdf_path


if __name__ == "__main__":
    plot_figure(Path(__file__).resolve().parents[1])
