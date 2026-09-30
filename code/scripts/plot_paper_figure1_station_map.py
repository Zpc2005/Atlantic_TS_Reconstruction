"""Render the Figure 1 station map from approved Figure-data inputs only."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


LOCATION_RELATIVE_PATH = Path("artifacts/paper/figure_data/figure1_station_locations.csv")
PROVENANCE_RELATIVE_PATH = Path("artifacts/paper/figure_data/figure1_station_provenance.json")
FIGURE_RELATIVE_DIR = Path("artifacts/paper/figures")
LOCATION_FIELDS = ["station_id", "station_name", "latitude", "longitude"]
NEW_STATIONS = frozenset({"0n3w", "20s10w"})
MAP_BOUNDS = (-45.0, 5.0, -25.0, 25.0)
LABEL_OFFSETS = {
    "0n0e": (8.0, 9.0),
    "0n3w": (-28.0, 9.0),
    "0n10w": (8.0, 9.0),
    "5s10w": (8.0, 9.0),
    "6s10w": (8.0, -13.0),
}


def _read_locations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LOCATION_FIELDS:
            raise ValueError(f"Unexpected location schema: {reader.fieldnames}")
        rows = list(reader)
    station_ids = {row["station_id"] for row in rows}
    if len(rows) != 20 or len(station_ids) != 20 or not NEW_STATIONS.issubset(station_ids):
        raise ValueError("Figure 1 requires 20 unique stations including both new stations")
    for row in rows:
        float(row["latitude"])
        float(row["longitude"])
    return rows


def _read_provenance(path: Path) -> dict[str, object]:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    required = {"source_file", "sha256", "extraction_time", "fields_used"}
    if not required.issubset(provenance):
        raise ValueError("Station provenance is incomplete")
    return provenance


def _coordinate_label(value: float, axis: str) -> str:
    if value == 0:
        return "0°"
    suffix = "W" if axis == "longitude" and value < 0 else "E"
    if axis == "latitude":
        suffix = "S" if value < 0 else "N"
    return f"{abs(int(value))}°{suffix}"


def _station_label_offset(
    station_id: str, latitude: float, longitude: float
) -> tuple[float, float]:
    if station_id in LABEL_OFFSETS:
        return LABEL_OFFSETS[station_id]
    horizontal = -8.0 if longitude >= 0 else 7.0
    vertical = -9.0 if latitude < 0 else 7.0
    return horizontal, vertical


def _draw_diamond(pdf: canvas.Canvas, x: float, y: float, size: float) -> None:
    pdf.saveState()
    pdf.translate(x, y)
    pdf.rotate(45)
    pdf.rect(-size / 2, -size / 2, size, size, stroke=1, fill=1)
    pdf.restoreState()


def _draw_map(pdf: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    page_width, page_height = 8.25 * inch, 5.6 * inch
    map_left, map_bottom = 0.82 * inch, 0.78 * inch
    map_width, map_height = 5.7 * inch, 4.05 * inch
    lon_min, lon_max, lat_min, lat_max = MAP_BOUNDS

    def x_coord(longitude: float) -> float:
        return map_left + (longitude - lon_min) / (lon_max - lon_min) * map_width

    def y_coord(latitude: float) -> float:
        return map_bottom + (latitude - lat_min) / (lat_max - lat_min) * map_height

    pdf.setFillColor(colors.HexColor("#F4F8FB"))
    pdf.setStrokeColor(colors.HexColor("#4D6B7A"))
    pdf.roundRect(map_left, map_bottom, map_width, map_height, 6, stroke=1, fill=1)

    pdf.setStrokeColor(colors.HexColor("#C8D7E1"))
    pdf.setLineWidth(0.45)
    for longitude in (-40, -30, -20, -10, 0):
        x = x_coord(float(longitude))
        pdf.line(x, map_bottom, x, map_bottom + map_height)
        pdf.setFillColor(colors.HexColor("#40515C"))
        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString(x, map_bottom - 0.20 * inch, _coordinate_label(longitude, "longitude"))
    for latitude in (-20, -10, 0, 10, 20):
        y = y_coord(float(latitude))
        pdf.line(map_left, y, map_left + map_width, y)
        pdf.setFillColor(colors.HexColor("#40515C"))
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(map_left - 0.12 * inch, y - 2.5, _coordinate_label(latitude, "latitude"))

    pdf.setStrokeColor(colors.HexColor("#4D6B7A"))
    pdf.setLineWidth(0.75)
    pdf.roundRect(map_left, map_bottom, map_width, map_height, 6, stroke=1, fill=0)
    pdf.setFillColor(colors.HexColor("#42677D"))
    pdf.setFont("Helvetica-Oblique", 9.5)
    pdf.drawRightString(
        map_left + map_width - 0.16 * inch,
        map_bottom + map_height - 0.18 * inch,
        "Tropical Atlantic",
    )

    for row in rows:
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        x, y = x_coord(longitude), y_coord(latitude)
        is_new = row["station_id"] in NEW_STATIONS
        pdf.setFillColor(colors.HexColor("#C84C4C") if is_new else colors.HexColor("#1F6FAF"))
        pdf.setStrokeColor(colors.white)
        pdf.setLineWidth(1.0)
        if is_new:
            _draw_diamond(pdf, x, y, 9.0)
        else:
            pdf.circle(x, y, 4.2, stroke=1, fill=1)
        dx, dy = _station_label_offset(row["station_id"], latitude, longitude)
        pdf.setFillColor(colors.HexColor("#263238"))
        pdf.setFont("Helvetica-Bold" if is_new else "Helvetica", 6.9)
        pdf.drawString(x + dx, y + dy, row["station_name"])

    legend_x = 6.92 * inch
    legend_y = 3.72 * inch
    pdf.setFillColor(colors.HexColor("#263238"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(legend_x, legend_y + 0.40 * inch, "Station class")
    pdf.setFillColor(colors.HexColor("#1F6FAF"))
    pdf.setStrokeColor(colors.white)
    pdf.circle(legend_x + 5, legend_y + 0.18 * inch, 4.2, stroke=1, fill=1)
    pdf.setFillColor(colors.HexColor("#263238"))
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(legend_x + 15, legend_y + 0.15 * inch, "Development (n=18)")
    pdf.setFillColor(colors.HexColor("#C84C4C"))
    pdf.setStrokeColor(colors.white)
    _draw_diamond(pdf, legend_x + 5, legend_y - 0.10 * inch, 9.0)
    pdf.setFillColor(colors.HexColor("#263238"))
    pdf.drawString(legend_x + 15, legend_y - 0.13 * inch, "New (n=2)")

    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(colors.HexColor("#40515C"))
    pdf.drawString(map_left, 0.25 * inch, "Longitude")
    pdf.saveState()
    pdf.translate(0.22 * inch, map_bottom + map_height / 2)
    pdf.rotate(90)
    pdf.drawString(0, 0, "Latitude")
    pdf.restoreState()
    pdf.setFont("Helvetica-Bold", 13)
    pdf.setFillColor(colors.HexColor("#1D2B34"))
    pdf.drawString(map_left, page_height - 0.40 * inch, "Figure 1. PIRATA station network")


def _find_pdf_renderer() -> str:
    """Return the bundled Poppler executable, avoiding a broken cmd wrapper."""
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
        raise RuntimeError("pdftoppm is required to render the Figure 1 PNG")
    return renderer


def plot_figure(repo_root: Path) -> tuple[Path, Path]:
    """Create the PDF master and a 300 dpi PNG render for Figure 1."""
    locations = _read_locations(repo_root / LOCATION_RELATIVE_PATH)
    _read_provenance(repo_root / PROVENANCE_RELATIVE_PATH)
    output_dir = repo_root / FIGURE_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "Figure1_station_map.pdf"
    png_path = output_dir / "Figure1_station_map.png"

    pdf = canvas.Canvas(str(pdf_path), pagesize=(8.25 * inch, 5.6 * inch))
    pdf.setTitle("Figure 1. PIRATA station network")
    pdf.setAuthor("Atlantic_TS_Reconstruction_v2_algorithm")
    _draw_map(pdf, locations)
    pdf.showPage()
    pdf.save()

    renderer = _find_pdf_renderer()
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_prefix = Path(temporary_dir) / "Figure1_station_map"
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
