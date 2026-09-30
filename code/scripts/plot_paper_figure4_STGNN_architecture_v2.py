"""Render Figure 4 v2 as a data-free Mask-aware STGNN mechanism diagram."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


FIGURE_DIRECTORY = Path("artifacts/paper/figures")
INK = colors.HexColor("#172B36")
MUTED = colors.HexColor("#61737F")
INPUT_BLUE = colors.HexColor("#2A7EAE")
INPUT_PALE = colors.HexColor("#E5F2F8")
TEMPORAL_PURPLE = colors.HexColor("#735AB0")
TEMPORAL_PALE = colors.HexColor("#F0ECF8")
SPATIAL_GREEN = colors.HexColor("#2E8563")
SPATIAL_PALE = colors.HexColor("#E5F3EC")
MASK_ORANGE = colors.HexColor("#D3642A")
MASK_PALE = colors.HexColor("#FDE9DD")
OUTPUT_BLUE = colors.HexColor("#285E8B")
OUTPUT_PALE = colors.HexColor("#E7F0F5")


def _heading(pdf: canvas.Canvas, x: float, y: float, text: str, color: colors.Color) -> None:
    pdf.setFillColor(color)
    pdf.setFont("Helvetica-Bold", 9.8)
    pdf.drawString(x, y, text)


def _caption(pdf: canvas.Canvas, x: float, y: float, text: str) -> None:
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.5)
    pdf.drawString(x, y, text)


def _feature_map(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: colors.Color,
    stroke: colors.Color,
    offset: float = 0.0,
) -> None:
    """Draw one slanted feature-map plane in an architecture-figure style."""
    path = pdf.beginPath()
    path.moveTo(x + offset, y)
    path.lineTo(x + width + offset, y)
    path.lineTo(x + width + offset + 10, y + height)
    path.lineTo(x + offset + 10, y + height)
    path.close()
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(1.0)
    pdf.drawPath(path, stroke=1, fill=1)


def _ribbon(pdf: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color: colors.Color) -> None:
    """Use a light band to show tensor movement without a workflow arrow."""
    pdf.setStrokeColor(color)
    pdf.setLineWidth(1.5)
    pdf.line(x1, y1 - 3, x2, y2 - 3)
    pdf.setLineWidth(0.55)
    pdf.line(x1, y1 + 3, x2, y2 + 3)


def _network(pdf: canvas.Canvas, x: float, y: float, color: colors.Color) -> None:
    nodes = ((-0.22, 0.14), (0.18, 0.16), (-0.04, -0.18), (0.25, -0.16))
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
    pdf.setStrokeColor(color)
    pdf.setLineWidth(0.9)
    for start, end in edges:
        pdf.line(x + nodes[start][0] * inch, y + nodes[start][1] * inch, x + nodes[end][0] * inch, y + nodes[end][1] * inch)
    for point_x, point_y in nodes:
        pdf.setFillColor(color)
        pdf.setStrokeColor(colors.white)
        pdf.circle(x + point_x * inch, y + point_y * inch, 3.7, stroke=1, fill=1)


def _input_tensor(pdf: canvas.Canvas) -> None:
    _heading(pdf, 0.58 * inch, 5.98 * inch, "Input ocean observation tensor", INPUT_BLUE)
    _caption(pdf, 0.58 * inch, 5.78 * inch, "X in R(station x depth x time)")
    base_x, base_y = 0.82 * inch, 3.57 * inch
    _feature_map(pdf, base_x, base_y, 1.10 * inch, 1.25 * inch, INPUT_PALE, INPUT_BLUE, 16)
    _feature_map(pdf, base_x, base_y, 1.10 * inch, 1.25 * inch, colors.white, INPUT_BLUE, 8)
    _feature_map(pdf, base_x, base_y, 1.10 * inch, 1.25 * inch, INPUT_PALE, INPUT_BLUE)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 7.2)
    pdf.drawString(base_x + 0.17 * inch, base_y + 0.90 * inch, "Observed T / S")
    pdf.setFont("Helvetica", 6.3)
    pdf.drawString(base_x + 0.17 * inch, base_y + 0.75 * inch, "observed channels")
    pdf.setFillColor(MASK_ORANGE)
    pdf.circle(base_x + 0.25 * inch, base_y + 0.40 * inch, 4, stroke=0, fill=1)
    pdf.setFillColor(INPUT_BLUE)
    pdf.circle(base_x + 0.48 * inch, base_y + 0.40 * inch, 4, stroke=0, fill=1)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 5.9)
    pdf.drawString(base_x + 0.13 * inch, base_y + 0.20 * inch, "Temperature / Salinity")

    mask_x, mask_y = 2.35 * inch, 4.15 * inch
    _feature_map(pdf, mask_x, mask_y, 0.74 * inch, 0.54 * inch, colors.white, INPUT_BLUE)
    pattern = ((1, 1, 0, 1), (1, 0, 1, 1))
    for row, values in enumerate(pattern):
        for column, observed in enumerate(values):
            pdf.setFillColor(INPUT_BLUE if observed else colors.white)
            pdf.setStrokeColor(colors.HexColor("#AFBFC7"))
            pdf.rect(mask_x + 7 + column * 10, mask_y + 26 - row * 11, 8, 8, stroke=1, fill=1)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 7.0)
    pdf.drawString(mask_x, 3.72 * inch, "Missing mask")
    pdf.setFillColor(MASK_ORANGE)
    pdf.drawString(mask_x, 3.57 * inch, "M in {0,1}")
    _caption(pdf, mask_x, 3.42 * inch, "missing state is explicit")

    temporal_x, temporal_y = 2.35 * inch, 2.47 * inch
    pdf.setFillColor(INPUT_PALE)
    pdf.setStrokeColor(INPUT_BLUE)
    pdf.circle(temporal_x + 0.30 * inch, temporal_y + 0.22 * inch, 0.31 * inch, stroke=1, fill=1)
    pdf.setStrokeColor(MASK_ORANGE)
    pdf.setLineWidth(1.2)
    pdf.arc(temporal_x + 0.10 * inch, temporal_y + 0.02 * inch, temporal_x + 0.50 * inch, temporal_y + 0.42 * inch, 30, 285)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 7.0)
    pdf.drawString(temporal_x, 2.13 * inch, "Temporal encoding")
    _caption(pdf, temporal_x, 1.98 * inch, "Month / seasonal cycle")

    graph_x, graph_y = 1.10 * inch, 2.55 * inch
    _network(pdf, graph_x + 0.34 * inch, graph_y + 0.15 * inch, INPUT_BLUE)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 7.0)
    pdf.drawString(0.84 * inch, 2.13 * inch, "Spatial graph")
    _caption(pdf, 0.84 * inch, 1.98 * inch, "station graph / adjacency A")


def _branches_and_fusion(pdf: canvas.Canvas) -> None:
    _heading(pdf, 3.68 * inch, 5.98 * inch, "Mask-aware Spatio-temporal Graph Neural Network", TEMPORAL_PURPLE)
    temporal_x, temporal_y = 4.40 * inch, 4.55 * inch
    _feature_map(pdf, temporal_x - 0.45 * inch, temporal_y - 0.50 * inch, 1.22 * inch, 0.92 * inch, TEMPORAL_PALE, TEMPORAL_PURPLE)
    pdf.setFillColor(TEMPORAL_PURPLE)
    pdf.setFont("Helvetica-Bold", 8.0)
    pdf.drawCentredString(temporal_x + 0.20 * inch, temporal_y + 0.14 * inch, "Temporal encoder")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.3)
    pdf.drawCentredString(temporal_x + 0.20 * inch, temporal_y + 0.27 * inch, "Temporal sequence")
    pdf.drawCentredString(temporal_x + 0.20 * inch, temporal_y - 0.02 * inch, "attention mechanism")
    pdf.drawCentredString(temporal_x + 0.20 * inch, temporal_y - 0.13 * inch, "temporal dependency")
    pdf.setFillColor(TEMPORAL_PURPLE)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawCentredString(temporal_x + 0.20 * inch, temporal_y - 0.33 * inch, "Temporal representation")

    spatial_x, spatial_y = 4.40 * inch, 2.40 * inch
    _feature_map(pdf, spatial_x - 0.45 * inch, spatial_y - 0.50 * inch, 1.22 * inch, 0.92 * inch, SPATIAL_PALE, SPATIAL_GREEN)
    _network(pdf, spatial_x - 0.50 * inch, spatial_y + 0.06 * inch, SPATIAL_GREEN)
    pdf.setFillColor(SPATIAL_GREEN)
    pdf.setFont("Helvetica-Bold", 8.0)
    pdf.drawCentredString(spatial_x + 0.34 * inch, spatial_y + 0.14 * inch, "Graph encoder")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.3)
    pdf.drawCentredString(spatial_x + 0.34 * inch, spatial_y + 0.27 * inch, "Station nodes")
    pdf.drawCentredString(spatial_x + 0.34 * inch, spatial_y - 0.02 * inch, "graph convolution")
    pdf.drawCentredString(spatial_x + 0.34 * inch, spatial_y - 0.13 * inch, "message passing")
    pdf.setFillColor(SPATIAL_GREEN)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawCentredString(spatial_x + 0.34 * inch, spatial_y - 0.33 * inch, "Spatial representation")

    gate_x, gate_y = 6.15 * inch, 3.45 * inch
    pdf.setFillColor(MASK_PALE)
    pdf.setStrokeColor(MASK_ORANGE)
    pdf.setLineWidth(2.2)
    pdf.circle(gate_x, gate_y, 0.70 * inch, stroke=1, fill=1)
    pdf.setFillColor(MASK_ORANGE)
    pdf.setFont("Helvetica-Bold", 9.0)
    pdf.drawCentredString(gate_x, gate_y + 9, "Mask-aware")
    pdf.drawCentredString(gate_x, gate_y - 3, "gating")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica", 6.4)
    pdf.drawCentredString(gate_x, gate_y - 17, "missing awareness")
    pdf.drawCentredString(gate_x, gate_y - 27, "observation reliability")
    pdf.setStrokeColor(MASK_ORANGE)
    pdf.setDash(3, 2)
    pdf.setLineWidth(1.1)
    pdf.line(5.60 * inch, 4.30 * inch, gate_x - 0.40 * inch, gate_y + 0.37 * inch)
    pdf.line(5.60 * inch, 2.66 * inch, gate_x - 0.40 * inch, gate_y - 0.37 * inch)
    pdf.setDash()

    fusion_x, fusion_y = 7.48 * inch, 3.45 * inch
    _ribbon(pdf, 5.70 * inch, 4.42 * inch, fusion_x - 0.48 * inch, fusion_y + 0.25 * inch, TEMPORAL_PURPLE)
    _ribbon(pdf, 5.70 * inch, 2.50 * inch, fusion_x - 0.48 * inch, fusion_y - 0.25 * inch, SPATIAL_GREEN)
    _ribbon(pdf, gate_x + 0.70 * inch, gate_y, fusion_x - 0.48 * inch, fusion_y, MASK_ORANGE)
    path = pdf.beginPath()
    path.moveTo(fusion_x, fusion_y + 0.72 * inch)
    path.lineTo(fusion_x + 0.68 * inch, fusion_y)
    path.lineTo(fusion_x, fusion_y - 0.72 * inch)
    path.lineTo(fusion_x - 0.68 * inch, fusion_y)
    path.close()
    pdf.setFillColor(OUTPUT_PALE)
    pdf.setStrokeColor(OUTPUT_BLUE)
    pdf.setLineWidth(1.8)
    pdf.drawPath(path, stroke=1, fill=1)
    pdf.setFillColor(OUTPUT_BLUE)
    pdf.setFont("Helvetica-Bold", 8.2)
    pdf.drawCentredString(fusion_x, fusion_y + 5, "Spatio-temporal")
    pdf.drawCentredString(fusion_x, fusion_y - 6, "fusion")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.2)
    pdf.drawCentredString(fusion_x, fusion_y - 18, "reliable feature tensor")


def _decoder_and_outputs(pdf: canvas.Canvas) -> None:
    _heading(pdf, 8.58 * inch, 5.98 * inch, "Reconstruction head", OUTPUT_BLUE)
    decoder_x, decoder_y = 9.05 * inch, 3.45 * inch
    _ribbon(pdf, 8.17 * inch, decoder_y, decoder_x - 0.55 * inch, decoder_y, OUTPUT_BLUE)
    _feature_map(pdf, decoder_x - 0.42 * inch, decoder_y - 0.56 * inch, 1.15 * inch, 1.12 * inch, OUTPUT_PALE, OUTPUT_BLUE)
    pdf.setFillColor(OUTPUT_BLUE)
    pdf.setFont("Helvetica-Bold", 8.2)
    pdf.drawCentredString(decoder_x + 0.18 * inch, decoder_y + 7, "Decoder")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 6.3)
    pdf.drawCentredString(decoder_x + 0.18 * inch, decoder_y - 6, "reconstruction")
    pdf.drawCentredString(decoder_x + 0.18 * inch, decoder_y - 16, "feature map")

    output_x, top, height = 10.78 * inch, 4.67 * inch, 2.67 * inch
    _ribbon(pdf, decoder_x + 0.83 * inch, decoder_y, output_x - 0.52 * inch, decoder_y, OUTPUT_BLUE)
    pdf.setFillColor(OUTPUT_PALE)
    pdf.setStrokeColor(OUTPUT_BLUE)
    pdf.setLineWidth(1.5)
    pdf.ellipse(output_x - 0.38 * inch, top - height, output_x + 0.38 * inch, top, stroke=1, fill=1)
    depths = (("1 m", 0.14), ("20 m", 0.37), ("40 m", 0.60), ("120 m", 0.83))
    for depth, fraction in depths:
        line_y = top - height * fraction
        pdf.setStrokeColor(colors.HexColor("#ABC4D4"))
        pdf.setLineWidth(0.7)
        pdf.line(output_x - 0.29 * inch, line_y, output_x + 0.29 * inch, line_y)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 6.0)
        pdf.drawRightString(output_x - 0.48 * inch, line_y - 2, depth)
    for point_x, fraction in ((-0.13, 0.14), (0.10, 0.37), (-0.04, 0.60), (0.16, 0.83)):
        pdf.setFillColor(MASK_ORANGE)
        pdf.circle(output_x + point_x * inch, top - height * fraction, 3.0, stroke=0, fill=1)
    for point_x, fraction in ((0.13, 0.14), (-0.09, 0.37), (0.16, 0.60), (-0.06, 0.83)):
        pdf.setFillColor(OUTPUT_BLUE)
        pdf.circle(output_x + point_x * inch, top - height * fraction, 3.0, stroke=0, fill=1)
    pdf.setFillColor(MASK_ORANGE)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawCentredString(output_x, 1.92 * inch, "Temperature reconstruction")
    pdf.setFillColor(OUTPUT_BLUE)
    pdf.drawCentredString(output_x, 1.76 * inch, "Salinity reconstruction")


def _find_pdf_renderer() -> str:
    bundled_renderer = Path(sys.executable).resolve().parents[1] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
    if bundled_renderer.is_file():
        return str(bundled_renderer)
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("pdftoppm is required to render the Figure 4 v2 PNG")
    return renderer


def plot_figure(repo_root: Path) -> tuple[Path, Path]:
    """Create the Figure 4 v2 PDF master and a 300 dpi PNG render."""
    output_dir = repo_root / FIGURE_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "Figure4_STGNN_architecture_v2.pdf"
    png_path = output_dir / "Figure4_STGNN_architecture_v2.png"
    page_width, page_height = 12.2 * inch, 6.8 * inch
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height))
    pdf.setTitle("Architecture of the Mask-aware STGNN")
    pdf.setAuthor("Atlantic_TS_Reconstruction_v2_algorithm")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 14.2)
    pdf.drawString(0.58 * inch, 6.38 * inch, "Architecture of the Mask-aware STGNN for tropical Atlantic temperature and salinity reconstruction")
    pdf.setStrokeColor(colors.HexColor("#D1DEE3"))
    pdf.setLineWidth(0.9)
    pdf.line(0.58 * inch, 6.18 * inch, 11.55 * inch, 6.18 * inch)
    _input_tensor(pdf)
    _branches_and_fusion(pdf)
    _decoder_and_outputs(pdf)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Oblique", 6.7)
    pdf.drawString(0.58 * inch, 0.40 * inch, "Value, mask, temporal, and spatial channels are fused into reliable multi-depth temperature and salinity reconstructions.")
    pdf.showPage()
    pdf.save()
    renderer = _find_pdf_renderer()
    with tempfile.TemporaryDirectory() as temporary_dir:
        prefix = Path(temporary_dir) / "Figure4_STGNN_architecture_v2"
        subprocess.run([renderer, "-r", "300", "-png", "-singlefile", str(pdf_path), str(prefix)], check=True)
        rendered_png = prefix.with_suffix(".png")
        if not rendered_png.is_file():
            raise RuntimeError("PDF renderer did not produce a PNG")
        png_path.write_bytes(rendered_png.read_bytes())
    return png_path, pdf_path


if __name__ == "__main__":
    plot_figure(Path(__file__).resolve().parents[1])
