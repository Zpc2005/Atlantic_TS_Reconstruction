"""Render a data-free three-stage methodological framework for Figure 3.

The figure is a conceptual description of the experimental design only. It
does not open data, prediction, result, or metric artifacts.
"""

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
NAVY = colors.HexColor("#1E4E6F")
BLUE = colors.HexColor("#2A79A8")
LIGHT_BLUE = colors.HexColor("#E7F1F6")
GREY = colors.HexColor("#F1F4F5")
MID_GREY = colors.HexColor("#C8D3D8")
INK = colors.HexColor("#1C2E38")
MUTED = colors.HexColor("#61737D")
WHITE = colors.white


def _text_lines(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    lines: tuple[str, ...],
    font: str = "Helvetica",
    size: float = 7.0,
    color: colors.Color = INK,
    leading: float = 9.3,
    centre: bool = True,
) -> None:
    """Draw a compact group of labels with a shared visual alignment."""
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    for number, line in enumerate(lines):
        text_y = y - number * leading
        if centre:
            pdf.drawCentredString(x, text_y, line)
        else:
            pdf.drawString(x, text_y, line)


def _module(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...] = (),
    *,
    fill: colors.Color = WHITE,
    border: colors.Color = MID_GREY,
    title_fill: colors.Color | None = None,
    title_color: colors.Color = INK,
    body_color: colors.Color = MUTED,
    title_size: float = 8.0,
) -> None:
    """Draw a sharp-cornered publication-style information module."""
    pdf.setFillColor(fill)
    pdf.setStrokeColor(border)
    pdf.setLineWidth(0.8)
    pdf.rect(x, y, width, height, stroke=1, fill=1)
    if title_fill is not None:
        band_height = min(17, height * 0.32)
        pdf.setFillColor(title_fill)
        pdf.rect(x, y + height - band_height, width, band_height, stroke=0, fill=1)
        title_y = y + height - band_height + 5.0
    else:
        title_y = y + height - 12.0
    _text_lines(
        pdf,
        x + width / 2,
        title_y,
        (title,),
        font="Helvetica-Bold",
        size=title_size,
        color=title_color,
        leading=8,
    )
    if lines:
        if title_fill is not None:
            start_y = y + height - band_height - 9
        else:
            start_y = y + height - 22
        _text_lines(
            pdf, x + width / 2, start_y, lines, size=6.4, color=body_color,
            leading=8.0,
        )


def _stage(
    pdf: canvas.Canvas,
    y: float,
    height: float,
    stage: str,
    title: str,
) -> tuple[float, float, float, float]:
    """Draw a left stage rail and a restrained dashed stage boundary."""
    rail_x, rail_width = 0.34 * inch, 0.46 * inch
    body_x, body_width = 0.96 * inch, 7.16 * inch
    pdf.setFillColor(NAVY)
    pdf.rect(rail_x, y, rail_width, height, stroke=0, fill=1)
    pdf.saveState()
    pdf.translate(rail_x + rail_width / 2, y + height / 2)
    pdf.rotate(90)
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 9.4)
    pdf.drawCentredString(0, 7.5, stage)
    pdf.setFont("Helvetica-Bold", 7.1)
    pdf.drawCentredString(0, -7.5, title)
    pdf.restoreState()
    pdf.setStrokeColor(MID_GREY)
    pdf.setLineWidth(0.85)
    pdf.setDash(3, 3)
    pdf.rect(body_x, y, body_width, height, stroke=1, fill=0)
    pdf.setDash()
    return body_x, y, body_width, height


def _arrow(
    pdf: canvas.Canvas,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    *,
    color: colors.Color = NAVY,
    width: float = 0.8,
) -> None:
    """Draw a small, restrained directional arrow between major modules."""
    pdf.setStrokeColor(color)
    pdf.setFillColor(color)
    pdf.setLineWidth(width)
    pdf.line(start_x, start_y, end_x, end_y)
    if abs(end_y - start_y) >= abs(end_x - start_x):
        sign = 1 if end_y > start_y else -1
        pdf.line(end_x, end_y, end_x - 3.2, end_y - sign * 5.3)
        pdf.line(end_x, end_y, end_x + 3.2, end_y - sign * 5.3)
    else:
        sign = 1 if end_x > start_x else -1
        pdf.line(end_x, end_y, end_x - sign * 5.3, end_y - 3.2)
        pdf.line(end_x, end_y, end_x - sign * 5.3, end_y + 3.2)


def _draw_stage_one(pdf: canvas.Canvas) -> None:
    body_x, y, body_width, height = _stage(pdf, 7.77 * inch, 2.98 * inch, "STAGE I", "DATA PREPARATION")
    _text_lines(pdf, body_x + 0.13 * inch, y + height - 0.20 * inch, ("OBSERVATIONAL FOUNDATION",), font="Helvetica-Bold", size=7.6, color=NAVY, centre=False)
    source_y, source_h = y + height - 0.88 * inch, 0.50 * inch
    _module(pdf, body_x + 0.13 * inch, source_y, 1.73 * inch, source_h, "PIRATA Temperature", ("20 stations", "1998-2024 monthly records"), fill=LIGHT_BLUE, border=BLUE, title_fill=BLUE, title_color=WHITE)
    _module(pdf, body_x + 5.30 * inch, source_y, 1.73 * inch, source_h, "PIRATA Salinity", ("18 stations", "1998-2024 monthly records"), fill=LIGHT_BLUE, border=BLUE, title_fill=BLUE, title_color=WHITE)
    central_x, central_w = body_x + 2.34 * inch, 2.48 * inch
    _module(pdf, central_x, source_y, central_w, source_h, "Native observations", ("temperature and salinity profiles",), fill=WHITE, border=MID_GREY, title_fill=GREY)
    _arrow(pdf, body_x + 1.86 * inch, source_y + source_h / 2, central_x, source_y + source_h / 2)
    _arrow(pdf, body_x + 5.30 * inch, source_y + source_h / 2, central_x + central_w, source_y + source_h / 2)
    process_y, process_h, process_w = y + height - 1.54 * inch, 0.36 * inch, 1.62 * inch
    process_x = (body_x + body_width / 2) - process_w / 2
    _module(pdf, process_x, process_y, process_w, process_h, "Quality control", ("consistent observation records",), fill=GREY)
    _arrow(pdf, central_x + central_w / 2, source_y, process_x + process_w / 2, process_y + process_h)
    depth_y = y + height - 2.05 * inch
    _module(pdf, process_x, depth_y, process_w, 0.36 * inch, "Target depth extraction", ("1 m   |   20 m   |   40 m   |   120 m",), fill=LIGHT_BLUE, border=BLUE)
    _arrow(pdf, process_x + process_w / 2, process_y, process_x + process_w / 2, depth_y + 0.36 * inch)
    dataset_y = y + 0.22 * inch
    _module(pdf, body_x + 0.13 * inch, dataset_y, 2.16 * inch, 0.43 * inch, "Unified T/S dataset", ("station  |  time  |  depth  |  variable",), fill=LIGHT_BLUE, border=BLUE, title_fill=BLUE, title_color=WHITE)
    _arrow(pdf, process_x, depth_y + 0.18 * inch, body_x + 2.29 * inch, dataset_y + 0.24 * inch)
    scenario_x, scenario_w = body_x + 2.86 * inch, 4.17 * inch
    pdf.setStrokeColor(MID_GREY)
    pdf.setLineWidth(0.65)
    pdf.setDash(2, 2)
    pdf.rect(scenario_x, dataset_y, scenario_w, 0.64 * inch, stroke=1, fill=0)
    pdf.setDash()
    _text_lines(pdf, scenario_x + scenario_w / 2, dataset_y + 0.49 * inch, ("REALISTIC MISSING RECONSTRUCTION PROBLEM",), font="Helvetica-Bold", size=6.9, color=NAVY)
    labels = (("M1", "Random missing"), ("M2", "Temporal gap"), ("M3", "Spatial extrapolation"))
    for index, (code, label) in enumerate(labels):
        module_x = scenario_x + 0.12 * inch + index * 1.31 * inch
        _module(pdf, module_x, dataset_y + 0.08 * inch, 1.12 * inch, 0.30 * inch, code, (label,), fill=GREY, border=MID_GREY, title_size=7.2)
    _arrow(pdf, body_x + 2.29 * inch, dataset_y + 0.24 * inch, scenario_x, dataset_y + 0.24 * inch)


def _draw_stage_two(pdf: canvas.Canvas) -> None:
    body_x, y, body_width, height = _stage(pdf, 4.39 * inch, 2.98 * inch, "STAGE II", "MODEL DEVELOPMENT")
    _text_lines(pdf, body_x + 0.13 * inch, y + height - 0.20 * inch, ("INPUT REPRESENTATION AND MULTI-MODEL LEARNING",), font="Helvetica-Bold", size=7.6, color=NAVY, centre=False)
    input_y = y + height - 0.85 * inch
    for index, label in enumerate(("Observed values", "Missing mask", "Temporal features", "Spatial graph")):
        _module(pdf, body_x + 0.13 * inch + index * 1.28 * inch, input_y, 1.10 * inch, 0.32 * inch, label, (), fill=LIGHT_BLUE, border=BLUE, title_size=6.7)
    feature_x = body_x + 5.38 * inch
    _module(pdf, feature_x, input_y, 1.65 * inch, 0.32 * inch, "Feature representation", ("value + mask + temporal + graph",), fill=GREY, title_size=7.0)
    _arrow(pdf, body_x + 5.23 * inch, input_y + 0.16 * inch, feature_x, input_y + 0.16 * inch)
    family_y, family_h = y + 1.14 * inch, 0.92 * inch
    families = ((body_x + 0.13 * inch, 2.15 * inch, "Classical baselines", ("WOA23 climatology", "OI", "DINEOF"), GREY, MID_GREY, INK), (body_x + 2.50 * inch, 1.70 * inch, "Machine learning", ("XGBoost",), LIGHT_BLUE, BLUE, INK), (body_x + 4.42 * inch, 2.61 * inch, "Deep learning models", ("MLP   |   LSTM   |   Transformer", "Mask-aware STGNN (proposed)"), NAVY, NAVY, WHITE))
    for family_x, family_w, title, lines, fill, border, title_color in families:
        _module(pdf, family_x, family_y, family_w, family_h, title, lines, fill=fill, border=border, title_color=title_color, body_color=WHITE if fill == NAVY else MUTED, title_size=8.0)
    _arrow(
        pdf, feature_x + 0.83 * inch, input_y,
        feature_x + 0.83 * inch, family_y + family_h,
    )
    strategy_y = y + 0.25 * inch
    _module(pdf, body_x + 0.13 * inch, strategy_y, 6.90 * inch, 0.50 * inch, "Training strategy", ("Training 1998-2014     |     Validation 2015-2018     |     Test 2019-2024", "training-only normalization     |     early stopping"), fill=GREY, border=MID_GREY, title_fill=LIGHT_BLUE, title_color=NAVY)
    _arrow(pdf, body_x + body_width / 2, family_y, body_x + body_width / 2, strategy_y + 0.50 * inch)


def _draw_stage_three(pdf: canvas.Canvas) -> None:
    body_x, y, body_width, height = _stage(pdf, 0.75 * inch, 3.12 * inch, "STAGE III", "RECONSTRUCTION AND EVALUATION")
    _text_lines(pdf, body_x + 0.13 * inch, y + height - 0.20 * inch, ("RELIABILITY-CONTROLLED SCIENTIFIC ASSESSMENT",), font="Helvetica-Bold", size=7.6, color=NAVY, centre=False)
    field_y = y + height - 0.86 * inch
    _module(pdf, body_x + 2.13 * inch, field_y, 2.90 * inch, 0.42 * inch, "Reconstructed temperature and salinity fields", ("multi-depth monthly reconstruction",), fill=LIGHT_BLUE, border=BLUE, title_fill=BLUE, title_color=WHITE)
    _module(pdf, body_x + 5.38 * inch, field_y, 1.65 * inch, 0.42 * inch, "Frozen evaluation protocol", ("fixed and reproducible",), fill=GREY)
    _arrow(pdf, body_x + 5.03 * inch, field_y + 0.21 * inch, body_x + 5.38 * inch, field_y + 0.21 * inch)
    branch_y, branch_h = y + 1.32 * inch, 0.91 * inch
    branches = ((body_x + 0.13 * inch, 2.15 * inch, "Point-wise accuracy", ("RMSE", "R²", "ACC"), LIGHT_BLUE, BLUE, INK), (body_x + 2.50 * inch, 2.15 * inch, "Robustness evaluation", ("M1 random", "M2 temporal gap", "M3 spatial"), GREY, MID_GREY, INK), (body_x + 4.87 * inch, 2.16 * inch, "Generalization", ("LOSO spatial validation", "new stations: 0n3w, 20s10w"), LIGHT_BLUE, BLUE, INK))
    for branch_x, branch_w, title, lines, fill, border, title_color in branches:
        _module(pdf, branch_x, branch_y, branch_w, branch_h, title, lines, fill=fill, border=border, title_color=title_color, title_size=8.0)
    _arrow(pdf, body_x + 6.205 * inch, field_y, body_x + 6.205 * inch, branch_y + branch_h)
    finding_y = y + 0.34 * inch
    _module(pdf, body_x + 0.13 * inch, finding_y, 6.90 * inch, 0.53 * inch, "Scientific findings", ("Reliable T/S reconstruction under sparse observations     |     Improved missing-data robustness", "Spatial-temporal learning capability across depth and station settings"), fill=WHITE, border=NAVY, title_fill=NAVY, title_color=WHITE, body_color=MUTED)
    _arrow(pdf, body_x + body_width / 2, branch_y, body_x + body_width / 2, finding_y + 0.53 * inch)


def _find_pdf_renderer() -> str:
    bundled_renderer = Path(sys.executable).resolve().parents[1] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
    if bundled_renderer.is_file():
        return str(bundled_renderer)
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("pdftoppm is required to render the Figure 3 PNG")
    return renderer


def plot_figure(repo_root: Path) -> tuple[Path, Path]:
    """Create editable-vector PDF and PNG Figure 3 outputs."""
    output_dir = repo_root / FIGURE_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "Figure3_framework.pdf"
    png_path = output_dir / "Figure3_framework.png"
    page_width, page_height = 8.5 * inch, 11.2 * inch
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height))
    pdf.setTitle("Figure 3. Overall experimental framework")
    pdf.setAuthor("Atlantic_TS_Reconstruction_v2_algorithm")
    _text_lines(pdf, 4.25 * inch, 10.96 * inch, ("Figure 3. Overall experimental framework for tropical Atlantic temperature and salinity reconstruction",), font="Helvetica-Bold", size=10.6, color=INK)
    pdf.setStrokeColor(MID_GREY)
    pdf.setLineWidth(0.7)
    pdf.line(0.34 * inch, 10.77 * inch, 8.12 * inch, 10.77 * inch)
    _draw_stage_one(pdf)
    _draw_stage_two(pdf)
    _draw_stage_three(pdf)
    _text_lines(pdf, 4.25 * inch, 0.35 * inch, ("Conceptual experimental framework; no experimental results or predictions are displayed.",), font="Helvetica-Oblique", size=6.6, color=MUTED)
    pdf.showPage()
    pdf.save()
    renderer = _find_pdf_renderer()
    with tempfile.TemporaryDirectory() as temporary_dir:
        prefix = Path(temporary_dir) / "Figure3_framework"
        subprocess.run([renderer, "-r", "300", "-png", "-singlefile", str(pdf_path), str(prefix)], check=True)
        rendered_png = prefix.with_suffix(".png")
        if not rendered_png.is_file():
            raise RuntimeError("PDF renderer did not produce a PNG")
        png_path.write_bytes(rendered_png.read_bytes())
    return png_path, pdf_path


if __name__ == "__main__":
    plot_figure(Path(__file__).resolve().parents[1])
