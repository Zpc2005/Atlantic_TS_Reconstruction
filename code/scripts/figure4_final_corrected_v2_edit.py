"""Minimal text-only correction for the audited Figure 4 raster-backed PDF.

The source figure is preserved as a 600-dpi raster reference.  Only the M1
scope/task note and the validation-selected kNN definition are repainted; all
panel geometry, colors, arrows, boxes, and scientific model blocks remain
unchanged.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PNG = ROOT / "artifacts/paper/figures/Figure4_final_corrected_600dpi.png"
OUTPUT_PNG = ROOT / "artifacts/paper/figures/Figure4_final_corrected_v2.png"
OUTPUT_PDF = ROOT / "artifacts/paper/figures/Figure4_final_corrected_v2.pdf"

FONT_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")

# The source page is 648 x 462.84 points and the reference PNG is 600 dpi.
PAGE_SIZE_PT = (648.0, 462.84)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font_obj: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font_obj)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) // 2 - bounds[0]
    y = top + (bottom - top - height) // 2 - bounds[1]
    draw.text((x, y), text, font=font_obj, fill=fill)


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    path: Path,
    max_size: int,
    max_width: int,
    min_size: int = 20,
) -> ImageFont.FreeTypeFont:
    for size in range(max_size, min_size - 1, -1):
        candidate = font(path, size)
        width = draw.textbbox((0, 0), text, font=candidate)[2]
        if width <= max_width:
            return candidate
    return font(path, min_size)


def build_png() -> None:
    if not SOURCE_PNG.exists():
        raise FileNotFoundError(SOURCE_PNG)

    image = Image.open(SOURCE_PNG).convert("RGB")
    draw = ImageDraw.Draw(image)
    dark_blue = (31, 65, 140)
    black = (20, 20, 20)

    # The current corrected source already contains no visible unsupported
    # attention/GCN/message-passing/gating labels.  No architecture block is
    # removed or moved here.

    # Task box heading: make the M1 scope explicit without changing the box.
    # The task/note box occupies the rightmost part of panel (a); keep the
    # repaint strictly inside it so the graph branch to its left is untouched.
    heading_box = (4050, 1395, 5205, 1535)
    draw.rectangle(heading_box, fill="white")
    heading = "8 independent reconstruction targets (M1)"
    heading_font = fit_font(draw, heading, FONT_BOLD, 78, heading_box[2] - heading_box[0] - 70, 48)
    centered_text(draw, heading_box, heading, heading_font, dark_blue)

    # Preserve the existing per-target statement and add the required scope
    # limitation as a third compact line inside the existing task/note box.
    note_box = (4050, 1700, 5205, 1890)
    draw.rectangle(note_box, fill="white")
    note_lines = (
        "One model is trained for each",
        "variable-depth target",
        "Architecture shown for M1",
        "reconstruction experiments",
    )
    note_font = fit_font(draw, note_lines[0], FONT_REGULAR, 36, note_box[2] - note_box[0] - 40, 26)
    scope_font = fit_font(draw, note_lines[2], FONT_REGULAR, 27, note_box[2] - note_box[0] - 40, 20)
    line_y = (1705, 1748, 1795, 1830)
    for text, y, text_font in (
        (note_lines[0], line_y[0], note_font),
        (note_lines[1], line_y[1], note_font),
        (note_lines[2], line_y[2], scope_font),
        (note_lines[3], line_y[3], scope_font),
    ):
        bounds = draw.textbbox((0, 0), text, font=text_font)
        x = note_box[0] + (note_box[2] - note_box[0] - (bounds[2] - bounds[0])) // 2
        draw.text((x, y), text, font=text_font, fill=black)

    # Bottom Symbols definition: k is validation-selected per variable-depth
    # task, not a single global fixed value.
    k_box = (1000, 3590, 1980, 3675)
    draw.rectangle(k_box, fill="white")
    k_text = "k: neighbor count (kNN), selected using validation data"
    k_font = fit_font(draw, k_text, FONT_REGULAR, 34, k_box[2] - k_box[0] - 12, 20)
    centered_text(draw, k_box, k_text, k_font, dark_blue)

    image.save(OUTPUT_PNG, format="PNG", dpi=(600, 600), optimize=False)


def build_pdf() -> None:
    # Embed the edited 600-dpi reference without rescaling its page geometry.
    # The source PDF is itself raster-backed; this preserves pixel identity of
    # the approved reference while retaining the exact original page size.
    pdf = canvas.Canvas(str(OUTPUT_PDF), pagesize=PAGE_SIZE_PT, pageCompression=1)
    pdf.drawImage(str(OUTPUT_PNG), 0, 0, width=PAGE_SIZE_PT[0], height=PAGE_SIZE_PT[1], mask="auto")
    pdf.showPage()
    pdf.save()


def main() -> None:
    build_png()
    build_pdf()
    print(f"wrote {OUTPUT_PNG}")
    print(f"wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
