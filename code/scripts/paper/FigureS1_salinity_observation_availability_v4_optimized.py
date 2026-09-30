"""Optimize S1 salinity availability presentation without changing data.

Rows are stations and columns are 20 m/40 m.  Each station row has one
shared y-range for both depths.  Internal grid lines are intentionally omitted
so only the panel borders remain; missing native months still break the line.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts/paper/FigureS1_salinity_observation_availability.py"
PNG_PATH = ROOT / "artifacts/paper/figures/FigureS1_salinity_observation_availability_v4_optimized.png"
PDF_PATH = ROOT / "artifacts/paper/figures/FigureS1_salinity_observation_availability_v4_optimized.pdf"

ROW_RANGES = {
    "15n38w": (35.0, 37.2, (35.0, 35.5, 36.0, 36.5, 37.0)),
    "20n38w": (36.2, 37.8, (36.2, 36.6, 37.0, 37.4, 37.8)),
    "0n35w": (35.2, 36.8, (35.2, 35.6, 36.0, 36.4, 36.8)),
}


def load_base():
    spec = importlib.util.spec_from_file_location("figure_s1_salinity_base_v4_opt", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load base renderer: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def line_series_pdf(base, c, data, station, depth, left, bottom, width, height):
    previous = None
    for idx, month in enumerate(base.MONTHS):
        value = data.get((station, depth, month))
        if value is None:
            previous = None
            continue
        point = (base.map_x(idx, left, width), base.map_y(value, bottom, height))
        if previous is not None:
            base.draw_line_pdf(c, previous[0], previous[1], point[0], point[1], base.BLUE, 1.2)
        previous = point


def line_series_png(base, d, data, station, depth, left, bottom, width, height):
    previous = None
    line_width = max(2, round(1.2 * base.SCALE))
    for idx, month in enumerate(base.MONTHS):
        value = data.get((station, depth, month))
        if value is None:
            previous = None
            continue
        point = (base.map_x(idx, left, width), base.map_y(value, bottom, height))
        px, py = point[0] * base.SCALE, (base.PAGE_H - point[1]) * base.SCALE
        if previous is not None:
            d.line(
                (previous[0] * base.SCALE, (base.PAGE_H - previous[1]) * base.SCALE, px, py),
                fill=base.rgb(base.BLUE), width=line_width,
            )
        previous = point


def vertical_pdf_text(base, c, x, y, text, size, bold=False):
    c.saveState()
    c.translate(x, y)
    c.rotate(90)
    c.setFillColor(base.HexColor(base.INK))
    c.setFont(base.PDF_BOLD if bold else base.PDF_REGULAR, size)
    c.drawCentredString(0, -size * 0.34, text)
    c.restoreState()


def vertical_png_text(base, d, x, y, text, size, bold=False):
    font = base.png_font(size, bold)
    bbox = d.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = round(4 * base.SCALE)
    layer = base.Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (255, 255, 255, 0))
    ld = base.ImageDraw.Draw(layer)
    ld.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=base.rgb(base.INK) + (255,))
    rotated = layer.rotate(90, expand=True)
    cx, cy = x * base.SCALE, (base.PAGE_H - y) * base.SCALE
    d._image.paste(rotated, (round(cx - rotated.width / 2), round(cy - rotated.height / 2)), rotated)


def draw_panel_pdf(base, c, data, station, depth, row, col, left, bottom, width, height):
    y_min, y_max, y_ticks = ROW_RANGES[station]
    base.Y_MIN, base.Y_MAX = y_min, y_max
    c.setStrokeColor(base.HexColor(base.SPINE))
    c.setLineWidth(0.7)
    c.setFillColor(base.white)
    c.rect(left, bottom, width, height, fill=1, stroke=1)
    if col == 0:
        for tick in y_ticks:
            y = base.map_y(tick, bottom, height)
            base.draw_pdf_text(c, left - 8, y - 3, f"{tick:g}", 8, base.INK, False, "right")
        vertical_pdf_text(base, c, left - 27, bottom + height / 2, "Salinity (PSU)", 8)
    if row == 0:
        base.draw_pdf_text(c, left + width / 2, bottom + height + 10, f"{int(depth)} m", 10, base.INK, True, "center")
    if col == 0:
        vertical_pdf_text(base, c, left - 54, bottom + height / 2, base.station_label(station), 8.5, True)
    if row == 2:
        for year in (1998, 2003, 2008, 2013, 2018, 2024):
            idx = next(i for i, (yy, mm) in enumerate(base.MONTHS) if yy == year and mm == 1) if year != 1998 else 0
            x = base.map_x(idx, left, width)
            base.draw_pdf_text(c, x, bottom - 14, str(year), 8, base.INK, False, "center")
    line_series_pdf(base, c, data, station, depth, left, bottom, width, height)


def draw_panel_png(base, d, data, station, depth, row, col, left, bottom, width, height):
    y_min, y_max, y_ticks = ROW_RANGES[station]
    base.Y_MIN, base.Y_MAX = y_min, y_max
    xy = (
        round(left * base.SCALE), round((base.PAGE_H - bottom - height) * base.SCALE),
        round((left + width) * base.SCALE), round((base.PAGE_H - bottom) * base.SCALE),
    )
    d.rectangle(xy, fill=base.rgb("#FFFFFF"), outline=base.rgb(base.SPINE), width=max(2, round(0.7 * base.SCALE)))
    if col == 0:
        for tick in y_ticks:
            y = base.map_y(tick, bottom, height)
            base.draw_png_text(d, left - 8, y - 3, f"{tick:g}", 8, base.INK, False, "right")
        vertical_png_text(base, d, left - 27, bottom + height / 2, "Salinity (PSU)", 8)
    if row == 0:
        base.draw_png_text(d, left + width / 2, bottom + height + 10, f"{int(depth)} m", 10, base.INK, True, "center")
    if col == 0:
        vertical_png_text(base, d, left - 54, bottom + height / 2, base.station_label(station), 8.5, True)
    if row == 2:
        for year in (1998, 2003, 2008, 2013, 2018, 2024):
            idx = next(i for i, (yy, mm) in enumerate(base.MONTHS) if yy == year and mm == 1) if year != 1998 else 0
            x = base.map_x(idx, left, width)
            base.draw_png_text(d, x, bottom - 14, str(year), 8, base.INK, False, "center")
    line_series_png(base, d, data, station, depth, left, bottom, width, height)


def render(base, data):
    base.PAGE_W, base.PAGE_H = 12.0 * 72.0, 9.0 * 72.0
    base.DEPTHS = (20.0, 40.0)
    left_margin, right_margin = 100.0, 20.0
    top_margin, bottom_margin = 42.0, 62.0
    gap_x, gap_y = 18.0, 18.0
    plot_width = (base.PAGE_W - left_margin - right_margin - gap_x) / 2
    plot_height = (base.PAGE_H - top_margin - bottom_margin - 2 * gap_y) / 3

    c = base.canvas.Canvas(str(PDF_PATH), pagesize=(base.PAGE_W, base.PAGE_H), pageCompression=1)
    c.setTitle("Supplementary Figure S1 - PIRATA salinity observation availability")
    c.setAuthor("Atlantic TS Reconstruction")
    c.setFillColor(base.white)
    c.rect(0, 0, base.PAGE_W, base.PAGE_H, fill=1, stroke=0)
    for row, station in enumerate(base.STATIONS):
        bottom = base.PAGE_H - top_margin - (row + 1) * plot_height - row * gap_y
        for col, depth in enumerate(base.DEPTHS):
            left = left_margin + col * (plot_width + gap_x)
            draw_panel_pdf(base, c, data, station, depth, row, col, left, bottom, plot_width, plot_height)
    legend_x, legend_y = base.PAGE_W / 2 - 50, 22
    base.draw_line_pdf(c, legend_x, legend_y + 4, legend_x + 28, legend_y + 4, base.BLUE, 1.2)
    c.setFillColor(base.HexColor(base.BLUE))
    c.circle(legend_x + 14, legend_y + 4, 1.8, fill=1, stroke=0)
    base.draw_pdf_text(c, legend_x + 38, legend_y, "Observed salinity", 9, base.INK)
    c.showPage()
    c.save()

    image = base.Image.new("RGB", (round(base.PAGE_W * base.SCALE), round(base.PAGE_H * base.SCALE)), base.rgb("#FFFFFF"))
    d = base.ImageDraw.Draw(image)
    for row, station in enumerate(base.STATIONS):
        bottom = base.PAGE_H - top_margin - (row + 1) * plot_height - row * gap_y
        for col, depth in enumerate(base.DEPTHS):
            left = left_margin + col * (plot_width + gap_x)
            draw_panel_png(base, d, data, station, depth, row, col, left, bottom, plot_width, plot_height)
    legend_x, legend_y = base.PAGE_W / 2 - 50, 22
    d.line(
        (legend_x * base.SCALE, (base.PAGE_H - legend_y - 4) * base.SCALE,
         (legend_x + 28) * base.SCALE, (base.PAGE_H - legend_y - 4) * base.SCALE),
        fill=base.rgb(base.BLUE), width=max(2, round(1.2 * base.SCALE)),
    )
    px, py, radius = (legend_x + 14) * base.SCALE, (base.PAGE_H - legend_y - 4) * base.SCALE, 1.8 * base.SCALE
    d.ellipse((px - radius, py - radius, px + radius, py + radius), fill=base.rgb(base.BLUE))
    base.draw_png_text(d, legend_x + 38, legend_y, "Observed salinity", 9, base.INK)
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(PNG_PATH, format="PNG", dpi=(base.PNG_DPI, base.PNG_DPI), optimize=True)
    return plot_width / plot_height


def main():
    base = load_base()
    base.register_fonts()
    data = base.read_native_salinity()
    ratio = render(base, data)
    retained = sum(
        value is not None
        for (_station, depth, _month), value in data.items()
        if depth in (20.0, 40.0)
    )
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Exact native salinity points retained for 20/40 m: {retained}")
    print(f"Panel width/height ratio: {ratio:.3f}")


if __name__ == "__main__":
    main()
