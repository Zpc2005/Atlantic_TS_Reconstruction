"""Render the final publication Figure 1 from frozen metadata and WOA23 fields."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Figure1_redesign_preview import render

# Restrained oceanographic ramps matching the manuscript's existing visual
# language: blue/cyan -> green/yellow -> warm red for SST, and blue ->
# cyan/green -> yellow for SSS.  These are visualization palettes only.
SST_CMAP = LinearSegmentedColormap.from_list(
    "figure1_sst_final",
    ["#2A77B5", "#35B7C5", "#9BCB70", "#F0D45B", "#E66B3D"],
    N=256,
)
SSS_CMAP = LinearSegmentedColormap.from_list(
    "figure1_sss_final",
    ["#164D7A", "#268EA5", "#49B58F", "#BBD86A", "#F1D34E"],
    N=256,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts/paper/figures/Figure1_final"


if __name__ == "__main__":
    png, pdf = render(
        output_dir=OUTPUT_DIR,
        png_name="Figure1_final_600dpi.png",
        pdf_name="Figure1_final_vector.pdf",
        sst_cmap=SST_CMAP,
        sss_cmap=SSS_CMAP,
        sst_levels=np.linspace(10.0, 30.0, 101),
        sss_levels=np.linspace(33.0, 38.0, 101),
        land_color="#E6E6E6",
        ocean_facecolor="#EDF5F6",
        draw_grid=False,
        station_size=50.0,
        target_label=True,
        validation_name="station_coordinate_validation.csv",
    )
    print(png)
    print(pdf)
