"""Render Figure 6 station-only Salinity reconstruction maps."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_paper_figure5_temperature_geographic_reconstruction import _render
DATA=Path('artifacts/paper/figure_data/figure6_salinity_geographic_reconstruction.csv')
OUT=Path('artifacts/paper/figures/Figure6_salinity_geographic_reconstruction')
def plot_figure(root:Path): return _render(root,'Salinity',DATA,OUT,'Salinity (source field units)')
if __name__=='__main__': plot_figure(Path(__file__).resolve().parents[1])
