"""Shared plot style for the Part 2 / Part 3 blog figures.

House style: serif (TeX Gyre Pagella / Palatino), white ground, black-edged marks, near-black ink
on a soft grey grid, bold sentence-case titles. The one source of truth for the per-STEP
categorical colors, so a given step is the same hue in every figure that shows all four steps
(validated with the dataviz skill's palette validator: the blue/orange/aqua/magenta order passes
the adjacent-pair CVD + lightness gates, with the direct labels + legend the figures already carry
as the required secondary encoding).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

# --- typography: serif stack (Palatino-family) ------------------------------------------
for _fp in (os.path.expanduser("~/.fonts/texgyrepagella-regular.otf"),
            os.path.expanduser("~/.fonts/texgyrepagella-bold.otf")):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"]

# --- ink + surface --------------------------------------------------------------------------
INK = "#1A211E"      # near-black primary ink (text, edges, error bars)
GRID = "#D5DCD7"     # soft grey-green grid / spines
MUTED = "#6B726E"    # secondary label ink
SURFACE = "white"

# --- per-step categorical palette (fixed; never reordered) ----------------------------------
STEP_COLORS = {
    "Step 1": "#2a78d6",   # blue
    "Step 2": "#eb6834",   # orange
    "Step 3": "#1baf7a",   # aqua
    "Step 4": "#e87ba4",   # magenta
}
INCIDENT = INK           # whole-incident aggregate: near-black, drawn dashed to stand apart


def apply_axes(fig, ax, *, grid_axis="y"):
    """The shared chrome: white ground, top/right spines off, left/bottom in grid-grey, zero-length
    ticks in ink, a recessive grid on `grid_axis` ('y', 'x', 'both', or None), marks above it."""
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(length=0, colors=INK)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def title(ax, text, **kw):
    """Bold, sentence-case title in ink."""
    ax.set_title(text, fontsize=17, fontweight="bold", color=INK, pad=10, **kw)
