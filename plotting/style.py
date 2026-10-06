# Shared style and helpers for all plotting scripts.
# Keeps figure size, fonts, colors and the roofline value in ONE place,
# so every figure looks consistent and no number is hard-coded twice.

import os
import matplotlib.pyplot as plt

# --- paths (relative to this file, so scripts work from any cwd) ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")

# --- physical constant for the GPU roofline (V100) ---
# 900 GB/s peak HBM2 bandwidth / ~288 bytes per lattice update -> ~3100 MLUPS.
ROOFLINE_MLUPS = 3100.0

# --- consistent colors: one color per problem size / layout ---
COLORS = {
    "512": "#1f77b4",
    "1024": "#d62728",
    "2048": "#2ca02c",
    "SoA": "#1f77b4",
    "AoS": "#d62728",
}

# Distinct marker per series too, so the figures stay readable when printed in
# black-and-white (color alone is not enough).
MARKERS = {
    "512": "o",
    "1024": "s",
    "2048": "^",
}
IDEAL_COLOR = "#888888"   # gray dashed for ideal / reference lines
ROOFLINE_COLOR = "#9467bd"

FIGSIZE = (6.0, 4.0)      # same size for every figure


def setup():
    """Apply the global matplotlib style. Call once at the top of each script."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": FIGSIZE,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })


def data_path(name):
    """Full path to a CSV file in data/."""
    return os.path.join(DATA_DIR, name)


def save(fig, name):
    """Save a figure as vector PDF in figures/ (created if missing)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)
