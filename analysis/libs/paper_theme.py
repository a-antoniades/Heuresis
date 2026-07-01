"""Shared figure theme for paper-grade NeurIPS plots.

Provides a strategy-keyed color palette (DeepMind Alpha-series tones) and an
``apply_paper_rcparams()`` helper that switches matplotlib to print-friendly
defaults. Importable from anywhere under ``analysis/`` and from the analysis
skill via ``sys.path.insert(0, str(RESEARCH_AGENT_ROOT / "analysis"))``.

Usage:
    from libs.paper_theme import STRATEGY_COLORS, color_for, apply_paper_rcparams
    apply_paper_rcparams()
    ax.plot(x, y, color=color_for("MAP-Elites"))
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Register bundled Arimo (Apache 2.0, metric-compatible Arial replacement)
# at import time so apply_paper_rcparams() can pin Arimo as the body font on
# machines that lack Microsoft Arial.
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
if _FONT_DIR.is_dir():
    for _ttf in _FONT_DIR.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(_ttf))
        except RuntimeError:
            pass

# Strategy palette. Keys are canonical labels; ``color_for()`` normalizes
# common variants (case, separators, "search" suffix) before lookup.
#
# Active palette: Scheme Q - Island-search Material palette (positional).
# Lifted from qd/islands/plotting.py:_island_colors so cross-figure
# consistency holds: when an island-internal figure and a strategy-
# comparison figure share a paper, linear-blue here is the same color as
# island-0-blue in the islands subfigure. Material Design with magenta
# for MAP-Elites (its own identity, not teal/linear's cool sibling).
STRATEGY_COLORS: dict[str, str] = {
    "linear":     "#2196F3",  # Material Blue
    "MAP-Elites": "#E91E63",  # Material Pink/Magenta
    "Island":     "#4CAF50",  # Material Green
    "omni-epic":  "#FF9800",  # Material Orange
    "curiosity":  "#9C27B0",  # Material Purple
    "go-explore": "#00BCD4",  # Material Cyan
}

# Alternative palette: Tailwind 500 (Scheme G).
# STRATEGY_COLORS = {
#     "linear":     "#3B82F6", "MAP-Elites": "#EAB308", "Island":     "#10B981",
#     "omni-epic":  "#F97316", "curiosity":  "#A855F7", "go-explore": "#EC4899",
# }

# Alternative palette: Material Design (previous active). Vivid but
# MAP-Elites=teal pulls into linear's cool family.
# STRATEGY_COLORS = {
#     "linear":         "#2196F3", "MAP-Elites": "#009688", "Island": "#4CAF50",
#     "omni-epic":      "#FF9800", "curiosity":  "#9C27B0", "go-explore": "#EC4899",
# }
#
# Alternative palette: muted DeepMind Alpha-series tones.
# STRATEGY_COLORS = {
#     "linear":         "#1A73C2",  # DM blue
#     "MAP-Elites":     "#0E8B85",  # DM teal
#     "Island":         "#5DA271",  # sage green
#     "omni-epic":      "#E8743B",  # burnt orange
#     "curiosity":      "#7B5EA7",  # DM purple
#     "go-explore":     "#B194D2",  # mauve
# }
#
# Alternative palette: Scheme P (Tailwind lightness-stratified, fully CB-safe,
# min ΔE 12.98 across all CVD types). Use this if colorblind safety is a hard
# requirement.
# STRATEGY_COLORS = {
#     "linear":     "#0284C7", "MAP-Elites": "#67E8F9", "Island":     "#065F46",
#     "omni-epic":  "#FCD34D", "curiosity":  "#6D28D9", "go-explore": "#FB7185",
# }

# Supporting palette (text, grid, references).
TEXT_COLOR     = "#1F2937"  # charcoal
GRID_COLOR     = "#D1D5DB"  # gray-300 (slightly darker than gray-200)
BASELINE_COLOR = "#6B7280"  # slate

_FALLBACK_CYCLE = [
    "#1A73C2", "#E8743B", "#0E8B85", "#7B5EA7",
    "#5DA271", "#B194D2", "#C4453B", "#8C6D31",
]


def _normalize(label: str) -> str:
    """Map a free-form label to the canonical key used in ``STRATEGY_COLORS``.

    Handles case, separator variants (``map_elites`` / ``mapelites`` /
    ``map-elites``), trailing ``"search"``/``"Search"``, and the ``"-plus"``
    suffix preserved on curiosity variants.
    """
    s = label.strip().lower()
    s = re.sub(r"\s+search\b", "", s)
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)

    aliases = {
        "linearsearch": "linear",
        "mapelites": "MAP-Elites",
        "map-elites": "MAP-Elites",
        "cell-targeted-mapelites": "MAP-Elites",
        "cell-targeted-map-elites": "MAP-Elites",
        "island": "Island",
        "islands": "Island",
        "omni-epic": "omni-epic",
        "omniepic": "omni-epic",
        "omni": "omni-epic",
        "curiosity": "curiosity",
        "go-explore": "go-explore",
        "goexplore": "go-explore",
        "go_explore": "go-explore",
        # curiosity-plus is the legacy 6th-strategy slot; the project's
        # final 6 categories use go-explore instead, so route the old
        # cache labels to the same color slot for backward compatibility.
        "curiosity-plus": "go-explore",
        "curiosityplus": "go-explore",
        "linear": "linear",
        # Display-label aliases — figures.py:label_for() returns display
        # names that get passed back into color_for, so route them too.
        "greedy": "linear",
    }
    return aliases.get(s, label)


def color_for(label: str, *, fallback_index: int | None = None) -> str:
    """Return the canonical color for a strategy label.

    If ``label`` doesn't match any known strategy, falls back to a slot in the
    cycle (``fallback_index`` if provided, else hashed) so unknown strategies
    still render distinctly without crashing.
    """
    key = _normalize(label)
    if key in STRATEGY_COLORS:
        return STRATEGY_COLORS[key]
    idx = fallback_index if fallback_index is not None else (hash(label) % len(_FALLBACK_CYCLE))
    return _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)]


# ---------------------------------------------------------------------------
# Display labels — what shows up in legends, axes, table headers.
# Separate from STRATEGY_COLORS keys (canonical normalized form) so we can
# rename in figures without touching the color palette logic.
# ---------------------------------------------------------------------------
STRATEGY_DISPLAY: dict[str, str] = {
    "linear":     "Greedy",
    "MAP-Elites": "MAP-Elites",
    "Island":     "Islands",
    "omni-epic":  "Omni",
    "curiosity":  "Curiosity",
    "go-explore": "Go-Explore",
}


# Canonical display order for paper figures. All comparison plots sort
# strategies by this index regardless of input cache order.
STRATEGY_ORDER: list[str] = [
    "linear",      # Greedy
    "Island",      # Islands
    "MAP-Elites",
    "go-explore",  # Go-Explore
    "omni-epic",   # Omni
    "curiosity",
]


def order_index_for(label: str) -> int:
    """Return the canonical-order index for a strategy label.

    Unknown labels sort to the end (preserving stable insertion order
    among themselves via the caller's tiebreaker).
    """
    key = _normalize(label)
    try:
        return STRATEGY_ORDER.index(key)
    except ValueError:
        return len(STRATEGY_ORDER) + 1


def display_label_for(label: str) -> str:
    """Return the display name for a strategy label.

    Normalizes via the same alias map as ``color_for`` then looks up
    ``STRATEGY_DISPLAY``. Falls back to a Title-Cased version of the
    canonical key if no display override is registered.
    """
    key = _normalize(label)
    if key in STRATEGY_DISPLAY:
        return STRATEGY_DISPLAY[key]
    # Reasonable default: title-case the key, replacing dashes with spaces.
    return key.replace("-", " ").title() if key else label


def apply_paper_rcparams() -> None:
    """Install NeurIPS-friendly matplotlib defaults: serif body, sans titles,
    smaller font sizes, vector-friendly DPI, subtle grid, no top/right spines.
    """
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "savefig.dpi":       300,
        "figure.dpi":        150,

        # Typography - Arial sans-serif throughout, no bold weights anywhere.
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arimo", "Arial", "Helvetica", "DejaVu Sans"],
        "font.weight":       "normal",
        "mathtext.fontset":  "custom",
        "mathtext.default":  "regular",   # math matches body font, no italic
        "mathtext.rm":       "Arimo",
        "mathtext.it":       "Arimo:italic",
        "mathtext.bf":       "Arimo:bold",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "normal",
        "axes.labelweight":  "normal",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.titleweight": "normal",

        # Axes - full box (all four spines), Nature-style inward ticks.
        "axes.edgecolor":    TEXT_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.spines.bottom": True,
        "axes.spines.left":   True,
        "axes.linewidth":    0.9,

        # Tick marks - outward (matplotlib / NeurIPS / modern ML default;
        # inward is Nature/Science convention but conflicts with data near
        # the axes in violin/fitness plots).
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "xtick.minor.size":  2,
        "ytick.minor.size":  2,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.top":         False,   # ticks only on the data axes
        "ytick.right":       False,

        # Grid (defaults applied to "both" axes; per-plot code can override)
        "axes.grid":         True,
        "axes.grid.axis":    "both",
        "axes.grid.which":   "major",
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "grid.alpha":        0.8,
        "grid.linestyle":    "--",

        # Lines / markers
        "lines.linewidth":   1.6,
        "lines.markersize":  6,
        "patch.linewidth":   0.6,

        # Legend
        "legend.frameon":    True,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  GRID_COLOR,
        "legend.fancybox":   False,

        # Misc
        "axes.unicode_minus": False,    # proper Unicode minus sign

        # Vector output for paper figures
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })
