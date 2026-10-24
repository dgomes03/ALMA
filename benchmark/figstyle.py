"""Shared figure style for thesis plots.

Reproduces the visual language of the inspiration paper (arXiv 2604.01193):
rounded light-gray panel cards, white hairline gridlines, circled panel
letters, a shared legend strip at the bottom, SF Pro typography (matching
the thesis fontspec setup).

Palette: 4 categorical hues validated for CVD separation (Machado 2009,
worst all-pairs deltaE 18.0) and >=3:1 contrast on the card surface.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch, PathPatch

# --- color tokens ---------------------------------------------------------
CARD = "#f5f5f7"          # panel card surface
PAGE = "#ffffff"          # page background
GRID = "#e3e3e8"          # gridlines on the card (gray, hairline)
INK = "#1d1d1f"           # primary text
INK_2 = "#6e6e73"         # secondary text (axis titles)
INK_3 = "#86868b"         # muted text (ticks)
BADGE = "#e8e8ed"         # panel-letter circle

# Fixed series order — assign in sequence, never cycle.
SERIES = {
    "bm25":          ("Sparse (BM25)",            "#bf4e2c"),
    "dense":         ("Dense (FAISS)",   "#3d6eb5"),
    "hybrid":        ("Hybrid (RRF)",    "#1f8a70"),
    "hybrid_rerank": ("Hybrid + rerank", "#b48507"),
}

# One base hue per dataset: red, blue, green, amber, purple, pink, olive,
# slate — assigned in order as datasets are added. Within a dataset, methods
# fade from the saturated base to a paler tint of the same hue.
DATASET_HUES = ["#bf4e2c", "#3d6eb5", "#1f8a70", "#b48507",
                "#8250c4", "#b8447c", "#5b7d2a", "#54708c"]
MAX_FADE = 0.55  # tint factor of the least-saturated bar (1.0 would be white)


def tint(hex_color, factor):
    """Mix a color toward white: 0 = unchanged, 1 = white."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: round(c + (255 - c) * factor)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def method_colors(hue_idx, methods):
    """{method: color} — one dataset hue, saturated -> pale in series order."""
    base = DATASET_HUES[hue_idx % len(DATASET_HUES)]
    n = max(len(methods) - 1, 1)
    return {m: tint(base, MAX_FADE * j / n) for j, m in enumerate(methods)}

_FONT_CANDIDATES = [
    "/Library/Fonts/SF-Pro.ttf",
    os.path.expanduser("~/Library/Fonts/SF-Pro.ttf"),
]
_MONO_CANDIDATES = [
    "/Library/Fonts/SF-Mono-Regular.otf",
    "/Library/Fonts/SF-Mono-Bold.otf",   # bold face for best-value highlights
    "/Library/Fonts/SF-Mono-Heavy.otf",  # heavy face (weight="heavy")
    os.path.expanduser("~/Library/Fonts/SF-Mono-Regular.otf"),
    "/System/Applications/Utilities/Terminal.app/Contents/Resources/Fonts/SF-Mono-Regular.otf",
]

# Numbers (ticks, value labels) wear the mono face; set by _register_fonts().
MONO = ["Menlo", "DejaVu Sans Mono"]


def _add_fonts(paths):
    families = []
    for path in paths:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                families.append(font_manager.FontProperties(fname=path).get_name())
            except Exception:
                pass
    return families


def _register_fonts():
    # Mutate MONO in place: consumers do `from figstyle import MONO`, which
    # binds the list object itself — rebinding here would leave them stale.
    mono_families = _add_fonts(_MONO_CANDIDATES)
    MONO[:0] = [f for f in dict.fromkeys(mono_families) if f not in MONO]
    families = _add_fonts(_FONT_CANDIDATES)
    # De-duplicate, keep order, then append fallbacks
    seen = []
    for f in families + ["SF Pro Display", "SF Pro", "Helvetica Neue", "Arial"]:
        if f not in seen:
            seen.append(f)
    return seen


_NUMERIC = set("0123456789.,%+-x ")


def mono_numeric_ticks(fig):
    """Set every purely numeric tick label in the figure to the mono face."""
    for ax in fig.axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            text = label.get_text()
            if text and all(c in _NUMERIC for c in text):
                label.set_fontfamily(MONO)


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _register_fonts(),
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "text.color": INK,
        "axes.edgecolor": "none",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 9,
        "axes.labelpad": 3,
        "legend.fontsize": 9,
        "axes.linewidth": 0.0,
        "pdf.fonttype": 42,  # embed TrueType so text stays selectable in the PDF
    })


def style_axes(ax):
    """Recessive card axes: no spines, white hairline grid, ticks as muted text."""
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.tick_params(which="both", length=0, pad=4)
    ax.set_axisbelow(True)


def card_panels(fig, n_panels, *, left=0.006, right=0.994, top=0.985,
                card_bottom=0.17, gap=0.014, pad_l=0.062, pad_r=0.016,
                pad_t=0.045, pad_b=0.135, widths=None):
    """Lay out `n_panels` rounded cards side by side, each holding one axes.

    Returns (axes_list, cards_list) where cards are (x0, y0, w, h) in
    figure coordinates — used to anchor the panel-letter badges.
    """
    if widths is None:
        widths = [1.0] * n_panels
    total = sum(widths)
    span = right - left - gap * (n_panels - 1)

    axes, cards = [], []
    x = left
    for i in range(n_panels):
        w = span * widths[i] / total
        card = (x, card_bottom, w, top - card_bottom)
        patch = FancyBboxPatch(
            (card[0], card[1]), card[2], card[3],
            boxstyle="round,pad=0,rounding_size=0.018",
            transform=fig.transFigure, facecolor=CARD, edgecolor="none",
            zorder=0, mutation_aspect=fig.get_figwidth() / fig.get_figheight(),
        )
        fig.add_artist(patch)
        ax = fig.add_axes([
            card[0] + pad_l,
            card[1] + pad_b,
            w - pad_l - pad_r,
            card[3] - pad_t - pad_b,
        ])
        style_axes(ax)
        axes.append(ax)
        cards.append(card)
        x += w + gap
    return axes, cards


def panel_letter(fig, card, letter):
    """Circled panel letter at the bottom-left corner of a card."""
    fig.text(
        card[0] + 0.018, card[1] + 0.052, letter,
        ha="center", va="center", fontsize=8, color="#3a3a3c", weight="bold",
        bbox=dict(boxstyle="circle,pad=0.35", facecolor=BADGE, edgecolor="none"),
    )


def bottom_legend(fig, methods, y=0.045, marker=False, colors=None):
    """Shared horizontal legend strip below the cards (line-key + label)."""
    from matplotlib.lines import Line2D
    colors = colors or {m: SERIES[m][1] for m in methods}
    handles = [
        Line2D([], [], color=colors[m], linewidth=2.5,
               marker="o" if marker else None, markersize=5)
        for m in methods
    ]
    labels = [SERIES[m][0] for m in methods]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=len(methods), frameon=False, handlelength=1.6,
               columnspacing=1.6, handletextpad=0.6)


def rounded_bar(ax, x, height, width, color, radius_px=4, zorder=3):
    """Bar with a 4px-rounded data end and a square baseline."""
    # Convert the pixel radius into data units (needs final limits set first)
    origin = ax.transData.transform((0, 0))
    rx = abs(ax.transData.inverted().transform(origin + [radius_px, 0])[0])
    ry = abs(ax.transData.inverted().transform(origin + [0, radius_px])[1])
    rx = min(rx, width / 2)
    ry = min(ry, height / 2)

    x0, x1, y1 = x - width / 2, x + width / 2, height
    verts = [
        (x0, 0), (x0, y1 - ry), (x0, y1), (x0 + rx, y1),   # left edge + corner
        (x1 - rx, y1), (x1, y1), (x1, y1 - ry),            # top + right corner
        (x1, 0), (x0, 0),
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                           edgecolor="none", zorder=zorder))


def series_line(ax, xs, ys, color, marker_size=5):
    """2px line with surface-ringed markers in the series color."""
    ax.plot(xs, ys, color=color, linewidth=2, zorder=3,
            solid_joinstyle="round", solid_capstyle="round")
    ax.plot(xs, ys, linestyle="none", marker="o", markersize=marker_size,
            markerfacecolor=color, markeredgecolor=CARD, markeredgewidth=1.5,
            zorder=4)
