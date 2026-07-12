import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

SET3 = plt.get_cmap("Set3").colors  # 12 pastel qualitative colors
# Reorder slightly so muddy yellow (index 1) isn't first
SET3_ORDERED = [SET3[i] for i in [0,3,4,5,6,7,8,9,10,11,2,1]]

def set3(n):
    """Return n colors cycling through Set3 (repeats if n>12)."""
    return [SET3_ORDERED[i % 12] for i in range(n)]

DARK = "#2b2b2b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.edgecolor": DARK,
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "axes.linewidth": 0.9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
})

def panel_label(ax, letter, x=-0.14, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold",
             va="top", ha="left", color=DARK)

def style_axes(ax, top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.tick_params(direction="out", length=3.5, width=0.9)

def save(fig, path, dpi=350):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print("saved:", path)

# ---------------------------------------------------------------
# Reusable non-bar chart primitives
# ---------------------------------------------------------------

def lollipop_h(ax, labels, values, colors, xlabel="", ref=0, markersize=11):
    """Horizontal lollipop (stem + dot) chart - alternative to barh."""
    y = np.arange(len(labels))[::-1]
    for yi, v, c in zip(y, values, colors):
        ax.plot([ref, v], [yi, yi], color=DARK, linewidth=1.4, zorder=1, solid_capstyle="round")
        ax.scatter([v], [yi], s=markersize**2, color=c, edgecolor=DARK, linewidth=1.1, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    if xlabel: ax.set_xlabel(xlabel)
    return y

def lollipop_v(ax, labels, values, colors, ylabel="", ref=0, markersize=13):
    """Vertical lollipop (stem + dot) chart - alternative to bar."""
    x = np.arange(len(labels))
    for xi, v, c in zip(x, values, colors):
        ax.plot([xi, xi], [ref, v], color=DARK, linewidth=1.4, zorder=1, solid_capstyle="round")
        ax.scatter([xi], [v], s=markersize**2, color=c, edgecolor=DARK, linewidth=1.1, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    if ylabel: ax.set_ylabel(ylabel)
    return x

def slope_chart(ax, categories, left_vals, right_vals, left_label, right_label,
                 colors=None, left_x=0, right_x=1):
    """Two-column slope / dumbbell chart connecting paired values per category."""
    n = len(categories)
    colors = colors or set3(n)
    for i, (lv, rv, c) in enumerate(zip(left_vals, right_vals, colors)):
        ax.plot([left_x, right_x], [lv, rv], color=c, linewidth=2.0, zorder=1, alpha=0.85)
        ax.scatter([left_x], [lv], s=90, color=c, edgecolor=DARK, linewidth=1, zorder=2)
        ax.scatter([right_x], [rv], s=90, color=c, edgecolor=DARK, linewidth=1, zorder=2, marker="D")
    ax.set_xlim(left_x - 0.35, right_x + 0.35)
    ax.set_xticks([left_x, right_x]); ax.set_xticklabels([left_label, right_label], fontweight="bold")
    return ax

def dumbbell_h(ax, labels, left_vals, right_vals, left_color, right_color, connector_color=None):
    """Horizontal dumbbell chart: two points per row connected by a line."""
    y = np.arange(len(labels))[::-1]
    connector_color = connector_color or DARK
    for yi, lv, rv in zip(y, left_vals, right_vals):
        ax.plot([lv, rv], [yi, yi], color=connector_color, linewidth=2.0, alpha=0.55, zorder=1)
    ax.scatter(left_vals, y, s=95, color=left_color, edgecolor=DARK, linewidth=1, zorder=2, label="_left")
    ax.scatter(right_vals, y, s=95, color=right_color, edgecolor=DARK, linewidth=1, zorder=2, marker="D", label="_right")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    return y

def treemap(ax, labels, sizes, colors, label_fontsize=8):
    """Simple treemap using squarify."""
    import squarify
    norm_sizes = squarify.normalize_sizes(sizes, 100, 100)
    rects = squarify.squarify(norm_sizes, 0, 0, 100, 100)
    for r, lab, val, c in zip(rects, labels, sizes, colors):
        ax.add_patch(plt.Rectangle((r["x"], r["y"]), r["dx"], r["dy"], facecolor=c,
                                     edgecolor="white", linewidth=2.2))
        if r["dx"] > 10 and r["dy"] > 8:
            ax.text(r["x"] + r["dx"]/2, r["y"] + r["dy"]/2 + 3, lab, ha="center", va="center",
                    fontsize=label_fontsize, fontweight="bold", color=DARK, wrap=True)
            ax.text(r["x"] + r["dx"]/2, r["y"] + r["dy"]/2 - 5, f"{val}%", ha="center", va="center",
                    fontsize=label_fontsize-0.5, color=DARK)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

def bump_chart(ax, categories, series_dict, colors=None):
    """Bump chart: rank trajectories of items across ordered categories.
    series_dict: {item_name: [values across categories]} - ranked internally (1=best)."""
    items = list(series_dict.keys())
    colors = colors or set3(len(items))
    n_cat = len(categories)
    ranks = {}
    for ci in range(n_cat):
        col_vals = [(it, series_dict[it][ci]) for it in items]
        col_vals.sort(key=lambda t: -t[1])
        for rank, (it, _) in enumerate(col_vals, start=1):
            ranks.setdefault(it, []).append(rank)
    for it, c in zip(items, colors):
        ax.plot(range(n_cat), ranks[it], "-o", color=c, linewidth=2.2, markersize=8,
                markeredgecolor=DARK, markeredgewidth=0.9, label=it, zorder=2)
    ax.set_xticks(range(n_cat)); ax.set_xticklabels(categories)
    ax.set_ylim(len(items) + 0.6, 0.4)
    ax.set_ylabel("Rank")
    return ax

def stacked_area(ax, x, series_dict, colors=None, labels=None):
    labels = labels or list(series_dict.keys())
    colors = colors or set3(len(series_dict))
    ax.stackplot(x, *series_dict.values(), colors=colors, labels=labels,
                 edgecolor="white", linewidth=1.2, alpha=0.92)
    return ax

