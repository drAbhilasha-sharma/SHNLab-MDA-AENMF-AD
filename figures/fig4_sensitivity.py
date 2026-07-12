import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(13.5, 8.6))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
cols = set3(12)

def line_panel(ax, x, y, default_idx, xlabel, title, color, logx=False, letter=""):
    ax.plot(x, y, "-o", color=color, markersize=7, linewidth=2, markeredgecolor=DARK, markeredgewidth=0.8, zorder=2)
    ax.scatter([x[default_idx]], [y[default_idx]], s=170, facecolor=cols[9], edgecolor=DARK,
               linewidth=1.3, zorder=3, marker="*", label="Default (optimal)")
    ax.axhline(0.782, color=DARK, linestyle=":", linewidth=0.8, alpha=0.6)
    if logx:
        ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(xlabel); ax.set_ylabel("MS LODO AUC")
    ax.set_title(title)
    ax.set_ylim(0.70, 0.80)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center")
    style_axes(ax); panel_label(ax, letter)

# NMF rank
ax = fig.add_subplot(gs[0,0])
line_panel(ax, [2,4,6,8], [0.753,0.782,0.771,0.750], 1, "NMF rank", "NMF Co-embedding Rank", cols[0], letter="A")

# AE latent dim
ax = fig.add_subplot(gs[0,1])
line_panel(ax, [32,64,128], [0.764,0.782,0.773], 1, "AE latent dimension", "Autoencoder Latent Size", cols[2], letter="B")

# SNF k
ax = fig.add_subplot(gs[0,2])
line_panel(ax, [15,20,25], [0.761,0.782,0.776], 1, "SNF k (neighbours)", "SNF Neighbourhood Size", cols[4], letter="C")

# Dropout
ax = fig.add_subplot(gs[1,0])
line_panel(ax, [0.1,0.3,0.5], [0.779,0.782,0.763], 1, "Dropout rate", "Dropout Rate", cols[6], letter="D")

# Learning rate
ax = fig.add_subplot(gs[1,1])
line_panel(ax, [0.0001,0.001,0.01], [0.771,0.782,0.746], 1, "Learning rate", "Adam Learning Rate", cols[8], logx=True, letter="E")

# Panel F: summary range/robustness - dumbbell/box style showing delta range per parameter
ax = fig.add_subplot(gs[1,2])
params = ["NMF rank", "AE latent\ndim", "SNF k", "Dropout\nrate", "Learning\nrate"]
mins = [0.750-0.782, 0.764-0.782, 0.761-0.782, 0.763-0.782, 0.746-0.782]
maxs = [0.771-0.782, 0.773-0.782, 0.776-0.782, 0.779-0.782, 0.771-0.782]
y = np.arange(len(params))
bar_c = set3(len(params))
for i, (mn, mx, c) in enumerate(zip(mins, maxs, bar_c)):
    ax.plot([mn, mx], [i, i], color=DARK, linewidth=1.4, zorder=1)
    ax.scatter([mn], [i], s=90, color=c, edgecolor=DARK, zorder=2, marker="v")
    ax.scatter([mx], [i], s=90, color=c, edgecolor=DARK, zorder=2, marker="^")
ax.axvline(0, color=cols[9], linewidth=1.6, label="Default (\u0394=0)")
ax.set_yticks(y); ax.set_yticklabels(params, fontsize=8.5)
ax.set_xlabel("\u0394 MS LODO AUC vs. default")
ax.set_title("Robustness Range per Hyperparameter")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
style_axes(ax); panel_label(ax, "F")

fig.suptitle("Figure 4. Hyperparameter sensitivity analysis: MS LODO AUC remains within \u00b10.037 of the reported\noptimum across all tested configurations",
             fontsize=11.5, fontweight="bold", y=1.03)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure4_Hyperparameter_Sensitivity.png")
