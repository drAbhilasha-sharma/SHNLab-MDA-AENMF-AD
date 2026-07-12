"""
Figure 2 - Model performance evaluation (CV, LODO, Y-randomisation) 
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(13, 9.2))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)
cols = set3(12)

# ---------- Panel A: 5-fold CV AUC & AUPR — line/marker plot ----------
ax = fig.add_subplot(gs[0, 0])
folds = [1, 2, 3, 4, 5]
auc = [0.998959, 1.0, 0.998933, 1.0, 1.0]
aupr = [0.998932, 1.0, 0.999018, 1.0, 1.0]
ax.plot(folds, auc, "-o", color=cols[0], linewidth=2.2, markersize=9,
        markeredgecolor=DARK, markeredgewidth=1, label="AUC")
ax.plot(folds, aupr, "-D", color=cols[4], linewidth=2.2, markersize=8,
        markeredgecolor=DARK, markeredgewidth=1, label="AUPR")
ax.set_xticks(folds)
ax.set_ylim(0.9985, 1.0008)
ax.set_xlabel("CV fold"); ax.set_ylabel("Score")
ax.set_title("5-fold Stratified Cross-Validation")
ax.legend(frameon=False, loc="lower right", fontsize=8)
style_axes(ax); panel_label(ax, "A")

# ---------- Panel B: LODO AUC/AUPR/F1 per disease — Cleveland dot plot ----------
ax = fig.add_subplot(gs[0, 1])
diseases = ["MS (n=20)", "RA (n=32)", "T1D (n=18)", "IBD (n=337)"]
metrics = {"AUC": [0.7821, 0.8134, 0.7643, 0.9312],
           "AUPR": [0.6934, 0.7812, 0.6721, 0.9187],
           "F1": [0.6500, 0.7188, 0.6111, 0.8783]}
markers = ["o", "D", "^"]
mcolors = [cols[2], cols[4], cols[6]]
y = np.arange(len(diseases))
for (name, vals), mk, c in zip(metrics.items(), markers, mcolors):
    ax.scatter(vals, y, s=95, marker=mk, color=c, edgecolor=DARK, linewidth=1, label=name, zorder=3)
for yi in y:
    ax.plot([0.5, 1.0], [yi, yi], color="lightgray", linewidth=6, zorder=1, solid_capstyle="round")
ax.axvline(0.5, color=DARK, linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_yticks(y); ax.set_yticklabels(diseases)
ax.set_xlim(0.45, 1.0); ax.set_xlabel("Score")
ax.set_title("Leave-One-Disease-Out (LODO)")
ax.legend(frameon=False, loc="lower right", fontsize=8)
style_axes(ax); panel_label(ax, "B")

# ---------- Panel C: Precision@10 vs @20 — slope chart per disease ----------
ax = fig.add_subplot(gs[0, 2])
p10 = [0.700, 0.750, 0.700, 0.900]
p20 = [0.900, 0.900, 0.850, 0.950]
dcolors = [cols[4], cols[2], cols[6], cols[9]]
for lbl, lv, rv, c in zip(diseases, p10, p20, dcolors):
    ax.plot([0, 1], [lv, rv], "-o", color=c, linewidth=2.2, markersize=8,
            markeredgecolor=DARK, markeredgewidth=1, zorder=2)
    ax.text(-0.06, lv, lbl.split(" ")[0], ha="right", va="center", fontsize=7.8)
ax.set_xlim(-0.35, 1.15)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Precision@10", "Precision@20"], fontweight="bold")
ax.set_ylim(0.6, 1.0); ax.set_ylabel("Precision")
ax.set_title("Top-k Precision Gain (LODO)")
style_axes(ax); ax.spines["left"].set_visible(False); ax.tick_params(left=False)
panel_label(ax, "C")

# ---------- Panel D: Y-randomisation null vs real performance — histogram ----------
ax = fig.add_subplot(gs[1, 0])
np.random.seed(42)
yrand_vals = [0.5234, 0.5087, 0.5312]
null_dist = np.random.normal(np.mean(yrand_vals), 0.012, 2000)
ax.hist(null_dist, bins=40, color=cols[7], edgecolor="white", linewidth=0.3, alpha=0.9,
        label="Y-rand null\n(simulated, n=3 runs)")
ax.axvline(np.mean(yrand_vals), color=DARK, linestyle="--", linewidth=1.1)
ax.axvline(0.9994, color=cols[9], linewidth=2.2, label="5-fold CV AUC = 0.9994")
ax.axvline(0.7821, color=cols[4], linewidth=2.2, label="MS LODO AUC = 0.7821")
ax.set_xlim(0.45, 1.03)
ax.set_xlabel("AUC"); ax.set_ylabel("Frequency")
ax.set_title("Y-Randomisation Control")
ax.legend(frameon=False, fontsize=6.8, loc="upper center")
style_axes(ax); panel_label(ax, "D")

# ---------- Panel E: evaluation regime comparison — lollipop chart ----------
ax = fig.add_subplot(gs[1, 1])
cats = ["5-fold CV\n(pooled)", "LODO\n(mean of 4)", "Baseline AUC\n(mean of folds)", "Y-rand\n(mean of 3)"]
vals = [0.9994, np.mean(metrics["AUC"]), np.mean([0.97561, 0.962076, 0.973022, 0.994361, 0.974699]), np.mean(yrand_vals)]
lolli_colors = [cols[9], cols[4], cols[8], cols[7]]
lollipop_v(ax, cats, vals, lolli_colors, ylabel="AUC", ref=0.45, markersize=15)
for xi, v in enumerate(vals):
    ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_ylim(0.45, 1.08)
ax.set_title("Evaluation Regime Comparison")
ax.tick_params(axis="x", labelsize=7.4)
style_axes(ax); panel_label(ax, "E")

# ---------- Panel F: fold composition — stacked-area chart ----------
ax = fig.add_subplot(gs[1, 2])
n_train = [650, 650, 652, 652, 652]
n_test = [164, 164, 162, 162, 162]
stacked_area(ax, folds, {"n_train": n_train, "n_test": n_test}, colors=[cols[10], cols[11]])
ax.set_xticks(folds)
ax.set_xlabel("CV fold"); ax.set_ylabel("Association pairs (n)")
ax.set_title("Fold Composition")
ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
ax.set_ylim(0, 900)
style_axes(ax); panel_label(ax, "F")

fig.suptitle("Figure 2. Systematic performance evaluation of MDA-AENMF-AD across cross-validation, leave-one-disease-out,\nand label-permutation regimes",
             fontsize=11.5, fontweight="bold", y=1.02)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure2_Model_Performance_Evaluation.png")
