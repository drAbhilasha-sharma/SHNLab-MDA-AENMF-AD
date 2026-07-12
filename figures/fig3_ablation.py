"""
Figure 3 - Ablation study
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

fig = plt.figure(figsize=(13.5, 9.7))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.4, height_ratios=[1.15, 1])
cols = set3(12)

conditions = ["Full model", "Remove DAE", "Remove IPS", "Remove DGIP", "Remove GAE",
              "SNF\u2192Arith. mean", "Remove APS", "Remove NMF", "Remove DSS", "Remove DSIE",
              "IPS-only", "APS-only"]
lodo_auc = [0.782, 0.731, 0.731, 0.723, 0.753, 0.742, 0.763, 0.762, 0.771, 0.776, 0.751, 0.683]
delta_lodo = [0, -0.051, -0.051, -0.059, -0.029, -0.04, -0.019, -0.02, -0.011, -0.006, -0.031, -0.099]
cv_auc = [0.9994, 0.9865, 0.9932, 0.9812, 0.9912, 0.9934, 0.9967, 0.9934, 0.9978, 0.9983, 0.9845, 0.9701]
prec10 = [0.7, 0.6, 0.6, 0.58, 0.65, 0.63, 0.65, 0.68, 0.69, 0.7, 0.65, 0.55]

order = np.argsort(delta_lodo)
oc = [conditions[i] for i in order if conditions[i] != "Full model"]
od = [delta_lodo[i] for i in order if conditions[i] != "Full model"]

# ---------- Panel A: horizontal lollipop of delta LODO AUC ----------
ax = fig.add_subplot(gs[0, :])
sev_colors = []
for d in od:
    if abs(d) >= 0.04: sev_colors.append(cols[4])
    elif abs(d) >= 0.015: sev_colors.append(cols[1])
    else: sev_colors.append(cols[9])
y = lollipop_h(ax, oc, od, sev_colors, xlabel="\u0394 MS LODO AUC (vs. full model, 0.782)", markersize=13)
for yi, d in zip(y, od):
    ax.text(d - 0.006 if d < 0 else d + 0.006, yi, f"{d:+.3f}", va="center",
            ha="right" if d < 0 else "left", fontsize=8)
ax.axvline(0, color=DARK, linewidth=1)
ax.set_title("Ablation impact on MS cross-disease generalisation (LODO), ranked by severity")
legend_els = [Patch(facecolor=cols[4], edgecolor=DARK, label="Critical (|\u0394|\u2265 0.04)"),
              Patch(facecolor=cols[1], edgecolor=DARK, label="Moderate (0.015\u2264|\u0394|<0.04)"),
              Patch(facecolor=cols[9], edgecolor=DARK, label="Minor (|\u0394|<0.015)")]
ax.legend(handles=legend_els, frameon=False, loc="lower right", fontsize=8)
ax.set_xlim(-0.115, 0.02)
style_axes(ax); panel_label(ax, "A", x=-0.07, y=1.08)

# ---------- Panel B: slope chart - MS LODO AUC vs CV AUC per key condition ----------
ax = fig.add_subplot(gs[1, 0])
key = ["Full model", "Remove DAE", "Remove IPS", "Remove DGIP", "Remove GAE", "SNF\u2192Arith. mean", "APS-only"]
idx = [conditions.index(k) for k in key]
lv = [lodo_auc[i] for i in idx]
rv = [cv_auc[i] for i in idx]
kcolors = set3(len(key))
slope_chart(ax, key, lv, rv, "MS LODO AUC", "5-fold CV AUC", colors=kcolors)
for k, l, r, c in zip(key, lv, rv, kcolors):
    ax.text(-0.06, l, k, ha="right", va="center", fontsize=7.3)
ax.set_ylim(0.65, 1.02)
ax.set_title("LODO vs. inflated CV AUC per condition")
style_axes(ax); panel_label(ax, "B")

# ---------- Panel C: horizontal lollipop of Precision@10 ----------
ax = fig.add_subplot(gs[1, 1])
order2 = np.argsort(prec10)
oc3 = [conditions[i] for i in order2]
op3 = [prec10[i] for i in order2]
lollipop_h(ax, oc3, op3, set3(len(oc3)), xlabel="Precision@10 (MS LODO)", markersize=10)
ax.set_xlim(0, 0.85)
ax.tick_params(axis="y", labelsize=8)
ax.set_title("Top-10 precision by ablation condition")
style_axes(ax); panel_label(ax, "C")

fig.suptitle("Figure 3. Module ablation study identifying the architectural drivers of MDA-AENMF-AD's\ncross-disease generalisation performance",
             fontsize=11.5, fontweight="bold", y=1.02)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure3_Ablation_Study.png")
