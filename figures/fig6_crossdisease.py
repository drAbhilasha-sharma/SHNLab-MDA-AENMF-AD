"""
Figure 6 - Cross-disease comparative analysis
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.42)
cols = set3(12)

metabs = ["D-Glucose","L-Lactic acid","Citric acid","Hippuric acid","L-Valine","L-Tryptophan","Choline","Butyric acid","GABA","Hypoxanthine"]
diseases = ["MS","RA","IBD","T1D","SLE"]
mat = np.array([
 [1.00,0.35,0.36,0.36,0.20,0.20,0.24,0.19,0.35,0.22],
 [0.12,0.92,0.87,0.96,0.08,0.15,0.38,0.14,0.92,0.08],
 [0.45,0.78,0.95,0.72,0.22,0.38,0.86,0.94,0.55,0.66],
 [0.38,0.29,0.44,0.21,0.55,0.42,0.31,0.28,0.18,0.33],
 [0.31,0.18,0.28,0.09,0.14,0.51,0.22,0.11,0.09,0.19],
])

# ---------- Panel A: heatmap ----------
ax = fig.add_subplot(gs[0, :])
im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(np.arange(10)); ax.set_xticklabels(metabs, rotation=38, ha="right", fontsize=8.3)
ax.set_yticks(np.arange(5)); ax.set_yticklabels(diseases, fontsize=9.5)
for i in range(5):
    for j in range(10):
        v = mat[i,j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.55 else DARK)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cbar.set_label("Normalised association score", fontsize=8.5)
ax.set_title("Cross-disease normalised association score matrix (shared top metabolites)")
for spine in ax.spines.values(): spine.set_visible(False)
panel_label(ax, "A", x=-0.06, y=1.14)

# ---------- Panel B: MS overlap with other diseases — slope chart ----------
ax = fig.add_subplot(gs[1, 0])
comp = ["MS vs RA","MS vs IBD","MS vs T1D","MS vs SLE"]
shared = [7,11,6,3]
specific = [13,9,14,17]
pct = [35,55,30,15]
ccolors = set3(4)
slope_chart(ax, comp, shared, specific, "Shared with MS", "MS-specific", colors=ccolors)
for lbl, s, sp, p, c in zip(comp, shared, specific, pct, ccolors):
    ax.text(-0.06, s, lbl, ha="right", va="center", fontsize=8)
    ax.text(1.06, sp, f"{p}% shared", ha="left", va="center", fontsize=7.5, color=c, fontweight="bold")
ax.set_ylabel("Metabolite count (of top-20)")
ax.set_ylim(0, 20)
ax.set_title("MS top-20 overlap with other autoimmune diseases")
style_axes(ax); panel_label(ax, "B")

# ---------- Panel C: radar chart of LODO metrics per disease ----------
ax = fig.add_subplot(gs[1, 1], polar=True)
metrics = ["AUC","AUPR","F1","Prec@10","Prec@20"]
data = {
 "MS":[0.7821,0.6934,0.6500,0.700,0.900],
 "RA":[0.8134,0.7812,0.7188,0.750,0.900],
 "T1D":[0.7643,0.6721,0.6111,0.700,0.850],
 "IBD":[0.9312,0.9187,0.8783,0.900,0.950],
}
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
disease_colors = {"MS":cols[4], "RA":cols[2], "T1D":cols[6], "IBD":cols[9]}
for d, vals in data.items():
    v = vals + vals[:1]
    ax.plot(angles, v, color=disease_colors[d], linewidth=2, label=d)
    ax.fill(angles, v, color=disease_colors[d], alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=8.5)
ax.set_ylim(0,1)
ax.set_title("LODO performance profile by disease", pad=18)
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.28,1.12))
panel_label(ax, "C", x=-0.12, y=1.12)

fig.suptitle("Figure 6. Cross-disease convergence and specificity of predicted metabolite associations across\nfour autoimmune disorders",
             fontsize=11.5, fontweight="bold", y=1.02)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure6_Cross_Disease_Analysis.png")
