"""
Figure 9 - Literature validation & model configuration summary
Chart types used: donut chart, lollipop plot, dumbbell chart, lollipop plot (log scale).
No solid bar charts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec
from collections import Counter

fig = plt.figure(figsize=(14, 9.8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.6, wspace=0.45)
cols = set3(12)

# ---------- Panel A: sample-type distribution — donut ----------
ax = fig.add_subplot(gs[0,0])
sample_types = ["Serum"]*7 + ["CSF-inclusive"]*6 + ["Urine-inclusive (non-CSF)"]*3 + ["Brain (MRS)"]*1 + ["Faecal / gut-derived"]*2
sc = Counter(sample_types)
wedges, texts, autotexts = ax.pie(sc.values(), colors=set3(len(sc)), startangle=90,
       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.6),
       autopct=lambda p: f"{p:.0f}%", pctdistance=0.78, textprops={"fontsize":8.5})
ax.legend(wedges, sc.keys(), frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0,0.5))
ax.set_title("Sample type of supporting evidence\n(19 MS literature-supported/probable predictions)")
panel_label(ax, "A", x=-0.1)

# ---------- Panel B: key reference contribution — lollipop ----------
ax = fig.add_subplot(gs[0,1])
pmid_labels = ["Cocco et al. 2016","Kim et al. 2017","Gebregiworgis et al. 2016",
               "Alwahsh et al. 2024","Chinese SCFA cohort study"]
counts = [5, 4, 4, 3, 2]
order = np.argsort(counts)
lollipop_h(ax, [pmid_labels[i] for i in order], [counts[i] for i in order], set3(5), xlabel="MS predictions supported (of top-20)", markersize=13)
ax.tick_params(axis="y", labelsize=7.4)
ax.set_title("Contribution of key literature sources")
style_axes(ax); panel_label(ax, "B", x=-0.55)

# ---------- Panel C: hyperparameter count per module — dumbbell chart ----------
ax = fig.add_subplot(gs[1,0])
modules = ["SNF","DAE","GAE","NMF","MLP","Training","Data"]
n_params = [3,5,4,4,4,8,4]
sens_tested = [2,2,1,1,0,1,0]
dumbbell_h(ax, modules[::-1], sens_tested[::-1], n_params[::-1], cols[9], cols[3])
ax.set_xlabel("Number of hyperparameters")
ax.set_title("Model configuration by module (Table S12)")
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=cols[9], markeredgecolor=DARK, markersize=9, label="Sensitivity-tested"),
           Line2D([0],[0], marker="D", color="w", markerfacecolor=cols[3], markeredgecolor=DARK, markersize=9, label="Total parameters")]
ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
style_axes(ax); panel_label(ax, "C")

# ---------- Panel D: key hyperparameter values — lollipop (log scale) ----------
ax = fig.add_subplot(gs[1,1])
hp_names = ["SNF k","DAE latent","GAE latent","NMF rank","MLP hidden1","MLP hidden2","Batch size","Max epochs"]
hp_vals = [20,64,64,4,64,32,64,200]
x = lollipop_v(ax, hp_names, hp_vals, set3(len(hp_names)), ylabel="Value (log scale)", ref=1, markersize=14)
ax.set_yscale("log")
ax.tick_params(axis="x", labelsize=7.4, rotation=30)
for xi, v in zip(x, hp_vals):
    ax.text(xi, v*1.25, str(v), ha="center", fontsize=8, fontweight="bold")
ax.set_ylim(1, 400)
ax.set_title("Final model hyperparameter values")
style_axes(ax); panel_label(ax, "D")

fig.suptitle("Figure 9. Independent literature support underlying MS predictions and final MDA-AENMF-AD\nmodel configuration (all runs: random_seed = 42)",
             fontsize=11.5, fontweight="bold", y=1.03)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure9_Literature_Validation_and_Model_Config.png")
