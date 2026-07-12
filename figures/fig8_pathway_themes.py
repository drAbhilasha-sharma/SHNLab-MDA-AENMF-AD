"""
Figure 8 - MS pathway theme / mechanistic analysis
Chart types used: horizontal lollipop plot, bubble scatter, pie chart, donut chart.
No solid bar charts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 9.8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.42)
cols = set3(12)

themes = ["Energy metabolism / Glycolysis","Gut microbiome / SCFA","CNS neurotransmitter fingerprint",
          "BCAA catabolism / mTORC1","One-carbon / Methylation cycle","Aromatic AA dysregulation",
          "Kynurenine / IDO1 immunometabolism","Purine catabolism / Oxidative stress"]
n_mets = [4,4,3,2,2,2,1,1]
combined_score = [0.664,0.327,0.295,0.343,0.239,0.139,0.069,0.076]
pct_top20 = [20,20,15,10,10,10,5,5]
ips_driven = ["Partial","No","Yes","Partial","No","Partial","Partial","No"]
tcol = set3(8)

# ---------- Panel A: horizontal lollipop of combined score ----------
ax = fig.add_subplot(gs[0,0])
order = np.argsort(combined_score)
labels = [themes[i] for i in order]
lollipop_h(ax, labels, [combined_score[i] for i in order], [tcol[i] for i in order],
           xlabel="Combined association score", markersize=12)
ax.tick_params(axis="y", labelsize=7.6)
ax.set_title("Combined score by pathway theme")
style_axes(ax); panel_label(ax, "A", x=-0.5)

# ---------- Panel B: N metabolites vs % top-20 — bubble scatter ----------
ax = fig.add_subplot(gs[0,1])
for i in range(8):
    ax.scatter(n_mets[i], pct_top20[i], s=combined_score[i]*1400, color=tcol[i], edgecolor=DARK, linewidth=1, alpha=0.85, zorder=3)
    ax.annotate(themes[i].split(" /")[0], (n_mets[i], pct_top20[i]),
                textcoords="offset points", xytext=(7,4), fontsize=6.8)
ax.set_xlabel("N metabolites in theme"); ax.set_ylabel("% of top-20 predictions")
ax.set_title("Theme size vs. representation\n(bubble area = combined score)")
ax.set_xlim(0.5, 4.7); ax.set_ylim(3, 25)
style_axes(ax); panel_label(ax, "B")

# ---------- Panel C: IPS-driven categories — pie chart ----------
ax = fig.add_subplot(gs[1,0])
from collections import Counter
cat_counts = Counter(ips_driven)
wedges, texts, autotexts = ax.pie(cat_counts.values(), labels=cat_counts.keys(),
       colors=[cols[9], cols[1], cols[4]], startangle=90, explode=[0.03]*len(cat_counts),
       autopct=lambda p: f"{p:.0f}%\n(n={round(p*8/100)})", textprops={"fontsize":8.5},
       wedgeprops=dict(edgecolor="white", linewidth=1.6))
ax.set_title("Recovery via immune pathway similarity (IPS)\nvs. raw metabolite structure (n=8 themes)")
panel_label(ax, "C", x=-0.1)

# ---------- Panel D: % top-20 donut with theme labels ----------
ax = fig.add_subplot(gs[1,1])
wedges, texts, autotexts = ax.pie(pct_top20, colors=tcol, startangle=90,
       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.6),
       autopct=lambda p: f"{p:.0f}%" if p > 6 else "", pctdistance=0.78, textprops={"fontsize":7.5})
ax.legend(wedges, themes, frameon=False, fontsize=6.6, loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.set_title("Composition of top-20 MS\npredictions by pathway theme")
panel_label(ax, "D", x=-0.15)

fig.suptitle("Figure 8. Mechanistic pathway-theme decomposition of the top-20 MS metabolite predictions,\nhighlighting the CNS-specific neurotransmitter fingerprint absent from RA/IBD",
             fontsize=11.5, fontweight="bold", y=1.02)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure8_MS_Pathway_Theme_Analysis.png")
