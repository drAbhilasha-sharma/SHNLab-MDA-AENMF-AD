"""
Figure 5 - MS top-20 predictions
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

fig = plt.figure(figsize=(14, 9.8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.55, width_ratios=[1.35, 1])
cols = set3(12)

metabolites = ["D-Glucose","L-Valine","Formic acid","Citric acid","Hippuric acid","L-Lactic acid",
    "GABA","Taurine","Choline","Propionic acid","L-Alanine","Hypoxanthine","L-Phenylalanine",
    "L-Tryptophan","\u03b1-D-Glucose","L-Methionine","Butyric acid","L-Proline","L-Leucine","L-Threonine"]
scores = [0.3479,0.2821,0.1738,0.1262,0.1248,0.1229,0.1228,0.0878,0.0847,0.0785,0.0772,0.0762,
    0.0699,0.0688,0.0674,0.0657,0.0655,0.0632,0.0613,0.0579]
theme = ["Energy metabolism","BCAA catabolism","One-carbon cycle","Energy metabolism","Gut microbiome/SCFA",
    "Energy metabolism","CNS neurotransmitter","CNS neurotransmitter","CNS neurotransmitter","Gut microbiome/SCFA",
    "Other","Purine catabolism","Aromatic AA","Aromatic AA","Energy metabolism","One-carbon cycle",
    "Gut microbiome/SCFA","Other","BCAA catabolism","Gut microbiome/SCFA"]
validation = (["Literature-supported"]*7 + ["No literature support found"] +
              ["Literature-supported"]*11 + ["Probable"])

theme_list = ["Energy metabolism","Gut microbiome/SCFA","CNS neurotransmitter","BCAA catabolism",
              "One-carbon cycle","Purine catabolism","Aromatic AA","Other"]
theme_color = {t: cols[i] for i, t in enumerate(theme_list)}

# ---------- Panel A: horizontal lollipop, top-20, colored by theme ----------
ax = fig.add_subplot(gs[:, 0])
bar_colors = [theme_color[t] for t in theme]
y = lollipop_h(ax, metabolites, scores, bar_colors, xlabel="MS LODO association score", markersize=12)
for yi, v in zip(y, validation):
    if v == "Probable":
        ax.scatter([-0.012], [yi], marker="*", s=90, color=cols[1], edgecolor=DARK, linewidth=0.6, zorder=3, clip_on=False)
    elif v == "No literature support found":
        ax.scatter([-0.012], [yi], marker="x", s=70, color=cols[3], linewidth=1.8, zorder=3, clip_on=False)
ax.set_xlim(-0.02, 0.39)
ax.set_title("Top-20 predicted MS-associated metabolites")
theme_handles = [Patch(facecolor=theme_color[t], edgecolor=DARK, label=t) for t in theme_list]
leg1 = ax.legend(handles=theme_handles, frameon=False, fontsize=7.3, loc="lower right",
                  title="Pathway theme", title_fontsize=7.6)
ax.add_artist(leg1)
status_handles = [plt.Line2D([0],[0], marker="*", color="w", markerfacecolor=cols[1],
               markeredgecolor=DARK, markersize=10, label="Probable\n(not lit.-supported)"),
               plt.Line2D([0],[0], marker="x", color=cols[3], markersize=9, linewidth=0,
               markeredgewidth=1.8, label="No literature\nsupport found")]
ax.legend(handles=status_handles, frameon=False, fontsize=7.3, loc="center right", bbox_to_anchor=(1.0, 0.30))
style_axes(ax); panel_label(ax, "A", x=-0.20)

# ---------- Panel B: donut - validation status ----------
ax = fig.add_subplot(gs[0, 1])
vc = {"Literature-supported": validation.count("Literature-supported"),
      "Probable": validation.count("Probable"),
      "No literature support found": validation.count("No literature support found")}
wedges, texts, autotexts = ax.pie(vc.values(), labels=None,
       autopct=lambda p: f"{p:.0f}%\n(n={round(p*20/100)})",
       colors=[cols[9], cols[1], cols[3]], startangle=90, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
       pctdistance=0.78, textprops={"fontsize":8.5})
ax.legend(wedges, vc.keys(), frameon=False, fontsize=8, loc="center", bbox_to_anchor=(0.5,-0.15), ncol=1)
ax.set_title("Literature validation status\n(Top-20 MS predictions; Table S11)")
panel_label(ax, "B", x=-0.15)

# ---------- Panel C: pathway theme % of top-20 — treemap ----------
ax = fig.add_subplot(gs[1, 1])
themes9 = ["Energy metabolism","Gut microbiome/SCFA","CNS neurotransmitter","BCAA catabolism",
           "One-carbon cycle","Aromatic AA","Kynurenine/IDO1","Purine catabolism"]
pct = [20,20,15,10,10,10,5,5]
tm_colors = [theme_color.get(t, cols[7]) for t in themes9]
order = np.argsort(pct)[::-1]
treemap(ax, [themes9[i].replace(" ", "\n") for i in order], [pct[i] for i in order],
        [tm_colors[i] for i in order], label_fontsize=7.5)
ax.set_title("Pathway theme composition\n(% of top-20 predictions)")
panel_label(ax, "C", x=-0.05, y=1.12)

fig.suptitle("Figure 5. Landscape of top-ranked, literature-validated metabolite predictions for multiple sclerosis\n(MS held out entirely during training; LODO evaluation)",
             fontsize=11.5, fontweight="bold", y=1.02)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure5_MS_Top20_Predictions.png")
