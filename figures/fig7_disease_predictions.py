"""
Figure 7 - Disease-specific top predictions
Chart types used: bump (rank-trajectory) chart, horizontal lollipop small multiples.
No solid bar charts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

fig = plt.figure(figsize=(13.5, 12.5))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.55, height_ratios=[1.1, 1, 1])
cols = set3(12)

theme_master = ["Energy metabolism","Gut microbiome/SCFA","CNS/Neuro","BCAA","One-carbon",
                 "Oxysterol/Lipid","Immune activation","Purine/Redox","Bile acid","Carotenoid/Antioxidant"]
tcolor = {t: cols[i] for i, t in enumerate(theme_master)}

# ---------- Panel A: bump chart — rank trajectory of shared metabolites across diseases ----------
ax = fig.add_subplot(gs[0, :])
metabs = ["D-Glucose","L-Lactic acid","Citric acid","Hippuric acid","Choline","GABA"]
disease_order = ["MS","RA","IBD","T1D","SLE"]
vals = {
 "D-Glucose":     [1.00,0.12,0.45,0.38,0.31],
 "L-Lactic acid": [0.35,0.92,0.78,0.29,0.18],
 "Citric acid":   [0.36,0.87,0.95,0.44,0.28],
 "Hippuric acid": [0.36,0.96,0.72,0.21,0.09],
 "Choline":       [0.24,0.38,0.86,0.31,0.22],
 "GABA":          [0.35,0.92,0.55,0.18,0.09],
}
bump_colors = set3(len(metabs))
bump_chart(ax, disease_order, vals, colors=bump_colors)
ax.set_title("Rank trajectory of key shared metabolites across autoimmune diseases\n(rank 1 = highest normalised association score within disease)")
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
style_axes(ax); panel_label(ax, "A", x=-0.045, y=1.14)

def lollipop_panel(ax, metabs, scores, themes, title, letter, xmax=None):
    bcolors = [tcolor.get(t, cols[7]) for t in themes]
    lollipop_h(ax, metabs, scores, bcolors, xlabel="Association score", markersize=10)
    if xmax: ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=9.5)
    style_axes(ax); panel_label(ax, letter, x=-0.34)

# MS
ax = fig.add_subplot(gs[1,0])
ms_m = ["D-Glucose","L-Valine","Formic acid","Citric acid","Hippuric acid","L-Lactic acid","GABA","Taurine","Choline","Propionic acid"][::-1]
ms_s = [0.3479,0.2821,0.1738,0.1262,0.1248,0.1229,0.1228,0.0878,0.0847,0.0785][::-1]
ms_t = ["Energy metabolism","BCAA","One-carbon","Energy metabolism","Gut microbiome/SCFA","Energy metabolism","CNS/Neuro","CNS/Neuro","CNS/Neuro","Gut microbiome/SCFA"][::-1]
lollipop_panel(ax, ms_m, ms_s, ms_t, "MS \u2014 Top-10 (LODO, MS held out)", "B", xmax=0.4)

# RA
ax = fig.add_subplot(gs[1,1])
ra_m = ["Hippuric acid","L-Lactic acid","GABA","Citric acid","Acetic acid","Neopterin","24-OH-cholesterol","Cholesterol","L-Cysteine","Ubiquinone-1"][::-1]
ra_s = [0.3349,0.3184,0.317,0.3005,0.1528,0.1491,0.1335,0.1314,0.0319,0.0312][::-1]
ra_t = ["Gut microbiome/SCFA","Energy metabolism","CNS/Neuro","Energy metabolism","Gut microbiome/SCFA","Immune activation","Oxysterol/Lipid","Oxysterol/Lipid","Purine/Redox","Energy metabolism"][::-1]
lollipop_panel(ax, ra_m, ra_s, ra_t, "RA \u2014 Top-10 (LODO)", "C", xmax=0.4)

# IBD
ax = fig.add_subplot(gs[2,0])
ibd_m = ["Butyric acid","Propionic acid","Hippuric acid","L-Lactic acid","Indole-3-propionic acid","D-Glucose","L-Tryptophan","Acetic acid","Citric acid","Choline"][::-1]
ibd_s = [0.9234,0.9187,0.9012,0.8978,0.8845,0.8712,0.8634,0.8521,0.8467,0.8389][::-1]
ibd_t = ["Gut microbiome/SCFA","Gut microbiome/SCFA","Gut microbiome/SCFA","Energy metabolism","Gut microbiome/SCFA","Energy metabolism","Immune activation","Gut microbiome/SCFA","Energy metabolism","Bile acid"][::-1]
lollipop_panel(ax, ibd_m, ibd_s, ibd_t, "IBD \u2014 Top-10 (LODO, highest AUC=0.931)", "D", xmax=1.0)

# SLE
ax = fig.add_subplot(gs[2,1])
sle_m = ["Formic acid","D-Glucose","L-Valine","Pyruvic acid","S-3-oxodecanoyl cysteamine","\u03b2-Cryptoxanthin","Dimethylamine","2-(2-Phenylacetoxy)propionylglycine","Lutein","Zeaxanthin"][::-1]
sle_s = [0.0124,0.0023,0.0019,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001][::-1]
sle_t = ["One-carbon","Energy metabolism","BCAA","Energy metabolism","Bile acid","Carotenoid/Antioxidant","Gut microbiome/SCFA","Bile acid","Carotenoid/Antioxidant","Carotenoid/Antioxidant"][::-1]
lollipop_panel(ax, sle_m, sle_s, sle_t, "SLE \u2014 Top-10 (model extrapolation,\nzero HMDB training associations)", "E", xmax=0.014)

handles = [Patch(facecolor=tcolor[t], edgecolor=DARK, label=t) for t in
           ["Energy metabolism","Gut microbiome/SCFA","CNS/Neuro","BCAA","One-carbon","Oxysterol/Lipid",
            "Immune activation","Bile acid","Carotenoid/Antioxidant"]]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.015))

fig.suptitle("Figure 7. Disease-specific top-10 metabolite predictions reveal shared gut-microbiome/energy-metabolism\naxes alongside disease-specific fingerprints (CNS in MS, SCFA dominance in IBD)",
             fontsize=11.5, fontweight="bold", y=1.015)
os.makedirs("figures", exist_ok=True)
save(fig, "figures/Figure7_Disease_Specific_Predictions.png")
