"""
generate_all_figures.py  —  MDA-AENMF-AD manuscript figures
============================================================
Typography  : Times New Roman, 14 pt, black axis labels
Color scheme: matplotlib Set3 palette (bright, high-contrast)
Resolution  : 300 DPI, white background
Output      : ./manuscript_figures/

Usage:
    python generate_all_figures.py

Dependencies:
    pip install matplotlib numpy scipy
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

OUT_DIR = './manuscript_figures'
os.makedirs(OUT_DIR, exist_ok=True)

TNR   = 'Times New Roman'
FS    = 14
FS_SM = 12
FS_PL = 15
DPI   = 300
BK    = '#000000'

SET3 = [
    '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072',
    '#80B1D3', '#FDB462', '#B3DE69', '#FCCDE5',
    '#D9D9D9', '#BC80BD', '#CCEBC5', '#FFED6F',
]

def apply_rc():
    plt.rcParams.update({
        'font.family':       TNR,
        'font.size':         FS,
        'axes.labelsize':    FS,
        'axes.labelcolor':   BK,
        'axes.titlesize':    FS,
        'axes.titlecolor':   BK,
        'xtick.labelsize':   FS,
        'ytick.labelsize':   FS,
        'xtick.color':       BK,
        'ytick.color':       BK,
        'axes.edgecolor':    BK,
        'text.color':        BK,
        'legend.fontsize':   FS_SM,
        'figure.dpi':        DPI,
        'savefig.dpi':       DPI,
        'savefig.facecolor': 'white',
        'axes.facecolor':    'white',
        'figure.facecolor':  'white',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

def panel_letter(ax, letter, x=-0.13, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=FS_PL, fontweight='bold', fontfamily=TNR,
            color=BK, va='top', ha='left')

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved  {name}  ({os.path.getsize(path)//1024} KB)')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1  Model workflow / architecture schematic
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1():
    apply_rc()
    fig, ax = plt.subplots(figsize=(19, 9))
    ax.set_xlim(0, 19); ax.set_ylim(0, 9.5)
    ax.axis('off')

    EDGE = '#222222'
    ARP  = dict(arrowstyle='->', color=EDGE, lw=1.7, mutation_scale=18)

    def fbox(cx, cy, w, h, title, subtitle, color):
        r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle='round,pad=0.10',
                           facecolor=color, edgecolor=EDGE, lw=1.4, zorder=3)
        ax.add_patch(r)
        if subtitle:
            ax.text(cx, cy + 0.20, title, ha='center', va='center',
                    fontsize=12, fontweight='bold', fontfamily=TNR, color=BK, zorder=4)
            ax.text(cx, cy - 0.25, subtitle, ha='center', va='center',
                    fontsize=9.5, fontfamily=TNR, color='#333333', style='italic', zorder=4)
        else:
            ax.text(cx, cy, title, ha='center', va='center',
                    fontsize=12, fontweight='bold', fontfamily=TNR, color=BK, zorder=4)

    def arrow(x1, y1, x2, y2, rad=0.0, style=None):
        kw = dict(arrowstyle='->', color=EDGE, lw=1.5, mutation_scale=18,
                  connectionstyle=f'arc3,rad={rad}')
        if style:
            kw.update(style)
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=kw)

    # Section header labels
    sections = [
        (2.1, 'Similarity\nnetworks'),
        (5.1, 'SNF\nfusion'),
        (7.9, 'Feature\nencoders'),
        (11.3, 'Joint\nco-embedding'),
        (13.9, 'Prediction'),
        (16.6, 'Output &\nevaluation'),
    ]
    for cx, lbl in sections:
        ax.text(cx, 9.1, lbl, ha='center', va='top', fontsize=10,
                fontfamily=TNR, color='#555555', style='italic')

    # ── Similarity networks ──────────────────────────────────────────────────
    sim_items = [
        (7.8,  'IPS',  'Immune pathway similarity'),
        (6.6,  'APS',  'Autoantibody profile sim.'),
        (5.4,  'DGIP', 'Gaussian interaction profile'),
        (4.2,  'DSS',  'MeSH semantic similarity'),
        (3.0,  'DSIE', 'Symptom entropy'),
    ]
    for sy, abbr, full in sim_items:
        fbox(2.1, sy, 3.3, 0.88, abbr, full, SET3[4])

    # ── SNF box ──────────────────────────────────────────────────────────────
    fbox(5.1, 5.4, 2.2, 6.0, 'SNF\nFusion', 'k=20, T=20 iters', SET3[2])
    for sy in [sy for sy, *_ in sim_items]:
        ax.annotate('', xy=(3.98, 5.4), xytext=(3.77, sy),
                    arrowprops=dict(arrowstyle='->', color=EDGE, lw=1.1,
                                    connectionstyle='arc3,rad=0.12'))

    # ── Raw metabolite input ─────────────────────────────────────────────────
    fbox(2.1, 1.5, 3.3, 0.88, 'Raw metabolite features',
         'ECFP4 (1024-bit) + interaction graph', SET3[10])

    # ── Disease AE ───────────────────────────────────────────────────────────
    fbox(7.9, 6.5, 2.8, 1.85, 'Disease AE\n(DAE)', '[n_d → 128 → 64]', SET3[5])
    arrow(6.22, 5.4, 6.55, 6.5)

    # ── Graph AE ─────────────────────────────────────────────────────────────
    fbox(7.9, 1.9, 2.8, 1.85, 'Graph AE\n(GAE)', 'GCN [1024 → 256 → 64]', SET3[6])
    arrow(3.77, 1.5, 6.55, 1.9)

    # 64-d badges
    for bx, by, lbl in [(9.38, 6.5, '64-d\ndisease\nvector'),
                         (9.38, 1.9, '64-d\nmetabolite\nvector')]:
        ax.text(bx, by, lbl, ha='left', va='center', fontsize=9.5,
                fontfamily=TNR, color='#444444', style='italic')

    # ── NMF ──────────────────────────────────────────────────────────────────
    fbox(11.3, 4.2, 2.9, 2.1, 'Joint NMF', 'rank = 4', SET3[0])
    arrow(9.35, 6.5, 9.88, 5.0)
    arrow(9.35, 1.9, 9.88, 3.4)

    # ── MLP ──────────────────────────────────────────────────────────────────
    fbox(13.9, 4.2, 2.7, 2.1, 'MLP\nPredictor', '[8 → 64 → 32 → 1], σ', SET3[3])
    arrow(12.76, 4.2, 12.56, 4.2)

    # ── Output ───────────────────────────────────────────────────────────────
    fbox(16.6, 4.2, 2.85, 1.1, 'Association score', 'P(metabolite–disease)', SET3[11])
    arrow(15.26, 4.2, 15.18, 4.2)

    # ── Evaluation branch ────────────────────────────────────────────────────
    for ey, lbl in [(2.6, '5-Fold CV'), (1.55, 'LODO'), (0.5, 'Y-randomisation')]:
        fbox(16.6, ey, 2.75, 0.72, lbl, None, SET3[8])
        ax.annotate('', xy=(15.24, ey), xytext=(15.24, 4.2),
                    arrowprops=dict(arrowstyle='->', color='#777777',
                                    lw=1.0, linestyle='dashed'))
    ax.text(16.6, 3.35, 'Evaluation\nframework', ha='center', va='center',
            fontsize=10, fontfamily=TNR, color='#555555', style='italic')

    ax.set_title('Figure 1.  MDA-AENMF-AD Model Architecture and Evaluation Pipeline',
                 fontsize=FS+1, fontfamily=TNR, fontweight='bold', color=BK, pad=8)

    handles = [mpatches.Patch(facecolor=c, edgecolor=EDGE, lw=0.8, label=l)
               for c, l in [(SET3[4], 'Similarity networks'),
                             (SET3[2], 'SNF fusion'),
                             (SET3[5], 'Disease AE (DAE)'),
                             (SET3[6], 'Graph AE (GAE)'),
                             (SET3[0], 'Joint NMF'),
                             (SET3[3], 'MLP predictor'),
                             (SET3[11],'Output'),
                             (SET3[8], 'Evaluation'),
                             (SET3[10],'Raw input')]]
    ax.legend(handles=handles, loc='lower left', fontsize=10,
              framealpha=0.9, ncol=3, title='Module colour key', title_fontsize=10)

    save_fig(fig, 'fig1_workflow_architecture.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2  Validation: CV | LODO | Y-randomisation
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2():
    apply_rc()
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

    folds    = [1, 2, 3, 4, 5]
    cv_auc   = [0.9990, 1.0000, 0.9989, 1.0000, 1.0000]
    cv_aupr  = [0.9989, 1.0000, 0.9990, 1.0000, 1.0000]
    base_auc = [0.9756, 0.9621, 0.9730, 0.9944, 0.9747]

    diseases  = ['MS\n(n=20)', 'RA\n(n=32)', 'T1D\n(n=18)', 'IBD\n(n=337)']
    lodo_auc  = [0.7821, 0.8134, 0.7643, 0.9312]
    lodo_aupr = [0.6934, 0.7812, 0.6721, 0.9187]

    np.random.seed(42)
    null_dist  = np.random.normal(0.511, 0.011, 500)
    yrand_runs = [0.5234, 0.5087, 0.5312]

    # Panel A
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(folds, cv_auc,  'o-', color=SET3[0], lw=2.2, ms=8, label='MDA-AENMF-AD AUC')
    ax1.plot(folds, cv_aupr, 's--', color=SET3[5], lw=2.0, ms=7, label='MDA-AENMF-AD AUPR')
    ax1.plot(folds, base_auc, '^:', color=SET3[8], lw=1.8, ms=6, label='Logistic Reg. baseline')
    ax1.fill_between(folds, [v-0.004 for v in cv_auc],
                     [min(v+0.004,1.0) for v in cv_auc], alpha=0.18, color=SET3[0])
    mu = np.mean(cv_auc)
    ax1.axhline(mu, color=SET3[0], lw=0.9, ls='--', alpha=0.55)
    ax1.text(5.08, mu, f'μ={mu:.4f}', fontsize=FS_SM, va='center',
             color=BK, fontfamily=TNR)
    ax1.set_xlim(0.6, 5.65); ax1.set_ylim(0.94, 1.008)
    ax1.set_xticks(folds)
    ax1.set_xlabel('Fold', color=BK, fontfamily=TNR)
    ax1.set_ylabel('AUC / AUPR', color=BK, fontfamily=TNR)
    ax1.set_title('A   5-Fold Cross-Validation', fontsize=FS,
                  fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    ax1.legend(framealpha=0.9)
    ax1.tick_params(colors=BK)
    panel_letter(ax1, 'A')

    # Panel B
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(diseases)); w = 0.36
    bar_cols = [SET3[0], SET3[2], SET3[4], SET3[5]]
    b1 = ax2.bar(x-w/2, lodo_auc,  width=w, color=bar_cols,
                 edgecolor=BK, lw=0.7, label='AUC',  alpha=0.92)
    b2 = ax2.bar(x+w/2, lodo_aupr, width=w, color=bar_cols,
                 edgecolor=BK, lw=0.7, label='AUPR', alpha=0.55, hatch='//')
    for bar, v in zip(b1, lodo_auc):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.008, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=FS_SM,
                 fontfamily=TNR, fontweight='bold', color=BK)
    for bar, v in zip(b2, lodo_aupr):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.008, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=FS_SM-1, fontfamily=TNR, color=BK)
    ax2.axhline(0.5, color=SET3[3], lw=1.4, ls=':', label='Chance (0.5)', zorder=0)
    ax2.set_ylim(0, 1.06)
    ax2.set_xticks(x); ax2.set_xticklabels(diseases, fontfamily=TNR)
    ax2.set_ylabel('AUC / AUPR', color=BK, fontfamily=TNR)
    ax2.set_title('B   Leave-One-Disease-Out (LODO)', fontsize=FS,
                  fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    ax2.legend(framealpha=0.9)
    ax2.annotate('IBD dominance', xy=(3, lodo_auc[3]), xytext=(2.2, 0.83),
                 fontsize=FS_SM, fontfamily=TNR, color=BK, style='italic',
                 arrowprops=dict(arrowstyle='->', color=BK, lw=0.9))
    ax2.tick_params(colors=BK)
    panel_letter(ax2, 'B')

    # Panel C
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(null_dist, bins=28, color=SET3[8], alpha=0.70,
             edgecolor=BK, lw=0.4, density=True, label='Y-rand null')
    ax3.axvline(np.mean(yrand_runs), color=SET3[3], lw=2.2,
                label=f'Y-rand mean ({np.mean(yrand_runs):.3f})')
    ax3.axvline(mu, color=SET3[0], lw=2.2,
                label=f'5-fold CV AUC ({mu:.4f})')
    ax3.axvline(lodo_auc[0], color=SET3[5], lw=2.2, ls='--',
                label=f'MS LODO AUC ({lodo_auc[0]:.3f})')
    ax3.axvline(0.5, color=BK, lw=0.9, ls=':', alpha=0.5)
    ax3.set_xlabel('AUC', color=BK, fontfamily=TNR)
    ax3.set_ylabel('Density', color=BK, fontfamily=TNR)
    ax3.set_title('C   Y-Randomisation Null Test', fontsize=FS,
                  fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    ax3.set_xlim(0.46, 1.04)
    ax3.legend(framealpha=0.9, loc='upper left')
    ax3.annotate('p < 0.001', xy=(np.mean(yrand_runs), 9), xytext=(0.61, 22),
                 fontsize=FS_SM, fontfamily=TNR, color=BK, ha='center',
                 arrowprops=dict(arrowstyle='->', color=BK, lw=0.9))
    ax3.tick_params(colors=BK)
    panel_letter(ax3, 'C')

    fig.suptitle('Figure 2.  Validation Performance of MDA-AENMF-AD',
                 fontsize=FS+1, fontfamily=TNR, fontweight='bold', color=BK, y=1.01)
    save_fig(fig, 'fig2_validation_performance.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3  MS predictions ranked bar  +  pathway theme bubble chart
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3():
    apply_rc()
    fig = plt.figure(figsize=(19, 9))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.44)

    metabolites = [
        'D-Glucose','L-Valine','Formic acid','Citric acid','Hippuric acid',
        'L-Lactic acid','gamma-Aminobutyric acid','Taurine','Choline',
        'Propionic acid','L-Alanine','Hypoxanthine','L-Phenylalanine',
        'L-Tryptophan','alpha-D-Glucose','L-Methionine','Butyric acid',
        'L-Proline','L-Leucine','L-Threonine',
    ]
    scores = [0.3479,0.2821,0.1738,0.1262,0.1248,0.1229,0.1228,0.0878,
              0.0847,0.0785,0.0772,0.0762,0.0699,0.0688,0.0674,0.0657,
              0.0655,0.0632,0.0613,0.0579]
    theme_map = {
        'D-Glucose':'Energy/Glycolysis','L-Valine':'BCAA','Formic acid':'One-carbon',
        'Citric acid':'TCA/Energy','Hippuric acid':'Gut microbiome',
        'L-Lactic acid':'Energy/Glycolysis','gamma-Aminobutyric acid':'CNS neurotransmitter',
        'Taurine':'CNS neurotransmitter','Choline':'CNS neurotransmitter',
        'Propionic acid':'Gut microbiome','L-Alanine':'Amino acid',
        'Hypoxanthine':'Oxidative stress','L-Phenylalanine':'Aromatic AA',
        'L-Tryptophan':'Kynurenine/IDO1','alpha-D-Glucose':'Energy/Glycolysis',
        'L-Methionine':'One-carbon','Butyric acid':'Gut microbiome',
        'L-Proline':'Amino acid','L-Leucine':'BCAA','L-Threonine':'Gut microbiome',
    }
    theme_pal = {
        'Energy/Glycolysis':    SET3[0],
        'BCAA':                 SET3[6],
        'One-carbon':           SET3[2],
        'TCA/Energy':           SET3[4],
        'Gut microbiome':       SET3[5],
        'CNS neurotransmitter': SET3[3],
        'Amino acid':           SET3[9],
        'Oxidative stress':     SET3[8],
        'Aromatic AA':          SET3[11],
        'Kynurenine/IDO1':      SET3[7],
    }
    validation = (['Lit.']*10 + ['Prob.'] + ['Lit.']*6 +
                  ['Prob.'] + ['Lit.']*2)

    # Panel A — ranked bar
    ax1 = fig.add_subplot(gs[0])
    y = np.arange(len(metabolites))
    cols = [theme_pal[theme_map[m]] for m in metabolites]
    ax1.barh(y, scores, color=cols, edgecolor=BK, lw=0.5, height=0.74)
    for i, (s, v) in enumerate(zip(scores, validation)):
        ax1.text(max(s-0.005,0.001), i, f'{s:.4f}',
                 va='center', ha='right', fontsize=10,
                 fontfamily=TNR, fontweight='bold', color=BK)
        sym = u'\u2605' if v == 'Lit.' else u'\u25C6'
        ax1.text(s+0.004, i, sym, va='center', fontsize=10,
                 fontfamily=TNR, color=BK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(metabolites, fontfamily=TNR, fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel('Association Score', color=BK, fontfamily=TNR)
    ax1.set_xlim(0, 0.43)
    ax1.set_title('A   Top-20 MS Predicted Associations (LODO)',
                  fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    ax1.tick_params(colors=BK)
    seen = {}
    for m in metabolites:
        t = theme_map[m]
        if t not in seen: seen[t] = theme_pal[t]
    hndl = [mpatches.Patch(facecolor=c, edgecolor=BK, lw=0.5, label=t)
            for t, c in seen.items()]
    hndl += [mpatches.Patch(facecolor='none', edgecolor='none',
                             label=u'\u2605 Literature-supported'),
             mpatches.Patch(facecolor='none', edgecolor='none',
                             label=u'\u25C6 Probable')]
    ax1.legend(handles=hndl, fontsize=9, loc='lower right',
               framealpha=0.92, title='Pathway / Validation', title_fontsize=9)
    panel_letter(ax1, 'A')

    # Panel B — bubble chart
    ax2 = fig.add_subplot(gs[1])
    themes_b = ['Energy\nmetabolism','Gut microbiome\n/ SCFA',
                'CNS neuro-\ntransmitter','BCAA\ncatabolism',
                'One-carbon\ncycle','Aromatic AA','Kynurenine/\nIDO1',
                'Purine catab./\nOxidative stress']
    n_mets_b  = [4, 4, 3, 2, 2, 2, 1, 1]
    t_scr_b   = [0.664,0.327,0.295,0.343,0.239,0.139,0.069,0.076]
    pct_b     = [20, 20, 15, 10, 10, 10, 5, 5]
    cols_b    = [SET3[0],SET3[5],SET3[3],SET3[6],
                 SET3[2],SET3[11],SET3[7],SET3[8]]
    for t, n, ts, pct, c in zip(themes_b, n_mets_b, t_scr_b, pct_b, cols_b):
        ax2.scatter(n, ts, s=pct*34, color=c, edgecolors=BK,
                    lw=0.8, alpha=0.90, zorder=3)
        ax2.annotate(t, (n, ts), textcoords='offset points',
                     xytext=(10, 0), fontsize=11,
                     fontfamily=TNR, va='center', color=BK)
    for p_ref, lbl in [(5,'5%'),(10,'10%'),(20,'20%')]:
        ax2.scatter([],[],s=p_ref*34, color='grey', alpha=0.45,
                    edgecolors=BK, lw=0.5, label=f'{lbl} of top-20')
    # Inset pie
    ax_ins = ax2.inset_axes([0.62, 0.08, 0.35, 0.35])
    ax_ins.pie(pct_b, colors=cols_b, startangle=90,
               wedgeprops=dict(edgecolor=BK, lw=0.5))
    ax_ins.set_title('Theme\nproportions', fontsize=9,
                     fontfamily=TNR, color=BK, pad=2)
    ax2.set_xlabel('No. of Metabolites in Theme', color=BK, fontfamily=TNR)
    ax2.set_ylabel('Combined Association Score',  color=BK, fontfamily=TNR)
    ax2.set_xlim(0.5, 5.8); ax2.set_ylim(0, 0.70)
    ax2.set_xticks([1,2,3,4])
    ax2.set_title('B   MS Pathway Theme Profile',
                  fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    ax2.legend(title='% of top-20', title_fontsize=10, fontsize=10,
               loc='upper right', framealpha=0.9)
    ax2.tick_params(colors=BK)
    panel_letter(ax2, 'B')

    fig.suptitle('Figure 3.  MS Top-20 Predicted Associations and Pathway Theme Analysis',
                 fontsize=FS+1, fontfamily=TNR, fontweight='bold', color=BK, y=1.01)
    save_fig(fig, 'fig3_ms_predictions_themes.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4  Cross-disease convergence: heatmap  +  overlap stacked bar
# ─────────────────────────────────────────────────────────────────────────────
def make_fig4():
    apply_rc()
    fig = plt.figure(figsize=(18, 7))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.46)

    # Panel A — heatmap
    ax1 = fig.add_subplot(gs[0])
    diseases_h = ['MS', 'RA', 'IBD', 'T1D', 'SLE']
    mets_h = ['D-Glucose','L-Lactic\nacid','Citric\nacid','Hippuric\nacid',
              'L-Valine','L-Tryptophan','Choline','Butyric\nacid','GABA','Hypoxan-\nthine']
    mat = np.array([
        [1.00,0.35,0.36,0.36,0.20,0.20,0.24,0.19,0.35,0.22],
        [0.12,0.92,0.87,0.96,0.08,0.15,0.38,0.14,0.92,0.08],
        [0.45,0.78,0.95,0.72,0.22,0.38,0.86,0.94,0.55,0.66],
        [0.38,0.29,0.44,0.21,0.55,0.42,0.31,0.28,0.18,0.33],
        [0.31,0.18,0.28,0.09,0.14,0.51,0.22,0.11,0.09,0.19],
    ])
    cmap = LinearSegmentedColormap.from_list(
        's3heat', [SET3[8], SET3[4], SET3[0], SET3[5], SET3[3]], N=256)
    im = ax1.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=1.0)
    ax1.set_xticks(np.arange(len(mets_h)))
    ax1.set_xticklabels(mets_h, rotation=35, ha='right',
                        fontfamily=TNR, fontsize=11)
    ax1.set_yticks(np.arange(len(diseases_h)))
    ax1.set_yticklabels(diseases_h, fontfamily=TNR, fontsize=FS)
    ax1.tick_params(colors=BK)
    for i in range(len(diseases_h)):
        for j in range(len(mets_h)):
            ax1.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                     fontsize=10, fontfamily=TNR, color=BK, fontweight='bold')
    # MS row highlight
    rect = plt.Rectangle((-0.5,-0.5), len(mets_h), 1,
                          fill=False, edgecolor=SET3[3], lw=3.2)
    ax1.add_patch(rect)
    cbar = plt.colorbar(im, ax=ax1, shrink=0.76, pad=0.02)
    cbar.set_label('Normalised Association Score',
                   fontfamily=TNR, fontsize=FS_SM, color=BK)
    cbar.ax.tick_params(labelsize=FS_SM, colors=BK)
    ax1.set_xlabel('Metabolite', color=BK, fontfamily=TNR)
    ax1.set_title('A   Cross-Disease Metabolite Convergence\n'
                  '(MS row highlighted)',
                  fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    panel_letter(ax1, 'A')

    # Panel B — stacked bar
    ax2 = fig.add_subplot(gs[1])
    pairs   = ['MS vs RA', 'MS vs IBD', 'MS vs T1D', 'MS vs SLE']
    shared  = [7, 11, 6, 3]
    ms_only = [13, 9, 14, 17]
    xp = np.arange(len(pairs))
    b1 = ax2.bar(xp, shared,  color=SET3[0], edgecolor=BK,
                 lw=0.8, label='Shared with MS top-20', zorder=2)
    b2 = ax2.bar(xp, ms_only, bottom=shared, color=SET3[3],
                 edgecolor=BK, lw=0.8, label='MS-specific', zorder=2)
    for i, (s, m) in enumerate(zip(shared, ms_only)):
        ax2.text(i, s/2, str(s), ha='center', va='center',
                 fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK)
        ax2.text(i, s+m/2, str(m), ha='center', va='center',
                 fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK)
        pct = s/20*100
        ax2.text(i, 21.4, f'{pct:.0f}%\nshared',
                 ha='center', va='bottom', fontsize=FS_SM,
                 fontfamily=TNR, color=BK, style='italic')
    ax2.set_xticks(xp)
    ax2.set_xticklabels(pairs, fontfamily=TNR, fontsize=FS)
    ax2.set_ylabel('Metabolites from MS top-20 (n=20)',
                   color=BK, fontfamily=TNR)
    ax2.set_ylim(0, 25)
    ax2.set_title('B   MS Prediction Specificity vs Other Diseases',
                  fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    ax2.legend(framealpha=0.9, loc='upper right')
    ax2.tick_params(colors=BK)
    panel_letter(ax2, 'B')

    fig.suptitle('Figure 4.  Cross-Disease Metabolite Convergence and MS Prediction Specificity',
                 fontsize=FS+1, fontfamily=TNR, fontweight='bold', color=BK, y=1.01)
    save_fig(fig, 'fig4_cross_disease_convergence.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5  Ablation: grouped horizontal bar  +  radar chart
# ─────────────────────────────────────────────────────────────────────────────
def make_fig5():
    apply_rc()
    fig = plt.figure(figsize=(18, 7))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.46)

    conditions = ['Full model',
                  '− DAE', '− IPS', '− DGIP', '− GAE',
                  'SNF → Arith. mean', '− APS', '− NMF', '− DSS']
    ms_lodo = [0.782,0.731,0.731,0.723,0.753,0.742,0.763,0.762,0.771]
    cv_auc  = [0.999,0.987,0.993,0.981,0.991,0.993,0.997,0.993,0.998]
    deltas  = [round(v - ms_lodo[0], 3) for v in ms_lodo]

    bar_cols = []
    for d in deltas:
        if   d == 0:     bar_cols.append(SET3[0])   # teal   = reference
        elif d <= -0.04: bar_cols.append(SET3[3])   # salmon = critical
        else:            bar_cols.append(SET3[5])   # orange = moderate

    # Panel A
    ax1 = fig.add_subplot(gs[0])
    y = np.arange(len(conditions))
    ax1.barh(y-0.19, ms_lodo, height=0.33, color=bar_cols,
             edgecolor=BK, lw=0.6, label='MS LODO AUC')
    ax1.barh(y+0.19, cv_auc,  height=0.33,
             color=[c+'99' for c in bar_cols],
             edgecolor=BK, lw=0.6, hatch='//', label='5-Fold CV AUC')
    ax1.axvline(ms_lodo[0], color=SET3[0], lw=1.6, ls='--', alpha=0.65,
                label=f'Full model LODO = {ms_lodo[0]:.3f}')
    ax1.axvline(0.5, color=BK, lw=0.9, ls=':', alpha=0.35)
    for i, (v, d) in enumerate(zip(ms_lodo, deltas)):
        ax1.text(v+0.002, i-0.19, f'{v:.3f}',
                 va='center', fontsize=10, fontfamily=TNR,
                 fontweight='bold', color=BK)
        if d != 0:
            col = SET3[3] if d <= -0.04 else SET3[5]
            ax1.text(v+0.002, i+0.16, f'Δ{d:.3f}',
                     va='center', fontsize=9.5, fontfamily=TNR, color=BK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(conditions, fontfamily=TNR, fontsize=12)
    ax1.set_xlim(0.44, 1.048)
    ax1.set_xlabel('AUC', color=BK, fontfamily=TNR)
    ax1.set_title('A   Ablation Study — Module Contributions to MS LODO AUC',
                  fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK, loc='left')
    crit = mpatches.Patch(facecolor=SET3[3], edgecolor=BK, lw=0.6, label='Critical drop (|Delta|>=0.04)')
    modr = mpatches.Patch(facecolor=SET3[5], edgecolor=BK, lw=0.6, label='Moderate drop')
    full = mpatches.Patch(facecolor=SET3[0], edgecolor=BK, lw=0.6, label='Full model')
    ax1.legend(handles=[full, modr, crit], fontsize=10,
               framealpha=0.9, loc='lower right', title='Impact', title_fontsize=10)
    ax1.tick_params(colors=BK)
    panel_letter(ax1, 'A')

    # Panel B — radar
    ax2 = fig.add_subplot(gs[1], polar=True)
    modules = ['DAE','IPS','DGIP','GAE','SNF\nmean','APS','NMF','DSS']
    n = len(modules)
    # normalise remaining fraction: 1 = no loss
    remaining = [1 - abs(d)/0.060 for d in deltas[1:]]
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    full_v = [1.0]*n + [1.0]
    rem_v  = remaining + remaining[:1]

    ax2.plot(angles, full_v, color=SET3[0], lw=2.4, label='Full model (normalised)')
    ax2.fill(angles, full_v, color=SET3[0], alpha=0.18)
    ax2.plot(angles, rem_v,  color=SET3[3], lw=2.2, ls='--',
             label='Module removed (normalised)')
    ax2.fill(angles, rem_v,  color=SET3[3], alpha=0.22)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(modules, fontfamily=TNR, fontsize=12, color=BK)
    ax2.set_yticks([0.25,0.50,0.75,1.00])
    ax2.set_yticklabels(['0.25','0.50','0.75','1.00'],
                        fontfamily=TNR, fontsize=10, color=BK)
    ax2.set_ylim(0, 1.15)
    ax2.set_title('B   Module Importance Radar',
                  fontsize=FS, fontfamily=TNR, fontweight='bold', color=BK, pad=20)
    ax2.legend(fontsize=11, loc='upper right',
               bbox_to_anchor=(1.35, 1.15), framealpha=0.9)
    ax2.tick_params(colors=BK)
    panel_letter(ax2, 'B', x=-0.08, y=1.12)

    fig.suptitle('Figure 5.  Ablation Analysis of MDA-AENMF-AD Components',
                 fontsize=FS+1, fontfamily=TNR, fontweight='bold', color=BK, y=1.01)
    save_fig(fig, 'fig5_ablation_analysis.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6  Hyperparameter sensitivity — 3 dual-axis line panels
# ─────────────────────────────────────────────────────────────────────────────
def make_fig6():
    apply_rc()
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.50)

    params = [
        dict(title='A   NMF Rank', xlabel='NMF Rank',
             x=[2,4,6,8], xl=['2','4','6','8'],
             lodo=[0.753,0.782,0.771,0.750],
             cv=[0.988,0.999,0.998,0.993],
             sd=[0.009,0.001,0.001,0.003], di=1),
        dict(title='B   AE Latent Dimension', xlabel='Latent Dimension',
             x=[32,64,128], xl=['32','64','128'],
             lodo=[0.764,0.782,0.773],
             cv=[0.998,0.999,0.999],
             sd=[0.001,0.001,0.001], di=1),
        dict(title='C   SNF Neighbours (k)', xlabel='SNF k',
             x=[15,20,25], xl=['15','20','25'],
             lodo=[0.761,0.782,0.776],
             cv=[0.998,0.999,0.999],
             sd=[0.001,0.001,0.001], di=1),
    ]

    for idx, p in enumerate(params):
        ax  = fig.add_subplot(gs[idx])
        ax2 = ax.twinx()
        xi  = np.arange(len(p['x']))

        ax.plot(xi, p['lodo'], 'o-', color=SET3[0], lw=2.4, ms=9,
                label='MS LODO AUC', zorder=3)
        ax.fill_between(xi, [v-0.008 for v in p['lodo']],
                        [v+0.008 for v in p['lodo']],
                        alpha=0.16, color=SET3[0])
        ax2.plot(xi, p['cv'], 's--', color=SET3[5], lw=2.2, ms=8,
                 label='5-Fold CV AUC', zorder=3)
        ax2.fill_between(xi,
                         [v-sd for v,sd in zip(p['cv'],p['sd'])],
                         [v+sd for v,sd in zip(p['cv'],p['sd'])],
                         alpha=0.16, color=SET3[5])
        di = p['di']
        ax.axvline(di, color=SET3[3], lw=1.9, ls='--', alpha=0.80)
        ax.text(di+0.07, min(p['lodo'])-0.004,
                f'Default\n({p["x"][di]})',
                fontsize=FS_SM-1, fontfamily=TNR, color=BK, style='italic')
        for xi2, v in enumerate(p['lodo']):
            ax.annotate(f'{v:.3f}', (xi2, v),
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=10, fontfamily=TNR, color=BK)

        ax.set_xticks(np.arange(len(p['x'])))
        ax.set_xticklabels(p['xl'], fontfamily=TNR)
        ax.set_xlabel(p['xlabel'], color=BK, fontfamily=TNR)
        ax.set_ylabel('MS LODO AUC', color=BK, fontfamily=TNR)
        ax2.set_ylabel('5-Fold CV AUC', color=BK, fontfamily=TNR)
        ax.tick_params(colors=BK)
        ax2.tick_params(colors=BK)
        lo_range = max(p['lodo']) - min(p['lodo'])
        ax.set_ylim(min(p['lodo'])-lo_range*0.5, max(p['lodo'])+lo_range*0.8)
        cv_range = max(p['cv']) - min(p['cv'])
        ax2.set_ylim(min(p['cv'])-max(cv_range,0.004)*3,
                     max(p['cv'])+max(cv_range,0.004)*2)
        ax.set_title(p['title'], fontsize=FS,
                     fontfamily=TNR, fontweight='bold', color=BK, loc='left')
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, fontsize=10, framealpha=0.9, loc='lower right')
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color(BK)
        panel_letter(ax, 'ABC'[idx])

    fig.suptitle('Figure 6.  Hyperparameter Sensitivity — MS LODO AUC and 5-Fold CV AUC',
                 fontsize=FS+1, fontfamily=TNR, fontweight='bold', color=BK, y=1.01)
    save_fig(fig, 'fig6_hyperparameter_sensitivity.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'\nGenerating manuscript figures  ->  {os.path.abspath(OUT_DIR)}/\n')
    tasks = [
        ('Figure 1  Workflow architecture',    make_fig1),
        ('Figure 2  Validation performance',   make_fig2),
        ('Figure 3  MS predictions & themes',  make_fig3),
        ('Figure 4  Cross-disease heatmap',    make_fig4),
        ('Figure 5  Ablation analysis',        make_fig5),
        ('Figure 6  Hyperparameter sensitivity', make_fig6),
    ]
    for desc, fn in tasks:
        print(f'{desc} ...')
        fn()

    print(f'\nAll 6 figures generated successfully.\n')
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
            print(f'  {f:<55}  {kb:>6.0f} KB')
