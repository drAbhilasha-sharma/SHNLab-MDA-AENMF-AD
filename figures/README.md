# MDA-AENMF-AD Manuscript Figures — Generation Scripts
## Setup
```bash
pip install -r requirements.txt
```

## Generate all figures
```bash
python run_all.py
```
PNGs are written to `./figures/`.

## Generate a single figure
```bash
python fig5_ms_predictions.py
```

## Files
- `style.py` — shared Set3-based color palette, matplotlib rcParams, and
  reusable non-bar chart helpers: `lollipop_h/v`, `slope_chart`,
  `dumbbell_h`, `treemap`, `bump_chart`, `stacked_area`.
- `fig2_performance.py` — CV / LODO / Y-randomisation performance
  (line plot, Cleveland dot plot, slope chart, histogram, lollipop,
  stacked-area).
- `fig3_ablation.py` — module ablation study (horizontal lollipop,
  slope chart, lollipop).
- `fig4_sensitivity.py` — hyperparameter sensitivity (line plots +
  dumbbell summary panel).
- `fig5_ms_predictions.py` — MS top-20 predictions (lollipop, donut,
  treemap).
- `fig6_crossdisease.py` — cross-disease comparison (heatmap, slope
  chart, radar chart).
- `fig7_disease_predictions.py` — disease-specific top-10 predictions
  (bump chart + lollipop small multiples).
- `fig8_pathway_themes.py` — MS pathway-theme mechanistic analysis
  (lollipop, bubble scatter, pie, donut).
- `fig9_litval_config.py` — literature validation & model configuration
  (donut, lollipop, dumbbell, lollipop).

## Editing the data
All values are hard-coded near the top of each script (copied directly
from Tables S1–S12 in `MDA_AENMF_AD_Supplementary_Data.xlsx`) — edit
those arrays directly if numbers change, no need to touch the plotting
logic below them.

## Notes
- All qualitative/categorical elements (lollipops, pies, slopes, bump
  lines) use the Set3 palette to match your existing graphical
  abstract / workflow figure.
- The one continuous-value panel (Fig. 6A cross-disease heatmap) uses a
  sequential colormap (YlGnBu) instead of Set3, since Set3 is a
  qualitative palette and is not appropriate for representing an
  ordered numeric scale — this is standard practice for heatmaps in
  quantitative journals.
