# MDA-AENMF-AD

**Predicting Metabolite–Disease Associations in Autoimmune Disorders via an Integrated Deep Learning Framework Incorporating Immune-Specific Similarity Networks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)]()

---

## Overview

MDA-AENMF-AD is a deep learning framework for predicting metabolite–disease associations (MDAs) in autoimmune disorders. It integrates five immune-specific disease similarity networks fused via Similarity Network Fusion (SNF), a Disease Autoencoder (DAE), a Metabolite Graph Autoencoder (GAE), joint Non-negative Matrix Factorisation (NMF), and an MLP predictor.

### Key results

| Metric | Value |
|--------|-------|
| 5-fold CV AUC | 0.9994 ± 0.0006 |
| 5-fold CV AUPR | 0.9994 ± 0.0006 |
| MS LODO AUC | 0.782 |
| RA LODO AUC | 0.813 |
| IBD LODO AUC | 0.931 |
| T1D LODO AUC | 0.764 |
| Y-rand AUC (×3) | ~0.51 (p < 0.001) |
| MS Precision@10 | 0.70 |
| MS Precision@20 | 0.90 |

---

## Repository Structure

```
MDA_AENMF_AD_repo/
├── src/
│   ├── models/
│   │   ├── dae.py               Disease Autoencoder
│   │   ├── gae.py               Graph Autoencoder (GCN-based)
│   │   ├── nmf.py               Joint NMF co-embedding
│   │   ├── mlp.py               MLP predictor
│   │   └── mda_aenmf_ad.py      Full integrated model
│   ├── data/
│   │   ├── dataset.py           Dataset loader & negative sampling
│   │   ├── similarity.py        IPS / APS / DGIP / DSS / DSIE networks
│   │   └── features.py          ECFP4 fingerprints & metabolite graph
│   ├── evaluation/
│   │   ├── cross_validation.py  5-fold stratified CV
│   │   ├── lodo.py              Leave-one-disease-out evaluation
│   │   ├── permutation.py       Y-randomisation null test
│   │   └── metrics.py           AUC, AUPR, F1, Precision@k
│   └── utils/
│       ├── snf.py               Similarity Network Fusion
│       ├── visualise.py         All manuscript figure generation
│       └── logger.py            Experiment logging
├── data/
│   ├── raw/
│   │   ├── hmdb_associations.csv        HMDB v5.0 metabolite-disease associations
│   │   ├── kegg_immune_pathways.csv     20 KEGG immune pathways membership
│   │   ├── aida_autoantibodies.csv      AIDA v2.0 autoantibody profiles
│   │   ├── mesh_similarity.csv          MeSH disease semantic similarity
│   │   ├── symptom_entropy.csv          Disease symptom information entropy
│   │   └── metabolite_smiles.csv        Canonical SMILES for all metabolites
│   └── processed/
│       ├── snf_consensus.npy            Fused SNF similarity matrix
│       ├── ecfp4_fingerprints.npy       ECFP4 (1024-bit) feature matrix
│       └── metabolite_graph.pkl         Metabolite interaction graph
├── results/
│   ├── predictions/
│   │   ├── case_study_MS.csv            MS top-20 predictions (LODO)
│   │   ├── case_study_RA.csv            RA top-20 predictions (LODO)
│   │   ├── case_study_IBD.csv           IBD top-20 predictions (LODO)
│   │   └── case_study_SLE.csv           SLE extrapolation predictions
│   ├── metrics/
│   │   ├── cv_results.csv               5-fold CV per-fold metrics
│   │   ├── lodo_results.csv             LODO AUC/AUPR per disease
│   │   └── permutation_results.csv      Y-randomisation runs
│   └── ablation/
│       ├── ablation_results.csv         Component ablation study
│       └── sensitivity_results.csv      Hyperparameter sensitivity
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_similarity_network_construction.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_cv_lodo.ipynb
│   └── 05_case_study_ms_analysis.ipynb
├── configs/
│   ├── default.yaml             Default hyperparameters
│   └── ablation.yaml            Ablation study configurations
├── tests/
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_data.py
├── docs/
│   └── api_reference.md
├── generate_all_figures.py      Reproduce all manuscript figures
├── train.py                     Main training entry point
├── evaluate.py                  Evaluation entry point
├── requirements.txt
├── environment.yml
├── setup.py
└── LICENSE
```

---

## Installation

### Option 1: pip
```bash
git clone https://github.com/[author]/MDA_AENMF_AD.git
cd MDA_AENMF_AD
pip install -r requirements.txt
```

### Option 2: conda
```bash
git clone https://github.com/[author]/MDA_AENMF_AD.git
cd MDA_AENMF_AD
conda env create -f environment.yml
conda activate mda_aenmf_ad
```

---

## Quick Start

### Training
```bash
python train.py --config configs/default.yaml
```

### Evaluation (5-fold CV + LODO)
```bash
python evaluate.py --mode all --config configs/default.yaml
```

### Reproduce all figures
```bash
python generate_all_figures.py
```

### Run MS case study
```python
from src.models.mda_aenmf_ad import MDAAENMFAD
from src.data.dataset import AutoimmuneDataset

dataset = AutoimmuneDataset('data/raw/')
model   = MDAAENMFAD.from_config('configs/default.yaml')
model.load_weights('results/checkpoints/best_model.pt')

ms_predictions = model.predict_disease('Multiple Sclerosis', top_k=20)
print(ms_predictions)
```

---

## Model Architecture

```
IPS ─┐
APS ─┤
DGIP─┼─→ SNF Fusion ─→ Disease AE (DAE) ─→ 64-d disease embedding ─┐
DSS ─┤                  [n_d→128→64→128→n_d]                        ├─→ Joint NMF ─→ MLP ─→ Score
DSIE─┘                                                               │   (rank=4)    [8→64→32→1]
                                                                     │
ECFP4 + Metabolite Graph ─→ Graph AE (GAE) ─→ 64-d metab. embedding─┘
                             GCN [1024→256→64]
```

---

## Evaluation Framework

Three complementary validation approaches:

1. **5-Fold Stratified CV** — Standard benchmark; AUC = 0.9994
2. **Leave-One-Disease-Out (LODO)** — Cross-disease generalisation; MS AUC = 0.782
3. **Y-Randomisation** — Null test; permuted AUC ≈ 0.51 (p < 0.001)

---

## Data Sources

| Source | Version | URL |
|--------|---------|-----|
| HMDB | v5.0 | https://hmdb.ca |
| KEGG | 2023 | https://www.kegg.jp |
| AIDA | v2.0 | https://aida.rare-diseases.eu |
| MeSH | 2023 | https://www.nlm.nih.gov/mesh |
| RDKit | 2023.03 | https://www.rdkit.org |

---

## Reproducing Results

All results can be reproduced from raw data:

```bash
# Step 1: Preprocess data
python src/data/dataset.py --preprocess

# Step 2: Build similarity networks
python src/data/similarity.py --all

# Step 3: Train model
python train.py --config configs/default.yaml --seed 42

# Step 4: Run all evaluations
python evaluate.py --mode cv lodo permutation

# Step 5: Generate figures
python generate_all_figures.py --outdir manuscript_figures/
```

---

## Citation

If you use MDA-AENMF-AD in your research, please cite:

```bibtex
@article{MDA_AENMF_AD_2026,
  title   = {Predicting Metabolite-Disease Associations in Autoimmune Disorders
             via an Integrated Deep Learning Framework Incorporating
             Immune-Specific Similarity Networks},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[pending]}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
