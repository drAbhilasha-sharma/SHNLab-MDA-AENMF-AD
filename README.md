# MDA-AENMF-AD

**Predicting Metabolite–Disease Associations in Autoimmune Disorders via an Integrated Deep Learning Framework Incorporating Immune-Specific Similarity Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

---

## Overview

MDA-AENMF-AD extends the MDA-AENMF architecture ([Gao et al. 2023](https://doi.org/10.1093/bib/bbad048)) with two novel immunologically grounded disease similarity networks designed specifically for the autoimmune metabolomics domain:

| Network | Abbreviation | Source |
|---|---|---|
| Immune Pathway Similarity | **IPS** | KEGG immune pathway co-activation profiles |
| Autoantibody Profile Similarity | **APS** | AIDA database autoantibody target annotations |

These networks are fused with three established similarity measures (MeSH Semantic Similarity, Gaussian Interaction Profile, Information Entropy Similarity) using Similarity Network Fusion (SNF). The integrated representation is processed by a disease auto-encoder to produce 64-dimensional disease feature vectors. Metabolite features are extracted by a Graph Attention Auto-encoder (GAE) and NMF factorisation. All features are concatenated into a 136-dimensional vector and classified by a multi-layer perceptron (MLP).

**Cross-validation performance (five-fold, 1:1 positive:negative sampling):**
| Metric | Mean | SD |
|---|---|---|
| AUC | 0.9996 | 0.0006 |
| AUPR | 0.9996 | 0.0006 |


---

## Repository Structure

```
SHNLab-MDA-AENMF-AD/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── train.py                    # Main training script
├── predict.py                  # Inference script for novel associations
├── data/
│   ├── README_data.md          # Data download and format instructions
│   ├── sample_associations.csv # Sample 50-row dataset for testing
│   └── __init__.py
├── models/
│   ├── mda_aenmf_ad.py        # Full model class
│   ├── disease_ae.py           # Disease auto-encoder
│   ├── graph_attention_ae.py   # GAE metabolite module
│   ├── nmf.py                  # NMF factorisation module
│   ├── similarity_networks.py  # IPS, APS, DSS, DGIP, DSIE, MSS, MGIP, MSIE
│   ├── snf.py                  # Similarity Network Fusion
│   └── __init__.py
├── utils/
│   ├── evaluation.py           # AUC, AUPR, cross-validation utilities
│   └── __init__.py
├── results/
│   ├── final_metrics.json      # Five-fold CV summary (Supplementary Table S5)
│   ├── lodo_results.csv        # LODO + Y-randomisation results (Supplementary Table S6)
│   ├── all_disease_predictions.csv        # Top-20 predictions, all 4 diseases (Tables S1-S4)
│   ├── case_study_Multiple_Sclerosis.csv  # Top-20 MS predictions (Supplementary Table S1)
│   ├── case_study_MS_annotated.csv        # MS predictions + literature validation (Supplementary Table S11)
│   ├── case_study_Rheumatoid_Arthritis.csv    # Top-20 RA predictions (Supplementary Table S2)
│   ├── case_study_Systemic_Lupus_Erythematosus.csv  # Top-20 SLE predictions (Supplementary Table S3)
│   └── case_study_Inflammatory_Bowel_Disease.csv    # Top-20 IBD predictions (Supplementary Table S4)
└── figures/                    # Figure generation scripts (fig2_performance.py ... fig9_litval_config.py);
                                 # see figures/README.md. Run figures/run_all.py to regenerate all PNGs.
```

Ablation results are produced by `data/05_run_all_real_data.py` together with
`figures/fig3_ablation.py` (source data: Supplementary Table S7).

---

## Installation

### Option 1: conda (recommended)

```bash
git clone https://github.com/drAbhilasha-sharma/SHNLab-MDA-AENMF-AD.git
cd SHNLab-MDA-AENMF-AD
conda env create -f environment.yml
conda activate mda-aenmf-ad
```

### Option 2: pip

```bash
git clone https://github.com/drAbhilasha-sharma/SHNLab-MDA-AENMF-AD.git
cd SHNLab-MDA-AENMF-AD
pip install -r requirements.txt
```

**Python version:** 3.10+ required. Tested on Ubuntu 22.04 and macOS 14 (CPU and CUDA 11.8).

---

## Data

The full curated dataset (407 metabolite–disease associations, 379 endogenous metabolites, 8 autoimmune diseases) is sourced from HMDB v5.0 and is not redistributed here due to the HMDB terms of use.

**To reproduce the full dataset:**

1. Download HMDB v5.0 metabolite XML: https://hmdb.ca/downloads
2. Run `python data/parse_hmdb.py --xml hmdb_metabolites.xml --out data/associations.csv`
3. See `data/README_data.md` for full instructions and alias mapping.

A 50-row sample dataset (`data/sample_associations.csv`) is provided for testing the pipeline end-to-end.

**PaDEL descriptors:** Compute using [PaDEL-Descriptor v2.21](http://www.yapcwsoft.com/dd/padeldescriptor/) with 2D descriptors enabled.

**KEGG immune pathways (IPS):** the immune-pathway similarity network is described in `models/similarity_networks.py`.

**AIDA autoantibody annotations (APS):** Download from https://aida.biomed.ntnu.no/

## Contact

Correspondence: Abhilasha Sharma ([ORCID: 0000-0002-2438-6556](https://orcid.org/0000-0002-2438-6556)) and Shivraj Hariram Nile ([ORCID: 0000-0003-3141-5754](https://orcid.org/0000-0003-3141-5754)), BRIC-National Agri-Food and Biomanufacturing Institute (BRIC-NABI), Mohali, Punjab, India.
