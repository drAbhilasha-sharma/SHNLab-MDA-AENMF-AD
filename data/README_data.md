# Data Preparation Guide

This document describes how to assemble the full dataset used in MDA-AENMF-AD.

## 1. HMDB v5.0 Association Data

The 407 experimentally verified metabolite–disease associations used in this study
were extracted from the Human Metabolome Database v5.0.

**Steps:**

1. Download the full HMDB XML from https://hmdb.ca/downloads  
   File: `hmdb_metabolites.zip` (~4 GB unzipped)

2. Run the parser:
   ```bash
   python utils/data_loader.py \
       --hmdb_path /path/to/hmdb_metabolites.xml \
       --disease_aliases data/disease_aliases.csv \
       --output_dir data/processed/
   ```

3. This produces:
   - `data/processed/associations.csv` — (met_idx, dis_idx, label) rows
   - `data/processed/metabolite_list.csv` — HMDB IDs and names
   - `data/processed/disease_list.csv` — canonical disease names

**Disease mapping:**  
Raw HMDB disease strings are standardised using `data/disease_aliases.csv`
(Supplementary Table S2). The 8 canonical diseases are:

| Canonical name | HMDB strings matched |
|---|---|
| Rheumatoid Arthritis | "Arthritis, Rheumatoid"; "Rheumatoid arthritis" |
| Multiple Sclerosis | "Multiple Sclerosis"; "Multiple sclerosis" |
| Inflammatory Bowel Disease | "Crohn's Disease"; "Ulcerative Colitis"; "IBD" |
| Type 1 Diabetes | "Diabetes Mellitus, Type 1"; "Insulin-Dependent Diabetes" |
| SLE | "Lupus Erythematosus, Systemic"; "Systemic lupus erythematosus" |
| Psoriasis | "Psoriasis" |
| Ankylosing Spondylitis | "Ankylosing Spondylitis"; "Spondylitis, Ankylosing" |
| Sjogren's Syndrome | "Sjogren's Syndrome"; "Sjögren Syndrome" |

**Note:** SLE, Psoriasis, Ankylosing Spondylitis, and Sjogren's Syndrome have
zero verified associations in HMDB v5.0. They are included as disease nodes
in the similarity networks but have no positive training examples.

---

## 2. PaDEL Molecular Descriptors

Compute 2,019 PaDEL descriptors for all 379 metabolites:

1. Download PaDEL-Descriptor 2.21 from http://www.yapcwsoft.com/dd/padeldescriptor/
2. Prepare SMILES file: `data/processed/metabolite_smiles.smi` (one SMILES per line)
3. Run PaDEL (Java):
   ```bash
   java -jar PaDEL-Descriptor.jar \
       -dir data/processed/metabolite_smiles.smi \
       -file data/processed/padel_descriptors.csv \
       -descriptortypes data/processed/padel_config.xml \
       -removesalt -standardizenitro -fingerprints
   ```
4. The wrapper `utils/padel_wrapper.py` handles this programmatically.

---

## 3. KEGG Immune Pathways (for IPS network)

The 20 KEGG immune pathways used (Supplementary Table S9):

| Pathway ID | Name |
|---|---|
| hsa04610 | Complement and coagulation cascades |
| hsa04611 | Platelet activation |
| hsa04620 | Toll-like receptor signalling |
| hsa04621 | NOD-like receptor signalling |
| hsa04622 | RIG-I-like receptor signalling |
| hsa04623 | Cytosolic DNA-sensing pathway |
| hsa04625 | C-type lectin receptor signalling |
| hsa04640 | Hematopoietic cell lineage |
| hsa04650 | Natural killer cell mediated cytotoxicity |
| hsa04657 | IL-17 signalling pathway |
| hsa04658 | Th1 and Th2 cell differentiation |
| hsa04659 | Th17 cell differentiation |
| hsa04660 | T cell receptor signalling |
| hsa04662 | B cell receptor signalling |
| hsa04664 | Fc epsilon RI signalling |
| hsa04666 | Fc gamma R-mediated phagocytosis |
| hsa04670 | Leukocyte transendothelial migration |
| hsa04672 | Intestinal immune network for IgA production |
| hsa04062 | Chemokine signalling pathway |
| hsa04060 | Cytokine-cytokine receptor interaction |

Download compound-pathway mappings:
```bash
python utils/data_loader.py --download_kegg --output_dir data/processed/
```

---

## 4. AIDA Autoantibody Data (for APS network)

Access the AIDA (Autoimmune Disease-Related Autoantibody) database at:
https://aida.medizin.uni-greifswald.de

Export the autoantibody-disease table and save as:
`data/processed/aida_autoantibodies.csv`
with columns: `disease`, `autoantibody_target`

---

## 5. Sample Data

`data/sample_associations.csv` contains 10 rows for pipeline testing:

```
met_idx,dis_idx,label,hmdb_id,metabolite_name,disease
0,0,1,HMDB0000162,L-Proline,Rheumatoid Arthritis
1,0,1,HMDB0000641,L-Glutamine,Rheumatoid Arthritis
2,1,1,HMDB0000122,D-Glucose,Multiple Sclerosis
...
```
