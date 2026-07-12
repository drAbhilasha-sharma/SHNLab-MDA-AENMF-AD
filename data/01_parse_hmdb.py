#!/usr/bin/env python3
"""
=============================================================================
SCRIPT 1 — parse_hmdb.py
=============================================================================
PURPOSE : Stream-parse the HMDB v5.0 XML bulk download and extract all
          metabolite–disease associations for 8 target autoimmune disorders.
INPUT   : hmdb_metabolites.xml  (~20 GB uncompressed)
            Download: wget https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip
            Then    : unzip hmdb_metabolites.zip
OUTPUT  : hmdb_autoimmune_associations.csv
          hmdb_metabolite_smiles.csv  (unique metabolites + SMILES)
RUNTIME : ~10–20 min on a modern laptop (streaming, low memory)
REQUIRES: pip install lxml pandas
=============================================================================
"""

import csv
import sys
import os
import time
import pandas as pd
from lxml import etree

# ─── Configuration ────────────────────────────────────────────────────────────
HMDB_XML   = "hmdb_metabolites.xml"
OUT_ASSOC  = "hmdb_autoimmune_associations.csv"
OUT_SMILES = "hmdb_metabolite_smiles.csv"

# Terms used to identify autoimmune diseases in HMDB disease name field
# (case-insensitive substring matching)
AUTOIMMUNE_TERMS = [
    "rheumatoid arthritis",
    "arthritis rheumatoid",
    "systemic lupus erythematosus",
    "lupus erythematosus systemic",
    "lupus erythematosus",
    "multiple sclerosis",
    "sclerosis multiple",
    "crohn",
    "ulcerative colitis",
    "inflammatory bowel",
    "type 1 diabetes",
    "diabetes mellitus type 1",
    "diabetes mellitus, type 1",
    "insulin-dependent diabetes",
    "juvenile diabetes",
    "psoriasis",
    "ankylosing spondylitis",
    "spondylitis ankylosing",
    "sjogren",
    "sjögren",
    "sicca syndrome",
]

# Standard disease names → canonical label
DISEASE_STANDARDIZE = {
    "rheumatoid arthritis":              "Rheumatoid Arthritis",
    "arthritis rheumatoid":              "Rheumatoid Arthritis",
    "systemic lupus erythematosus":      "Systemic Lupus Erythematosus",
    "lupus erythematosus systemic":      "Systemic Lupus Erythematosus",
    "lupus erythematosus":               "Systemic Lupus Erythematosus",
    "multiple sclerosis":                "Multiple Sclerosis",
    "sclerosis multiple":                "Multiple Sclerosis",
    "crohn disease":                     "Inflammatory Bowel Disease",
    "crohn's disease":                   "Inflammatory Bowel Disease",
    "ulcerative colitis":                "Inflammatory Bowel Disease",
    "inflammatory bowel disease":        "Inflammatory Bowel Disease",
    "type 1 diabetes":                   "Type 1 Diabetes",
    "diabetes mellitus type 1":          "Type 1 Diabetes",
    "diabetes mellitus, type 1":         "Type 1 Diabetes",
    "insulin-dependent diabetes mellitus":"Type 1 Diabetes",
    "juvenile diabetes":                 "Type 1 Diabetes",
    "psoriasis":                         "Psoriasis",
    "ankylosing spondylitis":            "Ankylosing Spondylitis",
    "spondylitis ankylosing":            "Ankylosing Spondylitis",
    "sjogren syndrome":                  "Sjögren's Syndrome",
    "sjögren syndrome":                  "Sjögren's Syndrome",
    "sjogren's syndrome":                "Sjögren's Syndrome",
    "sicca syndrome":                    "Sjögren's Syndrome",
}

TARGET_DISEASES = {
    "Rheumatoid Arthritis",
    "Systemic Lupus Erythematosus",
    "Multiple Sclerosis",
    "Inflammatory Bowel Disease",
    "Type 1 Diabetes",
    "Psoriasis",
    "Ankylosing Spondylitis",
    "Sjögren's Syndrome",
}

# HMDB XML namespace
NS  = "http://www.hmdb.ca"
T   = lambda tag: f"{{{NS}}}{tag}"

# ─── Helpers ──────────────────────────────────────────────────────────────────
def standardize_disease(raw_name: str):
    """Map raw HMDB disease string to canonical target disease or None."""
    low = raw_name.lower().strip()
    # Direct lookup
    if low in DISEASE_STANDARDIZE:
        return DISEASE_STANDARDIZE[low]
    # Substring fallback
    for term, std in DISEASE_STANDARDIZE.items():
        if term in low:
            return std
    return None


def is_autoimmune(disease_name: str) -> bool:
    """Quick pre-filter before full standardization."""
    low = disease_name.lower()
    return any(t in low for t in AUTOIMMUNE_TERMS)


# ─── Main Parser ──────────────────────────────────────────────────────────────
def parse_hmdb(xml_path: str):
    """
    Stream-parse HMDB XML using iterparse so the full 20 GB file
    is never loaded into memory. Yields one metabolite at a time.

    Returns
    -------
    assoc_rows  : list[dict]  — one row per (metabolite, disease) pair
    smiles_rows : list[dict]  — one row per unique metabolite with SMILES
    """
    if not os.path.exists(xml_path):
        sys.exit(f"ERROR: {xml_path} not found.\n"
                 "Download with:\n"
                 "  wget https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip\n"
                 "  unzip hmdb_metabolites.zip")

    assoc_rows  = []
    smiles_rows = []
    n_total     = 0
    n_with_ad   = 0
    t0          = time.time()

    context = etree.iterparse(xml_path, events=("end",), tag=T("metabolite"))

    for event, elem in context:
        n_total += 1

        # ── Extract core fields ──────────────────────────────────────────
        accession = elem.findtext(T("accession")) or ""
        name      = elem.findtext(T("name"))      or ""
        smiles    = elem.findtext(T("smiles"))    or ""
        inchikey  = elem.findtext(T("inchikey"))  or ""
        status    = elem.findtext(T("status"))    or ""

        # Filter: endogenous metabolites with a valid SMILES string
        if not smiles or not accession:
            elem.clear()
            continue

        # ── Extract disease associations ─────────────────────────────────
        diseases_el = elem.find(T("diseases"))
        if diseases_el is not None:
            met_has_ad = False
            for dis_el in diseases_el.findall(T("disease")):
                dis_name_raw = dis_el.findtext(T("name")) or ""
                if not is_autoimmune(dis_name_raw):
                    continue

                dis_std = standardize_disease(dis_name_raw)
                if dis_std not in TARGET_DISEASES:
                    continue

                # Collect PMIDs from reference elements
                pmids = []
                for ref in dis_el.findall(f".//{T('reference')}"):
                    pid = ref.findtext(T("pubmed_id")) or ""
                    if pid.strip():
                        pmids.append(pid.strip())

                assoc_rows.append({
                    "hmdb_id":       accession,
                    "metabolite":    name,
                    "smiles":        smiles,
                    "inchikey":      inchikey,
                    "disease":       dis_name_raw,
                    "disease_std":   dis_std,
                    "pmids":         "|".join(pmids),
                    "n_pmids":       len(pmids),
                })
                met_has_ad = True

            if met_has_ad:
                n_with_ad += 1
                smiles_rows.append({
                    "hmdb_id":   accession,
                    "metabolite":name,
                    "smiles":    smiles,
                    "inchikey":  inchikey,
                    "status":    status,
                })

        # Free memory — critical for large XML
        elem.clear()
        # Also clear parent references accumulated by iterparse
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        # Progress report
        if n_total % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  {n_total:>8,} metabolites scanned | "
                  f"{n_with_ad:>5} with AD associations | "
                  f"{len(assoc_rows):>5} total pairs | "
                  f"{elapsed/60:.1f} min elapsed")

    print(f"\n[Done] {n_total:,} metabolites scanned in {(time.time()-t0)/60:.1f} min")
    return assoc_rows, smiles_rows


# ─── Summary & save ───────────────────────────────────────────────────────────
def summarize_and_save(assoc_rows, smiles_rows):
    assoc_df  = pd.DataFrame(assoc_rows)
    smiles_df = pd.DataFrame(smiles_rows).drop_duplicates("hmdb_id")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Unique metabolites with AD association : {smiles_df.shape[0]}")
    print(f"  Total (metabolite, disease) pairs      : {assoc_df.shape[0]}")
    print(f"\n  Associations per disease:")
    for dis, cnt in assoc_df["disease_std"].value_counts().items():
        print(f"    {dis:<40} {cnt:>4}")
    print(f"\n  Metabolites per disease:")
    for dis, cnt in assoc_df.groupby("disease_std")["hmdb_id"].nunique().items():
        print(f"    {dis:<40} {cnt:>4}")
    print(f"\n  Average PMIDs per association : {assoc_df['n_pmids'].mean():.1f}")
    print(f"  Associations with ≥1 PMID    : {(assoc_df['n_pmids']>0).sum()}")
    print("="*60)

    assoc_df.to_csv(OUT_ASSOC,  index=False)
    smiles_df.to_csv(OUT_SMILES, index=False)
    print(f"\n  Saved: {OUT_ASSOC}")
    print(f"  Saved: {OUT_SMILES}")

    return assoc_df, smiles_df


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print(" HMDB Autoimmune Association Extractor")
    print("="*60)
    print(f" Input : {HMDB_XML}")
    print(f" Output: {OUT_ASSOC}")
    print(f"         {OUT_SMILES}")
    print("="*60)

    assoc_rows, smiles_rows = parse_hmdb(HMDB_XML)
    summarize_and_save(assoc_rows, smiles_rows)
