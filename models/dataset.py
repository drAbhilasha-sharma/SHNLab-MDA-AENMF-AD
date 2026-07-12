"""
AutoimmuneDataset
==================
Loads HMDB associations, builds negative samples, generates ECFP4
fingerprints, and constructs the metabolite interaction graph.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from pathlib import Path
from typing import List, Tuple, Optional


class AutoimmuneDataset:
    """
    Parameters
    ----------
    config : dict  (loaded from configs/default.yaml)
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.pairs:            Optional[pd.DataFrame] = None
        self.assoc_matrix:     Optional[np.ndarray]   = None
        self.ecfp4:            Optional[np.ndarray]   = None
        self.edge_index:       Optional[np.ndarray]   = None
        self.disease_names:    List[str] = []
        self.metabolite_names: List[str] = []

    # ── Loading ───────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Full load pipeline: associations → negatives → fingerprints → graph."""
        self._load_associations()
        self._add_negatives()
        self._build_ecfp4()
        self._build_graph()

    def _load_associations(self) -> None:
        df = pd.read_csv(self.cfg['data']['hmdb_path'],
                         encoding='utf-8', errors='replace')
        # Expected columns: disease, metabolite (+ optional rank/score)
        df = df[['disease', 'metabolite']].drop_duplicates()
        df['label'] = 1
        self.disease_names    = sorted(df['disease'].unique().tolist())
        self.metabolite_names = sorted(df['metabolite'].unique().tolist())

        # Build binary association matrix (n_diseases × n_metabolites)
        nd, nm = len(self.disease_names), len(self.metabolite_names)
        d2i = {d: i for i, d in enumerate(self.disease_names)}
        m2i = {m: i for i, m in enumerate(self.metabolite_names)}
        self.assoc_matrix = np.zeros((nd, nm), dtype=np.float32)
        for _, row in df.iterrows():
            self.assoc_matrix[d2i[row['disease']], m2i[row['metabolite']]] = 1.0

        self._pos_pairs = df[['disease', 'metabolite', 'label']].copy()

    def _add_negatives(self) -> None:
        """Stratified negative sampling: 1:1 per disease."""
        rng = np.random.default_rng(self.cfg['data']['random_seed'])
        neg_rows = []
        d2i = {d: i for i, d in enumerate(self.disease_names)}
        m2i = {m: i for i, m in enumerate(self.metabolite_names)}
        ratio = self.cfg['data']['neg_sampling_ratio']

        for d in self.disease_names:
            pos_mets = set(
                self._pos_pairs[self._pos_pairs['disease'] == d]['metabolite'])
            neg_pool = [m for m in self.metabolite_names if m not in pos_mets]
            n_neg    = int(len(pos_mets) * ratio)
            chosen   = rng.choice(neg_pool,
                                  size=min(n_neg, len(neg_pool)),
                                  replace=False)
            for m in chosen:
                neg_rows.append({'disease': d, 'metabolite': m, 'label': 0})

        self.pairs = pd.concat(
            [self._pos_pairs, pd.DataFrame(neg_rows)],
            ignore_index=True).sample(frac=1,
                                       random_state=self.cfg['data']['random_seed'])

    def _build_ecfp4(self) -> None:
        """Compute ECFP4 (1024-bit) fingerprints from SMILES."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            smiles_df = pd.read_csv(self.cfg['data']['smiles_path'])
            s2smi     = dict(zip(smiles_df['metabolite'], smiles_df['smiles']))
        except ImportError:
            print('RDKit not available — using zero fingerprints as placeholder.')
            s2smi = {}

        nbits = self.cfg['data'].get('ecfp_nbits', 1024)
        fps   = []
        for m in self.metabolite_names:
            smi = s2smi.get(m, '')
            if smi:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    fp  = AllChem.GetMorganFingerprintAsBitVect(
                              mol, radius=self.cfg['data'].get('ecfp_radius', 2),
                              nBits=nbits)
                    fps.append(np.array(fp))
                except Exception:
                    fps.append(np.zeros(nbits))
            else:
                fps.append(np.zeros(nbits))
        self.ecfp4 = np.array(fps, dtype=np.float32)

    def _build_graph(self) -> None:
        """Build metabolite co-occurrence graph from KEGG pathway membership."""
        nm = len(self.metabolite_names)
        m2i = {m: i for i, m in enumerate(self.metabolite_names)}
        # Fallback: fully connect metabolites sharing at least one HMDB disease
        G  = nx.Graph()
        G.add_nodes_from(range(nm))

        # Use co-association in same disease as proxy edge
        d2mets = {}
        for _, row in self._pos_pairs.iterrows():
            d2mets.setdefault(row['disease'], set()).add(row['metabolite'])
        for mets in d2mets.values():
            mets = [m2i[m] for m in mets if m in m2i]
            for a in range(len(mets)):
                for b in range(a + 1, len(mets)):
                    G.add_edge(mets[a], mets[b])

        edges = list(G.edges())
        if edges:
            src, dst = zip(*edges)
            # Undirected: add both directions
            self.edge_index = np.array(
                [list(src) + list(dst), list(dst) + list(src)], dtype=np.int64)
        else:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)

    def build_pair_features(self, nmf) -> Tuple[np.ndarray, np.ndarray]:
        """Build (n_pairs, 2*rank) feature matrix and labels from NMF factors."""
        d2i = {d: i for i, d in enumerate(self.disease_names)}
        m2i = {m: i for i, m in enumerate(self.metabolite_names)}
        X, y = [], []
        for _, row in self.pairs.iterrows():
            if row['disease'] not in d2i or row['metabolite'] not in m2i:
                continue
            di = d2i[row['disease']]; mi = m2i[row['metabolite']]
            X.append(np.concatenate([nmf.W_disease[di], nmf.W_metabolite[mi]]))
            y.append(row['label'])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
