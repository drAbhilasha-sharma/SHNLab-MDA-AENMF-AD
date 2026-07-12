"""
evaluate.py  —  MDA-AENMF-AD Evaluation Entry Point
=====================================================
Usage:
    python evaluate.py --mode all --config configs/default.yaml
    python evaluate.py --mode lodo
    python evaluate.py --mode cv
    python evaluate.py --mode permutation
"""

import argparse
import yaml
import numpy as np
import logging
import os
from pathlib import Path

from src.models.mda_aenmf_ad import MDAAENMFAD
from src.data.dataset import AutoimmuneDataset
from src.evaluation.cross_validation import run_cv
from src.evaluation.lodo import run_lodo
from src.evaluation.permutation import run_permutation_test


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--mode',   default='all',
                   choices=['all', 'cv', 'lodo', 'permutation'])
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('evaluate')

    ds = AutoimmuneDataset(cfg)
    ds.load()
    snf_matrix = np.load('data/processed/snf_consensus.npy')
    results_dir = Path(cfg['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ('all', 'cv'):
        log.info('Running 5-fold CV ...')
        cv_df = run_cv(cfg, ds.pairs, snf_matrix, ds.ecfp4, ds.edge_index,
                        ds.disease_names, ds.metabolite_names)
        cv_df.to_csv(results_dir / 'metrics/cv_results.csv', index=False)
        log.info(f'  CV mean AUC: {cv_df["auc"].mean():.4f}')

    if args.mode in ('all', 'lodo'):
        log.info('Running LODO evaluation ...')
        lodo_df = run_lodo(cfg, ds.pairs, snf_matrix, ds.ecfp4, ds.edge_index,
                            ds.disease_names, ds.metabolite_names)
        lodo_df.to_csv(results_dir / 'metrics/lodo_results.csv', index=False)
        log.info('  LODO results saved.')

    if args.mode in ('all', 'permutation'):
        log.info('Running Y-randomisation ...')
        perm_df = run_permutation_test(
            cfg, ds.pairs, snf_matrix, ds.ecfp4, ds.edge_index,
            ds.disease_names, ds.metabolite_names,
            n_runs=cfg['evaluation']['n_permutations'])
        perm_df.to_csv(results_dir / 'metrics/permutation_results.csv', index=False)
        log.info(f'  Y-rand mean AUC: {perm_df["perm_auc"].mean():.4f}')


if __name__ == '__main__':
    main()
