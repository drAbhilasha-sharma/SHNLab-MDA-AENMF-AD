"""
train.py  —  MDA-AENMF-AD Training Entry Point
================================================
Usage:
    python train.py --config configs/default.yaml [--seed 42]
"""

import argparse
import yaml
import numpy as np
import torch
import logging
import os
from pathlib import Path

from src.models.mda_aenmf_ad import MDAAENMFAD
from src.data.dataset import AutoimmuneDataset
from src.data.similarity import build_all_similarity_networks
from src.utils.snf import snf_fuse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=getattr(logging, cfg['output']['log_level']),
        format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('train')

    log.info('Loading dataset ...')
    ds = AutoimmuneDataset(cfg)
    ds.load()

    log.info('Building similarity networks ...')
    sim_nets = build_all_similarity_networks(cfg, ds.disease_names, ds.assoc_matrix)
    log.info(f'  Networks built: {list(sim_nets.keys())}')

    log.info('Running SNF fusion ...')
    snf_matrix = snf_fuse(list(sim_nets.values()),
                           k=cfg['model']['snf_k'],
                           t=cfg['model']['snf_t'])
    np.save(os.path.join(cfg['data'].get('processed_dir', 'data/processed'),
                          'snf_consensus.npy'), snf_matrix)
    log.info('  SNF complete.')

    log.info('Initialising model ...')
    model = MDAAENMFAD(cfg, len(ds.disease_names), len(ds.metabolite_names))
    model.disease_names    = ds.disease_names
    model.metabolite_names = ds.metabolite_names

    log.info('Pre-training DAE ...')
    model.pretrain(snf_matrix, ds.ecfp4, ds.edge_index)

    log.info('Building NMF co-embeddings ...')
    model.build_embeddings(snf_matrix, ds.ecfp4, ds.edge_index)

    log.info('Training MLP predictor ...')
    X, y = ds.build_pair_features(model.nmf)
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    losses = model.train_mlp(X, y, pos_weight=pos_weight)
    log.info(f'  Training complete. Final loss: {losses[-1]:.4f}')

    ckpt_dir = Path(cfg['output']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_dir / 'best_model.pt'))
    log.info(f'  Model saved to {ckpt_dir}/best_model.pt')


if __name__ == '__main__':
    main()
