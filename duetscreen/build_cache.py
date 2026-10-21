"""Build the training record cache (variants_data.pkl) exactly as used for the
released runs: prepare the delivered train/val split into segments, generate the
supervised views (synonymous/codon-conservative and fragment-confusion variants,
plus random-mutation contrastive negatives), and write one record list.

The record list is the input expected by `duetscreen.train`. Generation uses the
same module and the same configuration fields as the released training runs; the only
difference from the original host is where the files live.

Example
-------
python -m duetscreen.build_cache \
    --train-fasta data/train.fasta --val-fasta data/val.fasta \
    --output variants_data.pkl --seed 42 --workers 8
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import time

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the DuetScreen training cache.')
    parser.add_argument('--train-fasta', required=True)
    parser.add_argument('--val-fasta', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--max-semantic-attempts', type=int, default=None,
                        help='override the configured 1000 attempts (debug/smoke only)')
    args = parser.parse_args(argv)

    from .config import config
    from . import preprocessing
    from . import training_views

    config.prepared_train_fasta = args.train_fasta
    config.prepared_val_fasta = args.val_fasta
    config.use_existing_variants = False
    config.symmetric_variants = True
    config.downsample_ratio = 0.0

    random.seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    segments = preprocessing.preprocess_prepared_split(config)
    if not segments or not segments.get('train') or not segments.get('val'):
        raise SystemExit('preprocessing produced no usable segments')
    flat = [seg for split in ('train', 'val') for seg in segments.get(split, [])]
    positives = [s for s in flat if s['label'] == 1]
    negatives = [s for s in flat if s['label'] == 0]
    print(f'segments: {len(flat)} (pos {len(positives)} / neg {len(negatives)})', flush=True)

    variant_config = config.get_variant_generation_config()
    variant_config['num_workers'] = args.workers
    if args.max_semantic_attempts is not None:
        variant_config['max_semantic_attempts'] = args.max_semantic_attempts
    print('variant config: ' + ', '.join(f'{k}={v}' for k, v in variant_config.items()), flush=True)

    records = []
    if positives:
        pos_results = training_views.generate_triplet_views_parallel(positives, **variant_config)
        for seg, result in zip(positives, pos_results):
            result['split'] = seg.get('split')
            records.append(result)
    if negatives:
        print('symmetric augmentation: generating views for negatives (label 0)', flush=True)
        neg_results = training_views.generate_triplet_views_parallel(negatives, **variant_config)
        for seg, result in zip(negatives, neg_results):
            result['split'] = seg.get('split')
            result['contrastive_negative_views'] = []
            result['num_contrastive_negative_views'] = 0
            result['has_variants'] = len(result.get('positive_views') or []) > 0
            records.append(result)

    with open(args.output, 'wb') as f:
        pickle.dump(records, f, protocol=4)

    n_views = sum(len(r.get('positive_views') or []) for r in records)
    n_conf = sum(1 for r in records if 'confusion' in (r.get('view_types') or []))
    print(f'cache written: {args.output}')
    print(f'records={len(records)} views={n_views} with_confusion={n_conf} '
          f'elapsed={time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
