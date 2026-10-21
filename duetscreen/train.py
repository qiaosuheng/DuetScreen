"""Train the released DuetScreen model from a prepared record cache.

The training algorithm mirrors the protocol described in the paper: frozen
NT-V3-Pre-8M backbone (only the deconvolution tower is trainable), a dual-stream
downstream network, the 16-segment ESM2-35M protein channel, and the same
supervised multi-view objective used for every released seed.

The record cache is a pickle produced by the user's preparation pipeline; its
schema is documented in README.md. This repository intentionally does not
include any data-generation or sequence-obfuscation operator.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import time
import traceback
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from torch import nn

from .config import config
from . import model as model_module
from . import trainer as trainer_module
from .data import RealtimeSequenceDataset, collate_nucleotide_features
from .reproducibility import seed_training


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False))
    temporary.replace(path)


def resolved_config():
    return {name: jsonable(getattr(config, name)) for name in dir(config)
            if not name.startswith('_') and not callable(getattr(config, name))}


def audit_model(model):
    encoder = getattr(model, 'token_encoder', None)
    if encoder is None:
        encoder = model.nucleotide_embedding
    trainable = [(name, p.numel()) for name, p in encoder.named_parameters() if p.requires_grad]
    assert trainable and all('deconv_tower_blocks' in name for name, _ in trainable), trainable
    enc_ids = {id(p) for p in encoder.parameters()}
    downstream = sum(p.numel() for p in model.parameters() if id(p) not in enc_ids)
    assert downstream <= 4_000_000, downstream
    return {'downstream_parameters': downstream,
            'encoder_trainable': sum(n for _, n in trainable),
            'trainable_encoder_names': trainable,
            'model_summary': model.get_model_summary()}


class AuditedTrainer(trainer_module.NucleotideMambaTrainer):
    def create_model(self):
        result = super().create_model()
        metadata = audit_model(result)
        metadata.update({'loss_weights': self.loss_weights,
                         'resolved_config': resolved_config(),
                         'torch_version': torch.__version__, 'cuda': torch.version.cuda,
                         'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                         'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                         'job_source_sha256': sha(__file__)})
        write(Path(self.config.output_dir) / 'model_audit.json', metadata)
        return result

    def create_optimizer(self, model):
        optimizer, scheduler = super().create_optimizer(model)
        value = {'optimizer': 'AdamW', 'betas': optimizer.defaults['betas'],
                 'groups_after_scheduler_init': [
                     {k: v for k, v in group.items() if k != 'params'}
                     for group in optimizer.param_groups],
                 'scheduler_state': scheduler.state_dict()}
        write(Path(self.config.output_dir) / 'optimizer_audit.json', value)
        return optimizer, scheduler

    def train_epoch(self, *args, **kwargs):
        start = time.time()
        result = super().train_epoch(*args, **kwargs)
        assert math.isfinite(result['loss'])
        with (Path(self.config.output_dir) / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase': 'train', 'utc': time.time(),
                                     'seconds': time.time() - start,
                                     'metrics': result}, allow_nan=False) + '\n')
        return result

    def validate(self, *args, **kwargs):
        start = time.time()
        result = super().validate(*args, **kwargs)
        assert math.isfinite(result['loss'])
        with (Path(self.config.output_dir) / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase': 'validation', 'utc': time.time(),
                                     'seconds': time.time() - start,
                                     'metrics': result}, allow_nan=False) + '\n')
        return result


def configure(args):
    destination = Path(args.output_dir).expanduser().resolve()
    overrides = dict(
        block_type='v6:DuetScreen',
        pool_name='duetscreen',
        seed=args.seed,
        output_dir=str(destination),
        run_mode='train-only',
        objective_reweight=False,
        objective_no_path=False,
        pure_task_mode=False,
        symmetric_variants=True,
        use_existing_variants=True,
        variants_file_path=str(Path(args.records).expanduser().resolve()),
        transformer_model_repo=str(Path(args.ntv3_model).expanduser().resolve()),
        prepared_train_fasta=args.train_fasta,
        prepared_val_fasta=args.val_fasta,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_seq_length=args.max_seq_len,
        num_epochs=args.epochs,
        early_stop_patience=args.patience,
        freeze_transformer=True,
        n_layer=2,
        classifier_head='mlp',
    )
    external = {}
    for spec in args.external_test:
        name, _, path = spec.partition('=')
        if not name or not path:
            raise SystemExit(f'--external-test expects NAME=PATH, got {spec!r}')
        external[name] = path
    if not external:
        split_dir = Path(args.train_fasta).expanduser().resolve().parent
        defaults = [('nmi_test', 'eval_test.fasta'),
                    ('nmi_species', 'eval_generalization_species.fasta'),
                    ('nmi_family', 'eval_generalization_family.fasta')]
        for name, filename in defaults:
            candidate = split_dir / filename
            if candidate.is_file():
                external[name] = str(candidate)
    if not external and not args.smoke:
        raise SystemExit(
            'independent evaluation FASTA not found; pass --external-test NAME=PATH '
            '(the leakage audit needs at least one)')
    overrides['independent_test_files'] = external
    if args.no_contrastive:
        overrides['objective_no_contrastive'] = True
    for key, value in overrides.items():
        setattr(config, key, value)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def load_cache(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def smoke(args):
    destination = configure(args)
    seed_training(args.seed)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    trainer = AuditedTrainer(config)
    model = trainer.create_model()
    model.train()
    records = load_cache(config.variants_file_path)
    chosen = []
    for label in (0, 1):
        chosen.extend([r for r in records
                       if r.get('split') == 'train'
                       and r.get('original_label', r.get('label')) == label][:8])
    if len(chosen) < 16:
        raise SystemExit(f'smoke needs at least 8 train records per class (got {len(chosen)}); '
                         'build a larger cache or run full training instead')
    batch_size = len(chosen)
    ds = RealtimeSequenceDataset(chosen, max_seq_len=config.max_seq_len, pure_task=False)
    batch = trainer._move_batch_to_device(
        collate_nucleotide_features([ds[i] for i in range(batch_size)]))
    classification, contrastive = trainer.create_loss_functions()
    weights = trainer_module.build_loss_weights(config)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.amp.autocast(device_type, enabled=device_type == 'cuda'):
        output = model(original_sequences=batch['sequences'], labels=batch['labels'],
                       semantic_sequences=batch['semantic_sequences'],
                       confusion_sequences=batch['confusion_sequences'],
                       random_mutation_sequences=batch['random_mutation_sequences'],
                       training_mode=True)
        z0, z1, z2, aux, f0, features = output
        pos = batch['labels'] == 1
        ff = features.split(batch_size, dim=0)
        bce = (weights['orig'] * classification(z0.reshape(-1), batch['labels'])
               + weights['sem'] * classification(z1.reshape(-1), batch['labels'])
               + weights['conf'] * classification(z2.reshape(-1), batch['labels']))
        cl = contrastive(f0[pos], [ff[1][pos], ff[2][pos]], ff[3][pos])
        loss = (bce + weights['contrast'] * cl + weights['path'] * aux) / weights['norm']
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all().item() for g in gradients)
    item = {'smoke': 'passed', 'loss': float(loss), 'loss_weights': weights,
            'peak_allocated_gib': (torch.cuda.max_memory_allocated() / 2 ** 30)
            if torch.cuda.is_available() else None,
            **{k: v for k, v in audit_model(model).items() if k != 'model_summary'},
            'downstream_parameters': audit_model(model)['downstream_parameters']}
    write(destination / 'smoke.json', item)
    print('SMOKE_RESULT ' + json.dumps({k: v for k, v in item.items()
                                        if k != 'trainable_encoder_names'}), flush=True)
    del gradients, output, features, ff, f0, z0, z1, z2, aux, bce, cl, loss, model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(args):
    destination = configure(args)
    assert not (destination / 'complete.json').exists(), 'Completed run must not be retrained'
    assert not (destination / 'started.json').exists(), 'Partial run exists; explicit recovery required'
    start = time.time()
    write(destination / 'started.json',
          {'seed': args.seed, 'utc': start, 'pid': os.getpid(),
           'job_source_sha256': sha(__file__), 'config': resolved_config()})
    try:
        trainer = AuditedTrainer(config)
        model, train_history, val_history = trainer.train(load_cache(config.variants_file_path))
        assert train_history and len(train_history) == len(val_history)
        checkpoint = destination / 'best_nucleotide_v3_mamba_model.pth'
        assert checkpoint.exists()
        write(destination / 'complete.json',
              {'seed': args.seed, 'utc': time.time(), 'seconds': time.time() - start,
               'epochs': len(train_history),
               'selected_validation_combined_acc': max(v['combined_acc'] for v in val_history),
               'best_checkpoint_sha256': sha(checkpoint),
               'job_source_sha256': sha(__file__)})
    except Exception as exc:  # noqa: BLE001 - record every failure verbatim
        write(destination / 'failure.json',
              {'seed': args.seed, 'utc': time.time(), 'error_type': type(exc).__name__,
               'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def build_parser():
    parser = argparse.ArgumentParser(description='Train DuetScreen on a prepared record cache.')
    parser.add_argument('--records', required=True, help='pickle with the prepared record list')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ntv3-model', required=True, help='local folder of the NT-V3-Pre-8M checkpoint')
    parser.add_argument('--train-fasta', default='', help='optional, for split bookkeeping only')
    parser.add_argument('--val-fasta', default='', help='optional, for split bookkeeping only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-seq-len', type=int, default=2048)
    parser.add_argument('--external-test', action='append', default=[],
                        metavar='NAME=PATH',
                        help='independent evaluation FASTA for the leakage audit; '
                             'repeatable. Defaults to eval_test.fasta and the two '
                             'generalization fastas next to the training split')
    parser.add_argument('--no-contrastive', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='run one forward/backward check only')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    assert 'DUETSCREEN_PROTEIN_SEG_CACHE' in os.environ, \
        'Set DUETSCREEN_PROTEIN_SEG_CACHE to the 16-segment ESM2-35M cache built by duetscreen.protein_cache'
    if args.smoke:
        smoke(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
