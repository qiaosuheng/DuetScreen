"""Downstream network for the released dual-space model.

Production dependencies: torch >= 2.1 and the project's real mamba_ssm.Mamba2.
No vision weights, custom attention kernels, new loss, or fake Mamba fallback.
The official factories lock the agreed 256-d / 2-Mamba / 2048-token profile.
Algorithmic sources and deviations are cited in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from . import downstream_base as base


_RC_TABLE = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def reverse_complement(seq: str) -> str:
    """Reverse complement; non-ACGTN characters pass through (encoder cleans them)."""
    return seq.translate(_RC_TABLE)[::-1].upper()


# ---------------------------------------------------------------------------
# DuetScreen release: dual-space model with the protein segment channel.
#
# Optional training-time augmentation (both label preserving, default off):
#   DUETSCREEN_AUG_RC    reverse-complement flip of the original training view;
#   DUETSCREEN_AUG_DEOPT synonymous codon resampling inside the longest
#                        stop-free ORF interval, leaving the encoded protein
#                        unchanged. The protein channel stays reachable through
#                        the protein-keyed fallback index
#                        (DUETSCREEN_PROTEIN_INDEX).
# ---------------------------------------------------------------------------
import os as _os
import random as _random

_BASES = 'TCAG'
_AAS = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
_CODON_TABLE = {}
_i = 0
for _b1 in _BASES:
    for _b2 in _BASES:
        for _b3 in _BASES:
            _CODON_TABLE[_b1 + _b2 + _b3] = _AAS[_i]
            _i += 1
_SYNONYMS = {}
for _c, _a in _CODON_TABLE.items():
    if _a != '*':
        _SYNONYMS.setdefault(_a, []).append(_c)


def _translate(dna: str) -> str:
    return ''.join(_CODON_TABLE.get(dna[i:i + 3], 'X') for i in range(0, len(dna) - 2, 3))


def _best_interval_frame(seq, frame):
    best_start = best_len = 0
    cur = frame
    i = frame
    n = len(seq)
    while i + 3 <= n:
        if _CODON_TABLE.get(seq[i:i + 3]) == '*':
            if i - cur > best_len:
                best_start, best_len = cur, i - cur
            cur = i + 3
        i += 3
    tail = n - ((n - cur) % 3) - cur
    if tail > best_len:
        best_start, best_len = cur, tail
    return best_start, best_len


def _winning_orf(dna):
    """Longest stop-free interval across 6 frames (same policy as the cache
    precompute). Returns (strand_seq, start, length, is_rc) or None."""
    best = None
    best_prot_len = 0
    for s, is_rc in ((dna, False), (reverse_complement(dna), True)):
        for frame in range(3):
            st, ln = _best_interval_frame(s, frame)
            if ln >= 30 and ln // 3 > best_prot_len:
                best_prot_len = ln // 3
                best = (s, st, ln, is_rc)
    return best


def best_protein(dna: str) -> str:
    w = _winning_orf(dna.upper())
    if w is None:
        return ''
    s, st, ln, _ = w
    return _translate(s[st:st + ln])[:1000]


def _deopt_sequence(seq: str, p: float, rng) -> str:
    """Resample each codon inside the winning ORF interval with probability p
    to a uniformly chosen synonymous codon. Protein strictly unchanged."""
    dna = seq.upper()
    w = _winning_orf(dna)
    if w is None:
        return seq
    s, st, ln, is_rc = w
    chars = list(s)
    for i in range(st, st + ln, 3):
        aa = _CODON_TABLE.get(s[i:i + 3])
        if aa is None or aa == '*':
            continue
        if rng.random() < p:
            syn = _SYNONYMS[aa]
            chars[i:i + 3] = syn[rng.randrange(len(syn))]
    new = ''.join(chars)
    return reverse_complement(new) if is_rc else new




def maybe_augment(sequences):
    """Label-preserving train-time augmentation of the original view, driven by
    DUETSCREEN_AUG_RC / DUETSCREEN_AUG_DEOPT. No-op when both are unset/zero."""
    rc_p = float(_os.environ.get('DUETSCREEN_AUG_RC') or 0)
    deopt_p = float(_os.environ.get('DUETSCREEN_AUG_DEOPT') or 0)
    if rc_p <= 0 and deopt_p <= 0:
        return sequences
    out = []
    for s in sequences:
        if rc_p > 0 and _random.random() < rc_p:
            s = reverse_complement(s)
        if deopt_p > 0:
            s = _deopt_sequence(s, deopt_p, _random)
        out.append(s)
    return out


class ProteinFeatureCache:
    """Lazy singleton lookup of precomputed frozen ESM2-8M mean-pooled embeddings.

    The cache maps exact DNA sequence strings (the same strings the training
    cache and eval fastas contain) to fp16 [320] vectors. Sequences without an
    entry (or without a >=30nt stop-free interval) map to a zero vector; the
    downstream projection learns the 'no protein channel' bias from zeros.
    """
    _instance = None
    DIM = 320  # fallback only; the real dimension is inferred from the loaded table

    def __init__(self, path):
        import pickle
        import numpy as np
        with open(path, 'rb') as f:
            self.table = pickle.load(f)
        self.dim = len(next(iter(self.table.values()))) if self.table else self.DIM
        self.zero = np.zeros(self.dim, dtype=np.float16)
        # Optional protein-keyed fallback index: maps the ORF
        # protein string to its embedding, so augmented (RC / synonymous
        # codon-resampled) sequences still hit the protein channel.
        self.prot_index = None
        idx = _os.environ.get('DUETSCREEN_PROTEIN_INDEX')
        if idx:
            with open(idx, 'rb') as f:
                self.prot_index = pickle.load(f)

    @classmethod
    def get(cls):
        if cls._instance is None:
            import os
            default = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                   'protein_cache_esm2_8m.pkl')
            cls._instance = cls(os.environ.get('DUETSCREEN_PROTEIN_CACHE', default))
        return cls._instance

    def _lookup_one(self, s):
        v = self.table.get(s)
        if v is not None:
            return v
        return self.prot_index.get(best_protein(s), self.zero)

    def lookup(self, seqs, device, dtype):
        import numpy as np
        if self.prot_index is None:
            arr = np.stack([self.table.get(s, self.zero) for s in seqs])
        else:
            arr = np.stack([self._lookup_one(s) for s in seqs])
        return torch.from_numpy(arr).to(device=device, dtype=dtype)


class SegmentProteinCache:
    """Lazy singleton for the released model: fp16 [16, dim] segment-pooled ESM2
    embeddings per DNA sequence (coarse positional ORF profile). Missing
    sequences map to a zero [16, dim] block, same convention as the flat cache.
    """
    _instance = None

    def __init__(self, path):
        import pickle
        import numpy as np
        with open(path, 'rb') as f:
            self.table = pickle.load(f)
        first = next(iter(self.table.values()))
        self.segs, self.dim = first.shape
        self.zero = np.zeros((self.segs, self.dim), dtype=np.float16)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls(_os.environ['DUETSCREEN_PROTEIN_SEG_CACHE'])
        return cls._instance

    def lookup(self, seqs, device, dtype):
        import numpy as np
        arr = np.stack([self.table.get(s, self.zero) for s in seqs])
        return torch.from_numpy(arr).to(device=device, dtype=dtype)


BUDGET = int(_os.environ.get('DUETSCREEN_BUDGET_OVERRIDE', '4000000'))  # ALL downstream parameters, including BOTH output heads.


@dataclass(frozen=True)
class Candidate:
    id: str
    name: str
    venue: str
    source: str
    mechanism: str


MODELS = (
    Candidate('DuetScreen', 'SegmentSelfAttnProteinStream', 'DuetScreen release',
              'see paper; frozen ESM2-35M segment encoder',
              'Final model: frozen NTv3 nucleotide encoder -> dual-stream downstream '
              'network; 16 segment-pooled ESM2-35M protein embeddings pass one '
              'self-attention block and a nucleotide-conditioned cross-attention '
              'readout; the conditioned protein summary is late-fused with the '
              'pooled nucleotide representation.'),
)
MODELS_BY_ID = {c.id: c for c in MODELS}


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def downstream_parameters(model: nn.Module) -> int:
    excluded = {id(p) for p in model.token_encoder.parameters()}
    return sum(p.numel() for p in model.parameters() if id(p) not in excluded)


def _init_new_module(module: nn.Module):
    # Initializer conventions, applied only to newly created modules.
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def token_mean(x: torch.Tensor) -> torch.Tensor:
    # Preserve the original output dtype while accumulating AMP values in fp32.
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    return x.mean(1, dtype=dtype).to(x.dtype)






class FeedForward(nn.Module):
    def __init__(self, d: int, hidden: Optional[int] = None, dropout: float = .1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden or 2*d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden or 2*d, d),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TokenFusion(nn.Module):
    """Nonlinear fusion BEFORE pooling, with a fixed nonzero residual scale.

    Zeroing the secondary summary/input here is a dependency diagnostic, NOT
    an independently trained single-branch ablation.
    """
    def __init__(self, d: int, scale: float = .2):
        super().__init__()
        self.main_norm, self.side_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.in_proj = nn.Linear(2*d, d)
        self.out_proj = nn.Linear(d, d)
        self.scale = scale  # fixed, cannot be decayed/frozen by the optimizer

    def forward(self, main, side):
        delta = self.out_proj(F.gelu(self.in_proj(torch.cat(
            (self.main_norm(main), self.side_norm(side)), dim=-1))))
        return main + self.scale * delta








class GRN1D(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, d))
        self.beta = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x):
        gx = torch.linalg.vector_norm(x.float(), dim=1, keepdim=True)
        nx = gx / (gx.mean(-1, keepdim=True) + 1e-6)
        return x + (self.gamma * x * nx.to(x.dtype) + self.beta).to(x.dtype)


class ConvNeXtV2Block1D(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.expand, self.project = nn.Linear(d, 2*d), nn.Linear(2*d, d)
        self.grn, self.drop = GRN1D(2*d), nn.Dropout(.1)

    def forward(self, x):
        y = self.norm(self.dw(x.transpose(1, 2)).transpose(1, 2))
        return x + self.drop(self.project(self.grn(F.gelu(self.expand(y)))))




class Attention(nn.Module):
    """Torch SDPA; no full LxL attention in the configured full-length branch."""
    def __init__(self, d: int, heads: int = 4, dropout: float = .1):
        super().__init__()
        if d % heads:
            raise ValueError('d must be divisible by heads')
        self.q, self.k, self.v, self.out = [nn.Linear(d, d) for _ in range(4)]
        self.heads, self.dropout = heads, dropout

    def forward(self, query, context, bias=None):
        b, n, d = query.shape
        def split(x):
            return x.reshape(b, -1, self.heads, d//self.heads).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(context)), split(self.v(context))
        # bias may include differentiable relative position biases and -inf masks.
        if bias is not None:
            bias = bias.to(device=q.device, dtype=q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class AttentionBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ffn = Attention(d), FeedForward(d)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h)
        return x + self.ffn(self.norm2(x))




class GMLPBlock(nn.Module):
    def __init__(self, d: int, tokens: int):
        super().__init__()
        self.norm, self.gate_norm = nn.LayerNorm(d), nn.LayerNorm(2*d)
        self.up, self.down = nn.Linear(d, 4*d), nn.Linear(2*d, d)
        self.spatial = nn.Linear(tokens, tokens)
        self.drop = nn.Dropout(.1)

    def reset_special(self):
        nn.init.normal_(self.spatial.weight, std=1e-3)
        nn.init.ones_(self.spatial.bias)  # Required near-identity SGU initial gate.

    def forward(self, x):
        u, v = F.gelu(self.up(self.norm(x))).chunk(2, dim=-1)
        gate = self.spatial(self.gate_norm(v).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(self.down(u * gate))










class CrossAttentionBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_norm, self.context_norm, self.ffn_norm = [nn.LayerNorm(d) for _ in range(3)]
        self.attn, self.ffn = Attention(d), FeedForward(d)

    def forward(self, query, context):
        x = query + self.attn(self.q_norm(query), self.context_norm(context))
        return x + self.ffn(self.ffn_norm(x))










def make_branch(model_id, d, max_length, depth=2):
    """Released model: one ConvNeXt-style side branch (two blocks)."""
    if model_id != 'DuetScreen':
        raise ValueError(f'Unknown model {model_id}')
    return nn.Sequential(ConvNeXtV2Block1D(d), ConvNeXtV2Block1D(d))


class DuetScreenNetwork(base.DualStackNetwork):
    """Shared encoder/head and 4-/6-tuple contracts with the new sequence processing.

    Low-level constructor supports small-d tests. Use create_v6_network for
    official locked experiments. Changing dimensions creates a different run.
    """
    def __init__(self, model_id: str, **config):
        if model_id not in MODELS_BY_ID and model_id != 'F0':
            raise ValueError(f'Unknown model {model_id}')
        if config.get('compile_model', False):
            raise ValueError('Compile only after encoder attachment and validation')
        cfg = dict(base.BASE_NETWORK_CONFIG)
        cfg.update(config)
        kernel = config.get('main_kernel', 'mamba')
        cfg['block_type'] = {'mamba': 'dual_readout_mamba', 'conv': 'basic_conv',
                             'tr': 'basic_transformer'}[kernel]
        cfg.pop('main_kernel', None)
        cfg.pop('no_grn', None)
        if cfg['n_layer'] < 1 or cfg['block_type'] not in (
                'dual_readout_mamba', 'basic_conv', 'basic_transformer'):
            raise ValueError('Main stream: depth must be positive; '
                             'main_kernel must be mamba/conv/tr')
        super().__init__(**cfg)
        self.model_id = model_id
        self.block_type = 'v6:' + model_id + ('' if kernel == 'mamba' else ':' + kernel) + ('+nogrn' if config.get('no_grn') else '')
        # The old readout is removed, so it cannot bloat the count or optimizer.
        self.sequence_pooler = nn.Identity()
        # Initialize the common fusion FIRST, so different branch parameter
        # counts do not change its seed. F0 and T10 additionally share exact
        # initial Conv weights; only T10 then draws coupling parameters.
        self.fusion = TokenFusion(self.d_model)
        self.fusion.apply(_init_new_module)
        # Protein channel (released model): frozen ESM2-35M embeddings of 16
        # ORF segments -> projection -> segment self-attention ->
        # nucleotide-conditioned cross-attention readout -> late fusion.
        seg_cache = SegmentProteinCache.get()
        self.seg_proj = nn.Sequential(nn.Linear(seg_cache.dim, self.d_model),
                                      nn.GELU(),
                                      nn.Linear(self.d_model, self.d_model))
        self.seg_self_attn = AttentionBlock(self.d_model)
        self.pr_attn = CrossAttentionBlock(self.d_model)
        self.pr_norm = nn.LayerNorm(self.d_model)
        self.space_fuse = nn.Linear(2 * self.d_model, self.d_model)
        self.seg_proj.apply(_init_new_module)
        self.seg_self_attn.apply(_init_new_module)
        self.pr_attn.apply(_init_new_module)
        self.pr_norm.apply(_init_new_module)
        self.space_fuse.apply(_init_new_module)
        self.candidate_branch = make_branch(model_id, self.d_model, self.max_seq_len, depth=cfg['n_layer'])
        if config.get('no_grn'):
            def _strip_grn(container):
                for name, child in list(container.named_children()):
                    if isinstance(child, base.nn.Module) and type(child).__name__ == 'GRN1D':
                        setattr(container, name, nn.Identity())
                    else:
                        _strip_grn(child)
            _strip_grn(self.candidate_branch)
        self.candidate_branch.apply(_init_new_module)
        for module in self.candidate_branch.modules():
            if isinstance(module, GMLPBlock):
                module.reset_special()
        self.to(self.device)
        self.assert_budget()

    def assert_budget(self):
        n = downstream_parameters(self)
        if n > BUDGET:
            raise ValueError(f'{self.model_id}: downstream {n:,} exceeds {BUDGET:,}; check real Mamba version')
        return n

    def _validate_features(self, x):
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != self.embedding_dim:
            raise ValueError(f'Expected nonempty dense [B,L,{self.embedding_dim}] encoder features')
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f'L={x.shape[1]} exceeds locked max_seq_len={self.max_seq_len}')

    def _streams(self, x, disable_coupling=False):
        main = x
        for block in self.mamba_blocks:
            out = block(main)
            main = out[0] if isinstance(out, tuple) else out
        side = self.candidate_branch(x)
        return main, side

    def _pooled(self, main, side):
        out = self.fusion(main, side)
        return token_mean(out)

    def forward_features(self, features):
        """Production tensor entry point, before embedding_projection, all-valid.

        Variable-length PADDED sequences are NOT supported by this entry point.
        Do not mask only the last pool: every mixing branch needs the true policy.
        """
        self._validate_features(features)
        x = self.embedding_projection(features)
        main, side = self._streams(x)
        pooled = self.output_norm(self._pooled(main, side))
        return self.classifier(pooled), pooled, self.contrastive_projector(pooled), pooled.new_zeros(())

    def forward_sequence(self, sequences, variant_type=None, training_mode=True):
        if isinstance(self.token_encoder, base.EncoderStub):
            raise RuntimeError('Attach the REAL project encoder before forward_sequence; the collaboration stub is not training data')
        if training_mode and variant_type == 'original':
            sequences = maybe_augment(sequences)
        f = self.token_encoder(sequences, training_mode=training_mode)
        self._validate_features(f)
        main, side = self._streams(self.embedding_projection(f))
        h_nt = self.output_norm(self._pooled(main, side))
        segs = SegmentProteinCache.get().lookup(sequences, h_nt.device, h_nt.dtype)
        tokens = self.seg_self_attn(self.seg_proj(segs))
        h_pr = self.pr_norm(self.pr_attn(h_nt[:, None], tokens)[:, 0])
        pooled = self.space_fuse(torch.cat([h_nt, h_pr], dim=-1))
        return self.classifier(pooled), pooled, self.contrastive_projector(pooled), pooled.new_zeros(())

    def forward(self, *args, **kwargs):
        # Preserve the inherited six-tuple loss/view routing even after .to(...).
        self.device = next(self.parameters()).device
        out = super().forward(*args, **kwargs)
        return out

    @torch.no_grad()
    def diagnose_features(self, features):
        if any(m.training for m in self.modules()):
            raise ValueError('Call eval() before contribution diagnostics')
        self._validate_features(features)
        x = self.embedding_projection(features)
        main, side = self._streams(x)
        pooled = self._pooled(main, side)
        def predict(a, b):
            return self.classifier(self.output_norm(self._pooled(a, b)))
        full = predict(main, side)
        without_side = predict(main, torch.zeros_like(side))
        report = {
            'model': self.model_id,
            'main_rms': float(main.float().square().mean().sqrt()),
            'side_rms': float(side.float().square().mean().sqrt()),
            'full_logits': full.detach(),
            'zero_side_logits': without_side.detach(),
            'zero_side_logit_rms_delta': float((full-without_side).float().square().mean().sqrt()),
            'note': 'Intervention on fixed features; zeroed activations may be out of distribution. Not a trained ablation or a causal importance estimate.',
        }
        if self.model_id != 'T09':
            delta = self.fusion(main, side)-main
            report['scaled_fusion_rms_over_main'] = float(delta.float().square().mean().sqrt() /
                                                        main.float().square().mean().sqrt().clamp_min(1e-12))
            report['pooled_delta_rms_over_main_mean'] = float((pooled-token_mean(main)).float().square().mean().sqrt() /
                                                            token_mean(main).float().square().mean().sqrt().clamp_min(1e-12))
        if self.model_id == 'T10':
            raw_main, raw_side = self._streams(x, disable_coupling=True)
            report['no_coupling_logits'] = predict(raw_main, raw_side).detach()
            report['note'] += ' Zero-side at final fusion leaves earlier coupling effects intact; no_coupling is a separate intervention.'
        return report

    def diagnose_readout(self, sequences):
        if isinstance(self.token_encoder, base.EncoderStub):
            raise RuntimeError('Attach a real encoder or call diagnose_features on saved real features')
        return self.diagnose_features(self.token_encoder(sequences, training_mode=False))

    def load_matching_base_state_dict(self, state_dict):
        raise ValueError('The released model has a different readout; do not warm-start from an unrelated checkpoint')

    def load_matching_basic_state_dict(self, state_dict):
        raise ValueError('The released model starts from a fresh initialisation; warm-starting requires an explicit matched protocol')

    def get_model_summary(self):
        info = super().get_model_summary()
        info.update(model_id=self.model_id,
                    candidate_parameters=count_parameters(self.candidate_branch),
                    secondary_stack_parameters=count_parameters(self.candidate_branch),
                    secondary_stack_type=self.model_id,
                    fusion_parameters=count_parameters(self.fusion),
                    readout_parameters=count_parameters(self.fusion),
                    downstream_budget=BUDGET,
                    model_architecture=f'DuetScreen: NTv3 nucleotide dual-stream + ESM2-35M 16-segment protein channel ({self.model_id})',
                    source=asdict(MODELS_BY_ID[self.model_id]) if self.model_id in MODELS_BY_ID else None)
        return info


class MatchedSingleStack(base.DualStackNetwork):
    """Active-parameter capacity control: deeper Mamba + pointwise residual FFN.

    FFN width fills the residual budget within <= (2*d+1)/2 parameters.
    It mixes channels, not sequences. This is NOT an exact FLOPs match.
    """
    def __init__(self, target: int, reference: str, **runtime):
        cfg = dict(base.BASE_NETWORK_CONFIG, block_type='basic')
        cfg.update(runtime)
        cfg['compile_model'] = False
        super().__init__(**cfg)
        self.reference = reference
        self.target_parameters = int(target)
        block_count = count_parameters(self.mamba_blocks[0])
        common_count = downstream_parameters(self)-len(self.mamba_blocks)*block_count
        self.plan = matched_plan(self.target_parameters, block_count, common_count, self.d_model)
        for _ in range(self.plan['n_layer']-len(self.mamba_blocks)):
            block = base.BasicMambaBlock(self.d_model)
            block.apply(_init_new_module)
            self.mamba_blocks.append(block)
        self.n_layer = self.plan['n_layer']
        self.capacity_norm = nn.LayerNorm(self.d_model)
        self.capacity_ffn = FeedForward(self.d_model, hidden=self.plan['ffn_hidden'])
        self.capacity_norm.apply(_init_new_module)
        self.capacity_ffn.apply(_init_new_module)
        self.to(self.device)
        self.model_id = f'MC_{reference}'
        actual = downstream_parameters(self)
        if actual > BUDGET or abs(actual-self.target_parameters) > self.d_model+1:
            raise ValueError(f'Capacity match failed: {actual} vs {self.target_parameters}')

    def forward_features(self, features):
        if features.ndim != 3 or not 0 < features.shape[1] <= self.max_seq_len or features.shape[-1] != self.embedding_dim:
            raise ValueError('Expected dense, all-valid [B,L,embedding_dim] features')
        x = self.embedding_projection(features)
        for block in self.mamba_blocks:
            x = block(x)
        x = x + self.capacity_ffn(self.capacity_norm(x))
        pooled = self.output_norm(token_mean(x))
        return self.classifier(pooled), pooled, self.contrastive_projector(pooled), pooled.new_zeros(())

    def forward_sequence(self, sequences, variant_type=None, training_mode=True):
        if isinstance(self.token_encoder, base.EncoderStub):
            raise RuntimeError('Attach the real encoder; EncoderStub is not training data')
        return self.forward_features(self.token_encoder(sequences, training_mode=training_mode))

    def forward(self, *args, **kwargs):
        self.device = next(self.parameters()).device
        return super().forward(*args, **kwargs)

    def get_model_summary(self):
        out = super().get_model_summary()
        out.update(model_id=self.model_id, reference=self.reference,
                   target_parameters=self.target_parameters, capacity_plan=dict(self.plan),
                   parameter_difference=downstream_parameters(self)-self.target_parameters,
                   model_architecture='Single Mamba stack + residual channel FFN capacity control')
        return out


def matched_plan(target: int, block_count: int, common_count: int, d: int = 256):
    """Choose maximum feasible Mamba depth <=8, fill rest with an active FFN.

    FFN including LN has (2*d+1)*hidden+3*d parameters (both Linear biases).
    Depth is a explicitly simple fixed rule, NOT optimized on any result.
    """
    if not 0 < target <= BUDGET or block_count <= 0:
        raise ValueError('Invalid capacity target or block count')
    for depth in range(8, 1, -1):
        remaining = target-common_count-depth*block_count-3*d
        hidden = int(math.floor(remaining/(2*d+1)+.5))
        if hidden >= 1:
            actual = common_count+depth*block_count+(2*d+1)*hidden+3*d
            if actual > BUDGET:
                hidden -= 1
                actual -= 2*d+1
            if hidden >= 1 and abs(actual-target) <= d+1:
                return dict(n_layer=depth, ffn_hidden=hidden, target=target,
                            parameters=actual, difference=actual-target)
    raise ValueError('Target too small for the two-block single stack and FFN')


_RUNTIME_KEYS = {'device', 'enable_tf32', 'transformer_model_repo', 'freeze_transformer',
                 'trust_remote_code', 'use_caching', 'n_layer', 'main_kernel', 'no_grn'}


def _validate_runtime(kwargs):
    unknown = set(kwargs)-_RUNTIME_KEYS
    if unknown:
        raise ValueError(f'Locked profile: unsupported overrides {sorted(unknown)}')
    if kwargs.get('use_caching', False):
        raise ValueError('Keep encoder caching disabled for trainable encoder segments')


def create_v6_network(model_id: str, *, encoder=None, **runtime):
    """Build the released network. Encoder ownership stays with the caller.

    The downstream budget (everything except the encoder, output heads
    included) is checked before returning.
    """
    _validate_runtime(runtime)
    if model_id != 'DuetScreen':
        raise ValueError(f'Unknown model {model_id}')
    model = DuetScreenNetwork(model_id, **runtime)
    if encoder is not None:
        model.token_encoder = encoder.to(model.device)
    count = downstream_parameters(model)
    if count > BUDGET:
        raise ValueError(f'Downstream budget exceeded: {count:,} > {BUDGET:,}')
    return model

def optimizer_diagnostics(model: nn.Module, optimizer=None) -> dict:
    """Call after backward / scaler.unscale_, before optimizer.step/zero_grad.

    Covers EVERY trainable downstream parameter, including new side branches.
    It reports existing groups, never changes LR or weight decay policy.
    """
    encoder_ids = {id(p) for p in model.token_encoder.parameters()}
    membership = {}
    if optimizer is not None:
        for idx, group in enumerate(optimizer.param_groups):
            for p in group['params']:
                membership.setdefault(id(p), []).append(idx)
    groups = {}
    for name, p in model.named_parameters():
        if id(p) in encoder_ids:
            continue
        key = name.split('.')[0]
        g = groups.setdefault(key, dict(parameters=0, grad_none=[], grad_nonfinite=[],
                                       grad_zero=[], missing_optimizer=[], duplicate_optimizer=[],
                                       grad_squared=0., weight_squared=0., optimizer_indices=set()))
        g['parameters'] += p.numel()
        g['weight_squared'] += float(p.detach().float().square().sum())
        if p.requires_grad:
            if p.grad is None:
                g['grad_none'].append(name)
            else:
                v = p.grad.detach().float()
                if not torch.isfinite(v).all():
                    g['grad_nonfinite'].append(name)
                if not torch.count_nonzero(v):
                    g['grad_zero'].append(name)
                g['grad_squared'] += float(v.square().sum())
            if optimizer is not None:
                indices = membership.get(id(p), [])
                if not indices:
                    g['missing_optimizer'].append(name)
                if len(indices) > 1:
                    g['duplicate_optimizer'].append(name)
                g['optimizer_indices'].update(indices)
    for g in groups.values():
        g['grad_norm'] = math.sqrt(g.pop('grad_squared'))
        g['weight_norm'] = math.sqrt(g.pop('weight_squared'))
        indices = sorted(g.pop('optimizer_indices'))
        g['optimizer_groups'] = [dict(index=i, lr=float(optimizer.param_groups[i]['lr']),
                                    weight_decay=float(optimizer.param_groups[i].get('weight_decay', 0.)))
                                 for i in indices]
    return groups
