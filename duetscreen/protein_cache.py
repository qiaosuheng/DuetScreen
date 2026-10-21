"""Build the 16-segment ESM2-35M protein cache used by DuetScreen.

For every DNA sequence, the best stop-free ORF interval (>= 30 nt, six frames,
longest protein, capped at 1,000 aa) is translated and embedded at residue
level with ESM2-35M; the residues are mean-pooled into 16 contiguous equal
segments, giving one fp16 [16, 480] row per DNA sequence. Sequences without a
usable ORF are omitted and the model's zero-vector fallback applies.

Input may be a FASTA file or a pickle holding either a list of record dicts
(with `original_seq`) or a dict whose keys are DNA sequences. Output is a
pickle {dna_seq: fp16[16,480]} plus a <output>.stats.json file.

Example
-------
python -m duetscreen.protein_cache \
    --input records.fasta --output protein_seg_cache_35m.pkl \
    --esm2 /assets/esm2_t12_35M_UR50D --device cuda --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

SEG = 16
MAX_PROTEIN = 1000
MIN_INTERVAL = 30
_RC = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
STOPS = {'TAA', 'TAG', 'TGA'}


def rc(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def best_interval_frame(seq: str, frame: int):
    best_start = best_len = 0
    cur = frame
    i = frame
    n = len(seq)
    while i + 3 <= n:
        if seq[i:i + 3] in STOPS:
            if i - cur > best_len:
                best_start, best_len = cur, i - cur
            cur = i + 3
        i += 3
    tail = n - ((n - cur) % 3) - cur
    if tail > best_len:
        best_start, best_len = cur, tail
    return best_start, best_len


def best_protein(dna: str) -> str:
    dna = dna.upper()
    from Bio.Seq import Seq
    best = ''
    for s in (dna, rc(dna)):
        for frame in range(3):
            st, ln = best_interval_frame(s, frame)
            if ln >= MIN_INTERVAL and ln > 3 * len(best):
                p = str(Seq(s[st:st + ln]).translate())
                if len(p) > len(best):
                    best = p
    return best[:MAX_PROTEIN]


def segment_pool(rep_row: torch.Tensor, length: int) -> np.ndarray:
    bounds = np.linspace(0, length, SEG + 1).astype(int)
    out = torch.zeros(SEG, rep_row.shape[1])
    for k in range(SEG):
        a, b = bounds[k], bounds[k + 1]
        if b > a:
            out[k] = rep_row[a:b].mean(dim=0)
    return out.float().cpu().numpy().astype(np.float16)


def read_sequences(path: Path):
    if path.suffix.lower() in ('.fa', '.fasta', '.fna'):
        seqs = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                seqs.append('')
            elif seqs:
                seqs[-1] += line
        return seqs
    with path.open('rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        return list(payload.keys())
    if isinstance(payload, list):
        out = []
        for rec in payload:
            if isinstance(rec, dict):
                out.append(rec.get('original_seq') or rec.get('sequence') or '')
            elif isinstance(rec, str):
                out.append(rec)
        return [s for s in out if s]
    raise ValueError(f'unsupported input payload: {type(payload)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the DuetScreen 16-segment ESM2-35M cache.')
    parser.add_argument('--input', required=True, help='FASTA or pickle with DNA sequences')
    parser.add_argument('--output', required=True, help='output pickle path')
    parser.add_argument('--esm2', required=True,
                        help='local folder or HF id of esm2_t12_35M_UR50D')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    seqs = [s.upper() for s in read_sequences(Path(args.input)) if s]
    seqs = sorted(set(seqs))
    print(f'sequences: {len(seqs)}', flush=True)

    proteins = {}
    for i, s in enumerate(seqs):
        proteins[s] = best_protein(s)
        if (i + 1) % 20000 == 0:
            print(f'  ORF {i + 1}/{len(seqs)} ({time.time() - t0:.0f}s)', flush=True)
    usable = [(s, p) for s, p in proteins.items() if p]
    print(f'ORF done, usable={len(usable)}, empty={len(seqs) - len(usable)}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.esm2)
    model = AutoModel.from_pretrained(args.esm2).eval().to(args.device)

    usable.sort(key=lambda x: -len(x[1]))
    emb = {}
    for lo in range(0, len(usable), args.batch_size):
        chunk = usable[lo:lo + args.batch_size]
        enc = tokenizer([p for _, p in chunk], return_tensors='pt', padding=True)
        enc = {k: v.to(args.device) for k, v in enc.items()}
        with torch.no_grad():
            states = model(**enc, output_hidden_states=True).hidden_states
        rep = states[12]
        for j, (s, p) in enumerate(chunk):
            length = len(p)
            emb[s] = segment_pool(rep[j, 1:1 + length], length)
        if (lo // args.batch_size) % 50 == 0:
            print(f'  embed {lo}/{len(usable)} ({time.time() - t0:.0f}s)', flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump(emb, f, protocol=4)
    stats = dict(model=str(args.esm2), repr_layer=12, dim=480, segments=SEG,
                 n_sequences=len(seqs), n_embedded=len(emb),
                 pkl_mb=round(out.stat().st_size / 1e6, 1),
                 seconds=round(time.time() - t0, 1))
    out.with_suffix(out.suffix + '.stats.json').write_text(json.dumps(stats, indent=1))
    print('DONE ' + json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
