"""Report how many training records are covered by a protein-segment cache.

Usage: python tools/check_cache_coverage.py --records variants_data.pkl \
                                            --cache protein_seg_cache_35m.pkl
Prints the covered fraction and exits 0 if >= --min-coverage (default 0.99).
"""
import argparse
import pickle
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--min-coverage', type=float, default=0.99)
    args = ap.parse_args(argv)

    with open(args.records, 'rb') as f:
        records = pickle.load(f)
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)
    missing = sum(1 for r in records if (r.get('original_seq') or '').upper() not in cache)
    covered = len(records) - missing
    frac = covered / max(1, len(records))
    print(f'records={len(records)} covered={covered} missing={missing} fraction={frac:.4f}')
    return 0 if frac >= args.min_coverage else 1


if __name__ == '__main__':
    sys.exit(main())
