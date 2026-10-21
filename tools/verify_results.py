"""Verify DuetScreen outputs against the recorded reference values.

Two modes:
  --predictions <csv> --expected <json>   compare per-record scores
  --smoke <json>                          check the training smoke report

Exit code 0 = PASS.
"""
import argparse
import csv
import json
import sys


def verify_predictions(path, expected_path):
    ref = json.load(open(expected_path))
    tol = float(ref.get('tolerance', 2e-3))
    want = ref['scores']
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print('FAIL: prediction CSV is empty')
        return 1
    got = {}
    col = 'prediction_probability' if 'prediction_probability' in rows[0] else None
    if col is None:
        print('FAIL: prediction CSV has no prediction_probability column:', list(rows[0]))
        return 1
    for r in rows:
        key = r.get('sequence_id', '').split('|')[0]
        got[key] = float(r[col])
    missing = [k for k in want if k not in got]
    if missing:
        print('FAIL: missing records:', missing)
        return 1
    worst, worst_id = 0.0, None
    for key, target in want.items():
        diff = abs(got[key] - float(target))
        marker = 'ok ' if diff <= tol else 'DIFF'
        print(f'{marker} {key:32s} expected {float(target):.6f}  got {got[key]:.6f}  |d|={diff:.2e}')
        if diff > worst:
            worst, worst_id = diff, key
    if worst <= tol:
        print(f'PASS: {len(want)} records match within {tol:g} (max |d| {worst:.2e})')
        return 0
    print(f'FAIL: {worst_id} differs by {worst:.2e} (tolerance {tol:g})')
    return 1


def verify_smoke(path):
    rep = json.load(open(path))
    checks = {
        'smoke == passed': rep.get('smoke') == 'passed',
        'downstream parameters == 2987761': rep.get('downstream_parameters') == 2987761,
        'encoder trainable == 2763264': rep.get('encoder_trainable') == 2763264,
        'finite loss': isinstance(rep.get('loss'), float) and rep['loss'] == rep['loss'],
    }
    for name, ok in checks.items():
        print(('ok   ' if ok else 'FAIL ') + name)
    if all(checks.values()):
        print('PASS: training smoke report consistent with the released configuration')
        return 0
    print('FAIL: training smoke report inconsistent')
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions')
    ap.add_argument('--expected')
    ap.add_argument('--smoke')
    args = ap.parse_args(argv)
    if args.smoke:
        return verify_smoke(args.smoke)
    if not (args.predictions and args.expected):
        ap.error('need --predictions and --expected, or --smoke')
    return verify_predictions(args.predictions, args.expected)


if __name__ == '__main__':
    sys.exit(main())
