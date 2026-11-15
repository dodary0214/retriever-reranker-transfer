"""Compare analysis outputs with the reference values in results/summary.

usage: compare_outputs.py [OUTPUT_ROOT] [--atol 1e-9]
OUTPUT_ROOT defaults to $RRT_OUT or outputs/. Every CSV under results/summary/<subdir>/ is compared with
OUTPUT_ROOT/<subdir>/<file>: same columns and rows, numeric columns within --atol, other columns equal.
Exit status 1 if any file is missing or differs.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, 'results', 'summary')


def compare(ref_path, out_path, atol):
    a = pd.read_csv(ref_path, keep_default_na=False, na_values=[''])
    b = pd.read_csv(out_path, keep_default_na=False, na_values=[''])
    if list(a.columns) != list(b.columns):
        return 'columns differ'
    if len(a) != len(b):
        return f'rows differ ({len(a)} vs {len(b)})'
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            same_nan = np.isnan(xv) == np.isnan(yv)
            ok = same_nan & (np.isnan(xv) | (np.abs(xv - yv) <= atol))
            if not ok.all():
                i = int(np.argmin(ok))
                return f'column {c}, row {i}: {float(xv[i])!r} vs {float(yv[i])!r}'
        elif not x.astype(str).equals(y.astype(str)):
            i = int(np.argmax(x.astype(str).to_numpy() != y.astype(str).to_numpy()))
            return f'column {c}, row {i}: {x.iloc[i]!r} vs {y.iloc[i]!r}'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default=os.environ.get('RRT_OUT', os.path.join(REPO, 'outputs')))
    ap.add_argument('--atol', type=float, default=1e-9)
    args = ap.parse_args()
    n, bad = 0, []
    for sub in sorted(os.listdir(REF)):
        if not os.path.isdir(os.path.join(REF, sub)):
            continue
        for f in sorted(os.listdir(os.path.join(REF, sub))):
            if not f.endswith('.csv'):
                continue
            n += 1
            out_path = os.path.join(args.out, sub, f)
            msg = 'missing' if not os.path.exists(out_path) else compare(os.path.join(REF, sub, f), out_path, args.atol)
            if msg:
                bad.append((f'{sub}/{f}', msg))
    for f, msg in bad:
        print(f'DIFF {f}: {msg}')
    print(f'{n - len(bad)}/{n} files match (atol {args.atol})')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
