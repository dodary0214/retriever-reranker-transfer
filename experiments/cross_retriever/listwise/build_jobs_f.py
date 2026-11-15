"""Job lists in $RRT_DATA/listwise_f (CPU).
usage: build_jobs_f.py rep        -> rep/jobs.jsonl: same-input rerun of 10 E queries x 3 retrievers (phase 'rep')
       build_jobs_f.py rep50      -> rep50/jobs.jsonl: 40 more E queries (50 in total, by dataset share)
       build_jobs_f.py F <t_sec>  -> n = min(1206, floor(22 h / t)) queries; queries_F.csv, F/jobs.jsonl (phase 'F')
       build_jobs_f.py Fn <n>     -> n queries (the reported set F: n = 742); queries_F.csv, F/jobs.jsonl
F is drawn from the 1,206 queries outside E by dataset share (largest remainder) and a seed-0 permutation per
dataset, so a smaller n gives a subset of a larger one."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import EXP, LISTWISE_E, LISTWISE_F, POINTWISE, RETRIEVERS, read_run

P, P2 = LISTWISE_F, POINTWISE
g = pd.read_parquet(f'{P2}/grid.parquet')
DS = list(dict.fromkeys(g.dataset))
RUNS = {(d, r): read_run(f'{EXP}/common/runs/{d}.{r}.top100.txt') for d in DS for r in RETRIEVERS}
qm = pd.read_csv(f'{LISTWISE_E}/queries_mix.csv', dtype={'qid': str})


def jobs(keys, phase):
    return [dict(job_id=f'grid|{d}|{r}|{q}', phase=phase, dataset=d, qid=q, docids=[x for x, _ in RUNS[(d, r)][q]])
            for d, q in keys for r in RETRIEVERS]


def write(path, js):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for j in js:
            f.write(json.dumps(j) + '\n')


if sys.argv[1] == 'rep50':  # 50 E queries by dataset share, includes the 10 rep queries; writes the 40 new
    rep = set(map(tuple, pd.read_csv(f'{P}/rep/queries_rep.csv', dtype={'qid': str}).values))
    byds = {d: list(qm[qm.dataset == d].qid) for d in dict.fromkeys(qm.dataset)}
    quota = {d: 50 * len(v) / len(qm) for d, v in byds.items()}
    k = {d: math.floor(x) for d, x in quota.items()}
    for d in sorted(quota, key=lambda d: quota[d] - k[d], reverse=True)[:50 - sum(k.values())]:
        k[d] += 1
    keys = []
    for d, v in byds.items():
        have = [q for q in v if (d, q) in rep]
        assert len(have) <= k[d]
        rest = [v[i] for i in np.random.default_rng(0).permutation(len(v)) if (d, v[i]) not in rep]
        keys += [(d, q) for q in rest[:k[d] - len(have)]]
    write(f'{P}/rep50/jobs.jsonl', jobs(keys, 'rep'))
    pd.DataFrame(keys, columns=['dataset', 'qid']).to_csv(f'{P}/rep50/queries_rep50_new.csv', index=False)
    print('quota', k, 'new', len(keys))
elif sys.argv[1] == 'rep':
    idx = np.random.default_rng(0).choice(len(qm), 10, replace=False)
    keys = [tuple(qm.iloc[i][['dataset', 'qid']]) for i in idx]
    write(f'{P}/rep/jobs.jsonl', jobs(keys, 'rep'))
    pd.DataFrame(keys, columns=['dataset', 'qid']).to_csv(f'{P}/rep/queries_rep.csv', index=False)
    print(keys)
else:
    t = float(sys.argv[2])  # 'F <t>' (n from the time per query) or 'Fn <n>' (n given)
    E = set(map(tuple, qm[['dataset', 'qid']].values))
    allq = sorted({(d, q) for d, q in g[['dataset', 'qid']].drop_duplicates().values
                   if all(RUNS[(d, r)].get(q) for r in RETRIEVERS)})
    assert len(allq) == 1753
    rest = [k for k in allq if k not in E]
    assert len(rest) == 1206
    n = min(1206, math.floor(22 * 3600 / t)) if sys.argv[1] == 'F' else int(sys.argv[2])
    print(f'arg {t} -> n = {n}')
    if n < 300:
        print('n < 300: do not run (pre-specified minimum)')
        sys.exit(2)
    byds = {d: sorted(q for dd, q in rest if dd == d) for d in DS}
    byds = {d: v for d, v in byds.items() if v}
    if n < 1206:  # proportional, largest remainder
        quota = {d: n * len(v) / 1206 for d, v in byds.items()}
        k = {d: math.floor(x) for d, x in quota.items()}
        for d in sorted(quota, key=lambda d: quota[d] - k[d], reverse=True)[:n - sum(k.values())]:
            k[d] += 1
    else:
        k = {d: len(v) for d, v in byds.items()}
    keys = []
    for d, v in byds.items():
        perm = np.random.default_rng(0).permutation(len(v))
        keys += [(d, v[i]) for i in perm[:k[d]]]
    pd.DataFrame(keys, columns=['dataset', 'qid']).to_csv(f'{P}/queries_F.csv', index=False)
    write(f'{P}/F/jobs.jsonl', jobs(keys, 'F'))
    print({d: k[d] for d in k}, 'total', len(keys))
