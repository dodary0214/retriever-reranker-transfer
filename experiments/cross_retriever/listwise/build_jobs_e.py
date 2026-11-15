"""Query set E (547 queries: all DL19/DL20 queries + seed-0 samples of FiQA 200, DBpedia 150, NFCorpus 100) and its
listwise job lists (CPU). Writes $RRT_DATA/listwise_e/queries_mix.csv (dataset, qid, half) and jobs.jsonl
(job_id, phase, dataset, qid, docids) for run_listwise.py.
Phases: gate (dl19 bm25), dl (rest of dl19/dl20), swap, beir1, beir2. The original job file also contained "swap"
lists (an exploratory input-order study, not used by the analysis); they are added only if swap_candidates.parquet is
present in the job directory."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import EXP, LISTWISE_E, POINTWISE, RETRIEVERS, SEED, read_run

P, P2 = LISTWISE_E, POINTWISE
os.makedirs(P, exist_ok=True)

SAMPLE = {'fiqa': 200, 'dbpedia': 150, 'nfcorpus': 100}

grid = pd.read_parquet(f'{P2}/grid.parquet')
runs = {(d, r): read_run(f'{EXP}/common/runs/{d}.{r}.top100.txt') for d in ['dl19', 'dl20', *SAMPLE] for r in RETRIEVERS}
qrows = []
for ds in ['dl19', 'dl20', *SAMPLE]:
    qids = sorted(q for q in grid[grid.dataset == ds].qid.unique() if all(runs[(ds, r)].get(q) for r in RETRIEVERS))
    if ds in SAMPLE:
        k = SAMPLE[ds]
        perm = [qids[i] for i in np.random.default_rng(SEED).permutation(len(qids))][:k]
        qrows += [(ds, q, 1 if i < k // 2 else 2) for i, q in enumerate(perm)]
    else:
        qrows += [(ds, q, 0) for q in qids]
Q = pd.DataFrame(qrows, columns=['dataset', 'qid', 'half'])
Q.to_csv(f'{P}/queries_mix.csv', index=False)

jobs = []
for ds, qid, half in qrows:
    for r in RETRIEVERS:
        phase = 'gate' if (ds, r) == ('dl19', 'bm25') else 'dl' if half == 0 else f'beir{half}'
        jobs.append(dict(job_id=f'grid|{ds}|{r}|{qid}', phase=phase, dataset=ds, qid=qid,
                         docids=[d for d, _ in runs[(ds, r)][qid]]))
h = lambda q, d: hashlib.sha256(f'{q}|{d}'.encode()).hexdigest()
cand = pd.read_parquet(f'{P}/swap_candidates.parquet') if os.path.exists(f'{P}/swap_candidates.parquet') else \
    pd.DataFrame(columns=['dataset', 'qid', 'cond', 'seed', 'docid'])
for (ds, qid, cond, seed), g in cand.groupby(['dataset', 'qid', 'cond', 'seed']):
    jobs.append(dict(job_id=f'swap|{ds}|{cond}|{seed}|{qid}', phase='swap', dataset=ds, qid=qid,
                     docids=sorted(g.docid, key=lambda d: h(qid, d))))
order = ['gate', 'dl', 'swap', 'beir1', 'beir2']
jobs.sort(key=lambda j: order.index(j['phase']))
with open(f'{P}/jobs.jsonl', 'w') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')

J = pd.DataFrame(jobs)
J['n'] = J.docids.str.len()
J['windows'] = J.n.apply(lambda n: 1 if n <= 20 else 1 + int(np.ceil((n - 20) / 10)))
print(Q.groupby(['dataset', 'half']).size().to_string())
print(J.groupby('phase').agg(lists=('job_id', 'size'), calls=('windows', 'sum'), mean_len=('n', 'mean')).round(1).to_string())
print('total calls per model', J.windows.sum())
