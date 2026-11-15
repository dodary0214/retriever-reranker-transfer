"""Additional robustness rows (definitions fixed before computing). CPU only, existing outputs only.
Reuses analyze.py's data loading and bootstrap by executing its source up to the Fig. 5 block (that part writes no files).
Writes $RRT_OUT/derived/appendixB_extra_{flips,kappa}.csv."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from common import OUT as OUT_ROOT, QWEN, REPO  # noqa: E402
AN = os.path.join(REPO, 'analysis', 'analyze.py')
OUT = os.path.join(OUT_ROOT, 'derived')
os.makedirs(OUT, exist_ok=True)
src = open(AN).read()
cut = src.index('# ================================================================ figure 5')
g = {'__file__': AN, '__name__': 'analyze_prefix'}
exec(compile(src[:cut], AN, 'exec'), g)
EVAL, PAIRS, REL, QRELS, DLSET = g['EVAL'], g['PAIRS'], g['REL'], g['QRELS'], g['DLSET']
keyf, top, ratio, PL, NAME, SETS = g['keyf'], g['top'], g['ratio'], g['PL'], g['NAME'], g['SETS']
codes, kappa_w, BW, ci, LW = g['codes'], g['kappa_w'], g['BW'], g['ci'], g['LW']
print('evaluation queries', len(EVAL))

# ---- B2: replacement flips of relevant--non-relevant pairs; non-relevant = judged grade 0 (and, for reference, any)
G0 = {d: q[q.relevance == 0].groupby('query_id').doc_id.apply(set).to_dict() for d, q in QRELS.items()}
rows = []
n = len(EVAL)
for rr in LW:
    acc = {(v, p): (np.zeros(n), np.zeros(n)) for v in ('judged0', 'all') for p in PAIRS}
    for i, (d, q) in enumerate(EVAL):
        rel, z0 = REL[d].get(q, set()), G0[d].get(q, set())
        for p in PAIRS:
            ka, kb = keyf(rr, d, p[0], q), keyf(rr, d, p[1], q)
            ok = [x for x in set(top((d, q), p[0])) & set(top((d, q), p[1])) if x in ka and x in kb]
            R = [x for x in ok if x in rel]
            if not R:
                continue
            va, vb = np.array([ka[x] for x in R]), np.array([kb[x] for x in R])
            for v, N in (('all', [x for x in ok if x not in rel]), ('judged0', [x for x in ok if x in z0])):
                if not N:
                    continue
                A2, B2 = np.array([ka[x] for x in N]), np.array([kb[x] for x in N])
                sa, sb = np.sign(va[:, None] - A2[None, :]), np.sign(vb[:, None] - B2[None, :])
                valid = (sa != 0) & (sb != 0)
                acc[(v, p)][0][i] += ((sa != sb) & valid).sum()
                acc[(v, p)][1][i] += valid.sum()
    for v in ('judged0', 'all'):
        for p in PAIRS + ['pooled']:
            f, c = (sum(acc[(v, x)][0] for x in PAIRS), sum(acc[(v, x)][1] for x in PAIRS)) if p == 'pooled' else acc[(v, p)]
            r, lo, hi, _ = ratio(f, c, EVAL)
            rows.append(dict(reranker=NAME[rr], nonrel=v, retriever_pair=p if p == 'pooled' else PL(p),
                             n_pairs=int(c.sum()), flips=int(f.sum()), rate=r, lo=lo, hi=hi))
fl = pd.DataFrame(rows)
fl.to_csv(f'{OUT}/appendixB_extra_flips.csv', index=False)
print(fl.to_string())

# ---- B1 extra row: relevant set = grade >= 2 on DL19/DL20 (grade >= 1 on BEIR), kappa on relevant-set-identical queries
REL2 = {d: (q[q.relevance >= 2] if d in DLSET else q[q.relevance >= 1]).groupby('query_id').doc_id.apply(set).to_dict()
        for d, q in QRELS.items()}
krows = []
for defn, RS in (('grade>=1 (main)', REL), ('DL grade>=2', REL2)):
    for p in PAIRS:
        mask = np.array([(set(top(k, p[0])) & RS[k[0]].get(k[1], set())) == (set(top(k, p[1])) & RS[k[0]].get(k[1], set()))
                         for k in EVAL])
        kb = {}
        for s in ('PW4', 'MIX4'):
            rr = SETS[s]
            ca, cb = codes('nDCG@10', EVAL, rr, p[0], 0.01), codes('nDCG@10', EVAL, rr, p[1], 0.01)
            k1, nn, _, _ = kappa_w(ca, cb, mask, len(rr), np.ones((1, n), np.float32))
            kb[s] = kappa_w(ca, cb, mask, len(rr), BW(EVAL))[0]
            lo, hi = ci(kb[s])
            krows.append(dict(definition=defn, pair=PL(p), set=s, n_queries=int(mask.sum()),
                              n_dl=int(sum(m for m, k in zip(mask, EVAL) if k[0] in DLSET)), n_notie=int(nn[0]),
                              kappa=float(k1[0]), lo=lo, hi=hi))
        # paired difference on the same resamples (as analyze.py kappa_table)
        lo, hi = ci(kb['PW4'] - kb['MIX4'])
        krows.append(dict(definition=defn, pair=PL(p), set='PW4−MIX4', n_queries=int(mask.sum()),
                          kappa=krows[-2]['kappa'] - krows[-1]['kappa'], lo=lo, hi=hi))
kp = pd.DataFrame(krows)
kp.to_csv(f'{OUT}/appendixB_extra_kappa.csv', index=False)
print(kp.to_string())
