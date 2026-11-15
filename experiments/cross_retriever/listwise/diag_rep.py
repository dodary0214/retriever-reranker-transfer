"""Same-input rerun diagnostic: for the same 30 input lists, how different are the rerun
rankings from the stored ones? nDCG@10 change, and the share of relevant–nonrelevant / relevant–relevant document pairs
whose order differs between the two runs of the SAME input (noise floor for the replacement flip)."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import EXP, LISTWISE_E, LISTWISE_F

P = LISTWISE_F
rows = []
for m in sys.argv[1:] or ['zephyr', 'first']:
    if not os.path.exists(f'{P}/rep/out/{m}.jsonl'):
        continue
    old = {json.loads(l)['job_id']: json.loads(l)['ranking'] for l in open(f'{LISTWISE_E}/out/{m}.jsonl')}
    for l in open(f'{P}/rep/out/{m}.jsonl'):
        x = json.loads(l)
        _, d, r, q = x['job_id'].split('|')
        qr = pd.read_csv(f'{EXP}/common/qrels/{d}.tsv', sep='\t', names=['q', 'doc', 'g'], dtype={'q': str, 'doc': str})
        g = dict(zip(qr[qr.q == q].doc, qr[qr.q == q].g))
        a, b = old[x['job_id']], x['ranking']

        def nd(rk):
            dcg = sum(g.get(t, 0) / math.log2(i + 2) for i, t in enumerate(rk[:10]))
            ideal = sorted(g.values(), reverse=True)[:10]
            idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
            return dcg / idcg if idcg else 0.0
        pa, pb = {t: i for i, t in enumerate(a)}, {t: i for i, t in enumerate(b)}
        rel = [t for t in a if g.get(t, 0) >= 1]
        non = [t for t in a if g.get(t, 0) < 1]
        f_rn = sum((pa[u] < pa[v]) != (pb[u] < pb[v]) for u in rel for v in non)
        f_rr = sum((pa[u] < pa[v]) != (pb[u] < pb[v]) for i, u in enumerate(rel) for v in rel[i + 1:])
        rows.append(dict(model=m, job=x['job_id'], identical=a == b, ndcg_old=nd(a), ndcg_new=nd(b),
                         rn_pairs=len(rel) * len(non), rn_flips=f_rn, rr_pairs=len(rel) * (len(rel) - 1) // 2, rr_flips=f_rr,
                         all_pairs=len(a) * (len(a) - 1) // 2,
                         all_flips=sum((pa[u] < pa[v]) != (pb[u] < pb[v]) for i, u in enumerate(a) for v in a[i + 1:])))
t = pd.DataFrame(rows)
t.to_csv(f'{P}/rep/diag_rep.csv', index=False)
for m, s in t.groupby('model'):
    print(f'{m}: identical {s.identical.sum()}/{len(s)}; |ΔnDCG@10| mean {np.abs(s.ndcg_new - s.ndcg_old).mean():.4f} '
          f'max {np.abs(s.ndcg_new - s.ndcg_old).max():.4f}; mean nDCG old {s.ndcg_old.mean():.4f} new {s.ndcg_new.mean():.4f}; '
          f'flip rel–nonrel {s.rn_flips.sum() / s.rn_pairs.sum():.4f} ({s.rn_pairs.sum()} pairs), '
          f'rel–rel {s.rr_flips.sum() / s.rr_pairs.sum():.4f} ({s.rr_pairs.sum()}), all pairs {s.all_flips.sum() / s.all_pairs.sum():.4f}')
