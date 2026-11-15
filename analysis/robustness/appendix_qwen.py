"""Qwen3-Reranker-4B (LLM-based pointwise) on the evaluation queries x 3 retrievers, as a robustness addition that
leaves the main sets unchanged: reranker table row, sign agreement of its pairs, and kappa of the pointwise set with
MiniLM replaced by Qwen3-Reranker-4B (overall and per dataset).
CPU only. Reuses analyze.py's data loading, metrics and bootstrap by executing its source up to the Fig. 5 block (that
part writes no files). Reads $RRT_DATA/qwen_reranker/scores/*.parquet. Writes $RRT_OUT/derived/qwen_*.csv."""
import glob
import itertools
import os
import sys

import ir_measures
import numpy as np
import pandas as pd
from ir_measures import nDCG, RR, Judged

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from common import OUT as OUT_ROOT, QWEN, REPO  # noqa: E402
AN = os.path.join(REPO, 'analysis', 'analyze.py')
OUT = os.path.join(OUT_ROOT, 'derived')
os.makedirs(OUT, exist_ok=True)
CS = f'{QWEN}/scores'
src = open(AN).read()
cut = src.index('# ================================================================ figure 5')
g = {'__file__': AN, '__name__': 'analyze_prefix'}
exec(compile(src[:cut], AN, 'exec'), g)
EVAL, PAIRS, REL, QRELS, JUDGED, RUNS = g['EVAL'], g['PAIRS'], g['REL'], g['QRELS'], g['JUDGED'], g['RUNS']
RETRIEVERS, DLSET, PL, NAME, SETS = g['RETRIEVERS'], g['DLSET'], g['PL'], g['NAME'], g['SETS']
codes, kappa_w, BW, ci, same_rel, sign_table = g['codes'], g['kappa_w'], g['BW'], g['ci'], g['same_rel'], g['sign_table']
M = g['M']
print('evaluation queries', len(EVAL))

# ---- Qwen3-Reranker-4B scores -> per-retriever runs -> the same four measures as analyze.py
S = pd.concat([pd.read_parquet(f).assign(dataset=os.path.basename(f).split('-', 1)[1][:-8])
               for f in sorted(glob.glob(f'{CS}/*.parquet'))])
assert not S.duplicated(['dataset', 'qid', 'docid']).any()
SC = {(d, q): dict(zip(x.docid, x.score)) for (d, q), x in S.groupby(['dataset', 'qid'])}
missing = [k for k in EVAL if k not in SC]
assert not missing, f'Qwen scores missing for {len(missing)} evaluation queries'
rows_m = []
for d in sorted({k[0] for k in EVAL}):
    qs = [q for dd, q in EVAL if dd == d]
    for r in RETRIEVERS:
        lines = []
        for q in qs:
            sc = SC[(d, q)]
            for doc, _ in RUNS[(d, r)][q]:
                lines.append((q, doc, float(sc[doc])))  # KeyError = a candidate without a score -> stop
        run = pd.DataFrame(lines, columns=['query_id', 'doc_id', 'score'])
        jd = JUDGED[d]
        runj = run[[doc in jd.get(q, ()) for q, doc in zip(run.query_id, run.doc_id)]]
        res = {}
        for m, rn in ((nDCG @ 10, run), (RR @ 10, run), (Judged @ 10, run), ('nDCG_judged', runj)):
            meas = nDCG @ 10 if m == 'nDCG_judged' else m
            res[str(m)] = {x.query_id: x.value for x in ir_measures.iter_calc([meas], QRELS[d], rn)}
        for q in qs:
            rows_m.append((d, q, r, 'qwen', *(res[k].get(q, 0.0) for k in ('nDCG@10', 'RR@10', 'Judged@10', 'nDCG_judged'))))
MQ = pd.DataFrame(rows_m, columns=M.columns)
M2 = pd.concat([M, MQ], ignore_index=True)
g['MW'] = {m: M2.pivot_table(index=['dataset', 'qid'], columns=['retriever', 'reranker'], values=m)
           for m in ('nDCG@10', 'RR@10', 'Judged@10', 'nDCG_judged')}
NAME['qwen'] = 'Qwen3-Reranker-4B'
SETS['PW4q'] = ['qwen', 'bge-base', 'bge-m3', 'monot5']  # MiniLM replaced
X = g['X']

# ---- Table 1(b) row
t1 = dict(reranker='Qwen3-Reranker-4B', n=len(EVAL))
for r in RETRIEVERS:
    t1[f'ndcg10_{r}'] = X('nDCG@10', EVAL, ['qwen'], r)[:, 0].mean()
    t1[f'judged10_{r}'] = X('Judged@10', EVAL, ['qwen'], r)[:, 0].mean()
t1 = pd.DataFrame([t1])
t1.to_csv(f'{OUT}/qwen_table1b.csv', index=False)
print(t1.to_string())

# ---- Appendix A: kappa with PW4' (MiniLM -> Qwen3-Reranker-4B), MIX4 and PW4 for reference; all and relevant-set-identical
n = len(EVAL)
one = np.ones((1, n), np.float32)
SR = same_rel(EVAL)


def kap_rows(keys_mask_fn, label, metric='nDCG@10', th=0.01):
    rows = []
    for sub in ('all', 'same_rel'):
        for p in PAIRS:
            mask = keys_mask_fn(sub, p)
            kb = {}
            for s in ('PW4q', 'PW4', 'MIX4'):
                rr = SETS[s]
                ca, cb = codes(metric, EVAL, rr, p[0], th), codes(metric, EVAL, rr, p[1], th)
                k, nn, _, _ = kappa_w(ca, cb, mask, len(rr), one)
                kb[s] = kappa_w(ca, cb, mask, len(rr), BW(EVAL))[0]
                lo, hi = ci(kb[s])
                rows.append(dict(scope=label, subset=sub, pair=PL(p), set=s, n_queries=int(mask.sum()), n_notie=int(nn[0]),
                                 kappa=float(k[0]), lo=lo, hi=hi, _b=kb[s]))
            for a, b in (('PW4q', 'MIX4'), ('PW4q', 'PW4')):
                lo, hi = ci(kb[a] - kb[b])
                ka = next(x['kappa'] for x in rows[::-1] if x['set'] == a)
                kbv = next(x['kappa'] for x in rows[::-1] if x['set'] == b)
                rows.append(dict(scope=label, subset=sub, pair=PL(p), set=f'{a}−{b}', n_queries=int(mask.sum()),
                                 kappa=ka - kbv, lo=lo, hi=hi))
    return rows


kr = kap_rows(lambda sub, p: np.ones(n, bool) if sub == 'all' else SR[p], 'evaluation')
# Appendix B (per collection and per dataset): relevant-set-identical queries within each scope
SCOPES = [('TREC DL', lambda d: d in DLSET), ('BEIR', lambda d: d not in DLSET)] + \
         [(d, (lambda dd: (lambda d: d == dd))(d)) for d in dict.fromkeys(k[0] for k in EVAL)]
for coll, sel in SCOPES:
    cm = np.array([sel(k[0]) for k in EVAL])
    kr += [x for x in kap_rows(lambda sub, p: (np.ones(n, bool) if sub == 'all' else SR[p]) & cm, coll)
           if x['subset'] == 'same_rel']
kp = pd.DataFrame(kr).drop(columns='_b')
kp.to_csv(f'{OUT}/qwen_kappa.csv', index=False)
print(kp.to_string())

# ---- Fig. 3: sign agreement of the reranker pairs that involve Qwen3-Reranker-4B (relevant-set-identical units).
# The pooled PP/PL groups of the main text are not changed; Qwen pairs are reported on their own.
g['ALL6'] = g['ALL6'] + ['qwen']
g['PW'] = g['PW'] + ['qwen']   # kind(): Qwen counts as pointwise
st = pd.DataFrame(sign_table(EVAL, 'evaluation', subset='same_rel'))
st = st[st.pair.str.contains('Qwen3') | st.pair.str.startswith('[')]
st.to_csv(f'{OUT}/qwen_sign.csv', index=False)
print(st.to_string())
