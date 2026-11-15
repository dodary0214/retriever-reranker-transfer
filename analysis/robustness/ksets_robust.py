"""Robustness tables by k group + gap-matched pair-type comparison (definitions fixed before computing). CPU only, existing scores only. Executes analyze.py's source up to
'# ... run everything' (writes no files), with one textual change: table3()'s per-direction rows keep their bootstrap
arrays. Writes $RRT_OUT/derived/ksets_robust_*.csv."""
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
pre = src[:src.index('# ================================================================ run everything')]
a = "            v, lo, hi, _ = ratio(num, den, keys)\n            per_dir.append(dict(scope=scope, set=s, direction=f'{a}→{b}', retention=v, lo=lo, hi=hi))"
assert pre.count(a) == 1
pre = pre.replace(a, "            v, lo, hi, bb_ = ratio(num, den, keys)\n            per_dir.append(dict(scope=scope, set=s, "
                     "direction=f'{a}→{b}', retention=v, lo=lo, hi=hi, _b=bb_))")
g = {'__file__': AN, '__name__': 'analyze_prefix'}
exec(compile(pre, AN, 'exec'), g)
EVAL, PAIRS, SETS, PL, ALL6, LW, NAME = (g[x] for x in ('EVAL', 'PAIRS', 'SETS', 'PL', 'ALL6', 'LW', 'NAME'))
codes, kappa_w, BW, ci, same_rel, TH, RETRIEVERS = (g[x] for x in ('codes', 'kappa_w', 'BW', 'ci', 'same_rel', 'TH', 'RETRIEVERS'))
REL, QRELS, DLSET, top, DSETS, BUNDLE = g['REL'], g['QRELS'], g['DLSET'], g['top'], g['DSETS'], g['BUNDLE']
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
n = len(EVAL)
print('evaluation queries', n)

COMBOS = [tuple(c) for c in itertools.combinations(ALL6, 4)]
KOF = {c: sum(x in LW for x in c) for c in COMBOS}
GRP = {k: [c for c in KOF if KOF[c] == k] for k in (0, 1, 2)}
assert [len(GRP[k]) for k in (0, 1, 2)] == [1, 8, 6]


def kap(keys, rr, p, mask, metric, th):
    ca, cb = codes(metric, keys, rr, p[0], th), codes(metric, keys, rr, p[1], th)
    k, nn, _, _ = kappa_w(ca, cb, mask, len(rr), np.ones((1, len(keys)), np.float32))
    kb = kappa_w(ca, cb, mask, len(rr), BW(keys))[0]
    return float(k[0]), int(nn[0]), kb


def group_rows(vals, boots, ns):
    """vals/boots/ns: dict combo -> value / bootstrap array / n_notie; returns group summary incl. k0-k2"""
    r = {}
    gb = {}
    for k in (0, 1, 2):
        mem = GRP[k]
        r[f'k{k}'] = float(np.mean([vals[c] for c in mem]))
        gb[k] = np.nanmean(np.vstack([boots[c] for c in mem]), 0)
        r[f'k{k}_lo'], r[f'k{k}_hi'] = ci(gb[k])
        if ns is not None:
            r[f'n_k{k}_mean'] = float(np.mean([ns[c] for c in mem]))
            r[f'n_k{k}_min'], r[f'n_k{k}_max'] = min(ns[c] for c in mem), max(ns[c] for c in mem)
    r['k1_set_min'], r['k1_set_max'] = min(vals[c] for c in GRP[1]), max(vals[c] for c in GRP[1])
    r['k2_set_min'], r['k2_set_max'] = min(vals[c] for c in GRP[2]), max(vals[c] for c in GRP[2])
    r['diff_k0_k2'] = r['k0'] - r['k2']
    r['diff_lo'], r['diff_hi'] = ci(gb[0] - gb[2])
    return r


# ================================================================ R1: robustness table (per retriever pair)
SR = same_rel(EVAL)
REL2 = {d: (q[q.relevance >= 2] if d in DLSET else q[q.relevance >= 1]).groupby('query_id').doc_id.apply(set).to_dict()
        for d, q in QRELS.items()}
SR2 = {p: np.array([(set(top(k, p[0])) & REL2[k[0]].get(k[1], set())) == (set(top(k, p[1])) & REL2[k[0]].get(k[1], set()))
                    for k in EVAL]) for p in PAIRS}
SETTINGS = [('default_all', 'nDCG@10', 0.01, 'all'), ('default_rel', 'nDCG@10', 0.01, 'same_rel'),
            ('tie_0.005', 'nDCG@10', 0.005, 'same_rel'), ('tie_0.02', 'nDCG@10', 0.02, 'same_rel'),
            ('mrr_0.005', 'RR@10', 0.005, 'same_rel'), ('mrr_0.01', 'RR@10', 0.01, 'same_rel'),
            ('mrr_0.02', 'RR@10', 0.02, 'same_rel'), ('judged_0.005', 'nDCG_judged', 0.005, 'same_rel'),
            ('judged_0.01', 'nDCG_judged', 0.01, 'same_rel'), ('judged_0.02', 'nDCG_judged', 0.02, 'same_rel'),
            ('dl_grade2', 'nDCG@10', 0.01, 'same_rel2')]


def r1_rows(settings, combos_of, tag=''):
    rows, setrows = [], []
    for name, metric, th, sub in settings:
        for p in PAIRS:
            mask = {'all': np.ones(n, bool), 'same_rel': SR[p], 'same_rel2': SR2[p]}[sub]
            vals, boots, ns = {}, {}, {}
            for c in COMBOS:
                rr = combos_of(c)
                vals[c], ns[c], boots[c] = kap(EVAL, rr, p, mask, metric, th)
                setrows.append(dict(setting=name + tag, pair=PL(p), set='+'.join(NAME[x] for x in rr), k=KOF[c],
                                    kappa=vals[c], n_notie=ns[c], lo=ci(boots[c])[0], hi=ci(boots[c])[1]))
            r = dict(setting=name + tag, metric=metric, th=th, subset=sub, pair=PL(p), n_queries=int(mask.sum()))
            r.update(group_rows(vals, boots, ns))
            rows.append(r)
    return rows, setrows


R1, R1S = r1_rows(SETTINGS, list)
print('R1 main settings done')

# ---- Qwen3-Reranker-4B in place of MiniLM-L6 in all 15 sets (scores loaded as in appendix_qwen.py)
S = pd.concat([pd.read_parquet(f).assign(dataset=os.path.basename(f).split('-', 1)[1][:-8])
               for f in sorted(glob.glob(f'{CS}/*.parquet'))])
assert not S.duplicated(['dataset', 'qid', 'docid']).any()
SC = {(d, q): dict(zip(x.docid, x.score)) for (d, q), x in S.groupby(['dataset', 'qid'])}
assert all(k in SC for k in EVAL)
JUDGED, RUNS, M = g['JUDGED'], g['RUNS'], g['M']
rows_m = []
for d in sorted({k[0] for k in EVAL}):
    qs = [q for dd, q in EVAL if dd == d]
    for r in RETRIEVERS:
        lines = [(q, doc, float(SC[(d, q)][doc])) for q in qs for doc, _ in RUNS[(d, r)][q]]
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
MW_ORIG = g['MW']
g['MW'] = {m: pd.concat([M, MQ], ignore_index=True).pivot_table(index=['dataset', 'qid'], columns=['retriever', 'reranker'],
                                                                  values=m)
           for m in ('nDCG@10', 'RR@10', 'Judged@10', 'nDCG_judged')}
NAME['qwen'] = 'Qwen3-Reranker-4B'
q_of = lambda c: ['qwen' if x == 'minilm' else x for x in c]
RQ, RQS = r1_rows([('qwen_all', 'nDCG@10', 0.01, 'all'), ('qwen_rel', 'nDCG@10', 0.01, 'same_rel')], q_of)
# check: the Qwen k=0 set reproduces derived/qwen_kappa.csv (PW4q)
qk = pd.read_csv(f'{OUT}/qwen_kappa.csv')
qk = qk[(qk.scope == 'evaluation') & (qk.set == 'PW4q')]
for r in RQ:
    ref = qk[(qk.subset == r['subset']) & (qk.pair == r['pair'])].kappa.iloc[0]
    assert abs(ref - r['k0']) < 1e-9, (r['pair'], ref, r['k0'])
print('Qwen rows done; k=0 matches qwen_kappa.csv')
R1 = pd.DataFrame(R1 + RQ)
R1S = pd.DataFrame(R1S + RQS)
# check: default rows reproduce ksets_kappa_by_pair.csv (all 15 sets)
kbp = pd.read_csv(f'{OUT}/ksets_kappa_by_pair.csv')
chk = R1S[R1S.setting.isin(['default_all', 'default_rel'])]
mx = 0.0
for _, r in chk.iterrows():
    sub = 'all' if r.setting == 'default_all' else 'same_rel'
    ref = kbp[(kbp.subset == sub) & (kbp.pair == r.pair) & (kbp.set == r.set)].kappa
    mx = max(mx, abs(ref.iloc[0] - r.kappa))
print('default rows vs ksets_kappa_by_pair max |diff|', mx)
assert mx < 1e-9

# ================================================================ R2: per collection / dataset
GROUPS = [(b, [k for k in EVAL if BUNDLE(k[0]) == b]) for b in ('TREC DL', 'BEIR')] + \
         [(d, [k for k in EVAL if k[0] == d]) for d in DSETS if any(k[0] == d for k in EVAL)]
R2 = []
for sc, keys in GROUPS:
    SRs = same_rel(keys)
    vals, boots, nq = {}, {}, {}
    for c in COMBOS:
        vv, bb = [], []
        for p in PAIRS:
            k, _, kb = kap(keys, list(c), p, SRs[p], 'nDCG@10', TH)
            vv.append(k)
            bb.append(kb)
        vals[c] = float(np.mean(vv))
        boots[c] = np.nanmean(np.vstack(bb), 0)
    r = dict(scope=sc, n_queries=len(keys), **{f'nrel_{PL(p)}': int(SRs[p].sum()) for p in PAIRS})
    r.update(group_rows(vals, boots, None))
    R2.append(r)
R2 = pd.DataFrame(R2)
# check against analyze.py appendixC_kappa.csv (PW4 = k=0) where defined
ck = pd.read_csv(f'{OUT_ROOT}/tables/appendixC_kappa.csv')
for _, r in R2.iterrows():
    z = ck[(ck.scope == r.scope) & (ck.subset == 'same_rel') & (ck.set == 'PW4')].kappa
    if len(z) and np.isfinite(r.k0):
        assert abs(z.mean() - r.k0) < 1e-9 or not np.isfinite(z.mean()), (r.scope, z.mean(), r.k0)
print('R2 done')

# ================================================================ R3: gain retention by direction
g['MW'] = MW_ORIG
R3v, R3b = {}, {}
for c in COMBOS:
    nm = '+'.join(c)
    SETS[nm] = list(c)
    _, pdir = g['table3'](EVAL, 'evaluation', nm)
    for x in pdir:
        R3v[(c, x['direction'])] = x['retention']
        R3b[(c, x['direction'])] = x['_b']
R3 = []
for dr in dict.fromkeys(d for _, d in R3v):
    r = dict(direction=dr)
    r.update(group_rows({c: R3v[(c, dr)] for c in COMBOS}, {c: R3b[(c, dr)] for c in COMBOS}, None))
    R3.append(r)
R3 = pd.DataFrame(R3)
old = pd.read_csv(f'{OUT_ROOT}/tables/table3_retention_by_direction.csv')
for _, r in R3.iterrows():
    assert abs(old[(old.set == 'PW4') & (old.direction == r.direction)].retention.iloc[0] - r.k0) < 1e-9
print('R3 done; k=0 matches table3_retention_by_direction.csv')

# ================================================================ G: gap-matched pair-type comparison
v3 = pd.read_csv(f'{OUT_ROOT}/figures/fig3_values.csv')
v3 = v3[(v3.subset == 'same_rel') & ~v3.pair.str.startswith('[') & v3.group.isin(['PP', 'PL'])]
gapmax = float(v3[v3.group == 'PP'].abs_diff.max())
sel = {'PP': v3[v3.group == 'PP'].pair.tolist(), 'PL': v3[(v3.group == 'PL') & (v3.abs_diff <= gapmax)].pair.tolist()}
X = g['X']
W = BW(EVAL)
acc = {}
for r1, r2 in itertools.combinations(ALL6, 2):
    pn = f'{NAME[r1]}–{NAME[r2]}'
    if pn not in sel['PP'] + sel['PL']:
        continue
    Sg = {}
    for r in RETRIEVERS:
        d = X('nDCG@10', EVAL, [r1], r)[:, 0] - X('nDCG@10', EVAL, [r2], r)[:, 0]
        Sg[r] = np.where(d >= TH, 1, np.where(d <= -TH, -1, 0))
    num = sum(((Sg[a] != 0) & (Sg[b] != 0) & (Sg[a] == Sg[b]) & SR[(a, b)]).astype(float) for a, b in PAIRS)
    den = sum(((Sg[a] != 0) & (Sg[b] != 0) & SR[(a, b)]).astype(float) for a, b in PAIRS)
    acc[pn] = (num, den)
GR_, GB_ = [], {}
for t in ('PP', 'PL'):
    num = sum(acc[p][0] for p in sel[t])
    den = sum(acc[p][1] for p in sel[t])
    v, lo, hi, b = g['ratio'](num, den, EVAL)
    GB_[t] = b
    GR_.append(dict(type=t, pairs='; '.join(sel[t]), n_pairs=len(sel[t]), units=int(den.sum()), agree=v, lo=lo, hi=hi,
                    gap_max_pp=gapmax))
GR_ = pd.DataFrame(GR_).set_index('type')
assert abs(GR_.loc['PP', 'agree'] - v3[v3.group == 'PP'].agree.mul(v3[v3.group == 'PP'].units).sum()
           / v3[v3.group == 'PP'].units.sum()) < 1e-9
d = GB_['PP'] - GB_['PL']
dv = GR_.loc['PP', 'agree'] - GR_.loc['PL', 'agree']
dlo, dhi = ci(d)
nonover = GR_.loc['PL', 'hi'] < GR_.loc['PP', 'lo']
verdict = 'A' if (GR_.loc['PL', 'agree'] < GR_.loc['PP', 'agree'] and (nonover or dlo > 0)) else 'B'
GR_ = GR_.reset_index()
GR_ = pd.concat([GR_, pd.DataFrame([dict(type='PP-PL', agree=dv, lo=dlo, hi=dhi, gap_max_pp=gapmax,
                                         pairs=f'CI non-overlap {bool(nonover)}; verdict {verdict}')])])

for nm_, df in (('A3', R1), ('A3_sets', R1S), ('B1', R2), ('S3', R3), ('gapmatched', GR_)):
    df.to_csv(f'{OUT}/ksets_robust_{nm_}.csv', index=False)
print(R1[['setting', 'pair', 'n_queries', 'k0', 'k0_lo', 'k0_hi', 'n_k0_mean', 'k1', 'k2', 'n_k2_mean', 'diff_k0_k2',
          'diff_lo', 'diff_hi']].round(3).to_string())
print(R2.drop(columns=[c for c in R2.columns if c.endswith(('_min', '_max'))]).round(3).to_string())
print(R3[['direction', 'k0', 'k0_lo', 'k0_hi', 'k1', 'k2', 'diff_k0_k2', 'diff_lo', 'diff_hi']].round(3).to_string())
print(GR_.round(3).to_string())
print('GAP-MATCHED VERDICT', verdict)
