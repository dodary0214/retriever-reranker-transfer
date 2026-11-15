"""Main analysis of Experiment 1 (and the Experiment 2 summary values). Every main quantity is computed on one set, the
evaluation queries with outputs of all six rerankers: E (547 queries) + F (742 queries) = 1,289 queries. Breakdowns by
collection (TREC DL / BEIR) and dataset are computed on the same set. H1-H3 (set F only) are reported in RESULTS.md only.
CPU only. Reads $RRT_DATA, $RRT_CONTROLLED_DATA and $RRT_CONTROLLED_RESULTS; writes $RRT_OUT/tables/*.csv|tex,
$RRT_OUT/figures/*.pdf|csv and $RRT_OUT/RESULTS.md (RESULTS_existing.md when set F is absent). Some labels in the output
tables and RESULTS.md are in Korean; docs/output_labels.md translates them."""
import glob
import itertools
import json
import os
import sys
import time

import ir_measures
import numpy as np
import pandas as pd
from ir_measures import RR, Judged, nDCG
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import (CONTROLLED_DATA, CONTROLLED_RESULTS, EXP, LISTWISE_E, LISTWISE_F, OUT, POINTWISE, REPO, RETRIEVERS,
                    read_run)

T0 = time.time()
P = OUT
P2 = POINTWISE
MIX = LISTWISE_E
B = LISTWISE_F
NB, SEED, TH, EPS = 10000, 0, 0.01, 1e-9
PW = ['minilm', 'bge-base', 'bge-m3', 'monot5']
LW = ['zephyr', 'first']
ALL6 = PW + LW
SETS = {'PW4': PW, 'MIX4': ['bge-m3', 'monot5', 'zephyr', 'first']}
NAME = {'minilm': 'MiniLM', 'bge-base': 'bge-base', 'bge-m3': 'bge-m3', 'monot5': 'monoT5', 'zephyr': 'RankZephyr',
        'first': 'FIRST'}
SIZE = {'minilm': '22M', 'bge-base': '278M', 'bge-m3': '568M', 'monot5': '220M', 'zephyr': '7B', 'first': '7B'}
PAIRS = list(itertools.combinations(RETRIEVERS, 2))
PL = lambda p: f'{p[0]}–{p[1]}'
kind = lambda p: 'PP' if all(x in PW for x in p) else 'LL' if all(x in LW for x in p) else 'PL'
os.makedirs(f'{P}/tables', exist_ok=True)
os.makedirs(f'{P}/figures', exist_ok=True)
LOG = []


def log(*a):
    s = ' '.join(map(str, a))
    LOG.append(s)
    print(f'[{time.time() - T0:6.1f}s]', s, flush=True)


# ================================================================ query sets
g0 = pd.read_parquet(f'{P2}/grid.parquet')
DSETS = list(dict.fromkeys(g0.dataset))
RUNS = {(d, r): read_run(f'{EXP}/common/runs/{d}.{r}.top100.txt') for d in DSETS for r in RETRIEVERS}
K0 = sorted(set(map(tuple, g0[['dataset', 'qid']].drop_duplicates().values)), key=lambda k: (DSETS.index(k[0]), k[1]))
ALLQ = [k for k in K0 if all(len(RUNS[(k[0], r)].get(k[1], [])) > 0 for r in RETRIEVERS)]
EQ_set = set(map(tuple, pd.read_csv(f'{MIX}/queries_mix.csv', dtype={'qid': str})[['dataset', 'qid']].values))
E = [k for k in ALLQ if k in EQ_set]
HAVE_B = os.path.exists(f'{B}/F/out/B_DONE')
F = []
if HAVE_B:
    FQ = set(map(tuple, pd.read_csv(f'{B}/queries_F.csv', dtype={'qid': str})[['dataset', 'qid']].values))
    F = [k for k in ALLQ if k in FQ]
assert len(ALLQ) == 1753 and len(E) == 547
EVAL = [k for k in ALLQ if k in set(E) | set(F)]  # evaluation queries
DLSET = ('dl19', 'dl20')
BUNDLE = lambda d: 'TREC DL' if d in DLSET else 'BEIR'
MAINSC = '평가 질의'
log(f'ALL {len(ALLQ)}, E {len(E)}, F {len(F)}, evaluation {len(EVAL)} (B {"present" if HAVE_B else "absent"})')

# ================================================================ metrics from runs
QRELS, REL, JUDGED = {}, {}, {}
for d in DSETS:
    q = pd.read_csv(f'{EXP}/common/qrels/{d}.tsv', sep='\t', names=['query_id', 'doc_id', 'relevance'],
                    dtype={'query_id': str, 'doc_id': str, 'relevance': int})
    QRELS[d] = q
    REL[d] = q[q.relevance >= 1].groupby('query_id').doc_id.apply(set).to_dict()
    JUDGED[d] = q.groupby('query_id').doc_id.apply(set).to_dict()


def run_paths(d, r, rr):
    if rr in PW:
        return [f'{P2}/runs/{d}.{r}.{rr}.txt']
    return [p for p in (f'{MIX}/runs/{d}.{r}.{rr}.txt', f'{B}/runs/{d}.{r}.{rr}.txt') if os.path.exists(p)]


def read_lines(paths):
    rows = []
    for p in paths:
        for l in open(p):
            q, _, doc, rank, score, _ = l.split()
            rows.append((q, doc, int(rank), float(score)))
    return rows


MET = {}  # (metric) -> {(d, q, r, rr): value}
rows_m = []
for d in DSETS:
    qs = {k[1] for k in ALLQ if k[0] == d}
    for r in RETRIEVERS:
        for rr in ALL6:
            lines = [x for x in read_lines(run_paths(d, r, rr)) if x[0] in qs]
            if not lines:
                continue
            run = pd.DataFrame(lines, columns=['query_id', 'doc_id', 'rank', 'score'])
            got = set(run.query_id)
            jd = JUDGED[d]
            runj = run[[doc in jd.get(q, ()) for q, doc in zip(run.query_id, run.doc_id)]]
            res = {}
            for m, rn in ((nDCG @ 10, run), (RR @ 10, run), (Judged @ 10, run), ('nDCG_judged', runj)):
                meas = nDCG @ 10 if m == 'nDCG_judged' else m
                v = {x.query_id: x.value for x in ir_measures.iter_calc([meas], QRELS[d], rn[['query_id', 'doc_id', 'score']])}
                res[str(m)] = v
            for q in got:
                rows_m.append((d, q, r, rr, *(res[k].get(q, 0.0) for k in ('nDCG@10', 'RR@10', 'Judged@10', 'nDCG_judged'))))
M = pd.DataFrame(rows_m, columns=['dataset', 'qid', 'retriever', 'reranker', 'nDCG@10', 'RR@10', 'Judged@10', 'nDCG_judged'])
# validation against stored grids
gm = pd.read_parquet(f'{MIX}/grid_mix.parquet')
chk = M.merge(gm[gm.reranker.isin(ALL6)], on=['dataset', 'qid', 'retriever', 'reranker'], how='inner')
bad = (chk['nDCG@10'] - chk.ndcg10).abs().max()
log(f'nDCG@10 recomputed vs grid_mix: {len(chk)} rows, max |diff| {bad:.2e}')
assert bad < 1e-6, 'nDCG mismatch with stored grid -> stop'
if HAVE_B:
    fq = M[[(d, q) in set(F) for d, q in zip(M.dataset, M.qid)]]
    assert fq[fq.reranker.isin(LW)].groupby('reranker').size().min() == 3 * len(F), 'B output incomplete'
MW = {m: M.pivot_table(index=['dataset', 'qid'], columns=['retriever', 'reranker'], values=m)
      for m in ('nDCG@10', 'RR@10', 'Judged@10', 'nDCG_judged')}


def X(metric, keys, rr, r):
    return MW[metric].loc[keys][r][rr].to_numpy()


# ================================================================ bootstrap helpers
_BW = {}


def BW(keys, strat=False):
    k = (len(keys), strat, keys[0], keys[-1])
    if k not in _BW:
        rng = np.random.default_rng(SEED)
        n = len(keys)
        W = np.zeros((NB, n), np.float32)
        if not strat:
            for s in range(0, NB, 500):
                idx = rng.integers(0, n, (500, n))
                for i in range(500):
                    W[s + i] = np.bincount(idx[i], minlength=n)
        else:
            ds = np.array([x[0] for x in keys])
            for d in dict.fromkeys(ds):
                pos = np.flatnonzero(ds == d)
                idx = rng.integers(0, len(pos), (NB, len(pos)))
                for i in range(NB):
                    W[i, pos] += np.bincount(idx[i], minlength=len(pos))
        _BW[k] = W
    return _BW[k]


ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))


def ratio(num, den, keys):
    W = BW(keys)
    b = (W @ num) / (W @ den)
    return float(num.sum() / den.sum()), *ci(b), b


# ================================================================ kappa
def codes(metric, keys, rr, r, th):
    x = X(metric, keys, rr, r)
    s = np.sort(x, 1)
    c = np.argmax(x, 1)
    c[(s[:, -1] - s[:, -2]) < th] = len(rr)
    return c


def kappa_w(ca, cb, mask, K, W):
    ok = mask & (ca < K) & (cb < K)
    A = np.eye(K)[np.where(ok, ca, 0)] * ok[:, None]
    Bm = np.eye(K)[np.where(ok, cb, 0)] * ok[:, None]
    n = W @ ok.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        po = (W @ (ok & (ca == cb)).astype(float)) / n
        pa, pb = (W @ A) / n[:, None], (W @ Bm) / n[:, None]
        pe = (pa * pb).sum(1)
        return (po - pe) / (1 - pe), n, po, pe


def top(k, r):
    return [doc for doc, _ in RUNS[(k[0], r)].get(k[1], [])]


_SR = {}


def same_rel(keys):
    kk = (len(keys), keys[0], keys[-1])
    if kk not in _SR:
        _SR[kk] = {p: np.array([(set(top(k, p[0])) & REL[k[0]].get(k[1], set())) ==
                                (set(top(k, p[1])) & REL[k[0]].get(k[1], set())) for k in keys]) for p in PAIRS}
    return _SR[kk]


def kappa_table(keys, scope, metric='nDCG@10', th=TH, sets=('PW4', 'MIX4'), boot=True):
    rows, boots = [], {}
    SR = same_rel(keys)
    one = np.ones((1, len(keys)), np.float32)
    for sub in ('all', 'same_rel'):
        for p in PAIRS:
            mask = np.ones(len(keys), bool) if sub == 'all' else SR[p]
            for s in sets:
                rr = SETS[s]
                ca, cb = codes(metric, keys, rr, p[0], th), codes(metric, keys, rr, p[1], th)
                k, n, po, pe = kappa_w(ca, cb, mask, len(rr), one)
                r = dict(scope=scope, metric=metric, th=th, subset=sub, pair=PL(p), set=s, n_queries=int(mask.sum()),
                         n_notie=int(n[0]), po=float(po[0]), pe=float(pe[0]), kappa=float(k[0]))
                if boot:
                    kb = kappa_w(ca, cb, mask, len(rr), BW(keys))[0]
                    boots[(sub, p, s)] = kb
                    r['lo'], r['hi'] = ci(kb)
                rows.append(r)
            if boot and len(sets) == 2:
                d = boots[(sub, p, 'PW4')] - boots[(sub, p, 'MIX4')]
                lo, hi = ci(d)
                rows.append(dict(scope=scope, metric=metric, th=th, subset=sub, pair=PL(p), set='PW4−MIX4',
                                 n_queries=int(mask.sum()), kappa=rows[-2]['kappa'] - rows[-1]['kappa'], lo=lo, hi=hi))
    return rows


# ================================================================ sign agreement
def sign_table(keys, scope, metric='nDCG@10', th=TH, boot=True, subset='all'):
    """units = (query, retriever pair); subset 'same_rel' keeps the units whose two retrievers have the same relevant set"""
    rows, per = [], {}
    n = len(keys)
    SRm = same_rel(keys) if subset == 'same_rel' else {p: np.ones(n, bool) for p in PAIRS}
    for r1, r2 in itertools.combinations(ALL6, 2):
        S, D = {}, []
        for r in RETRIEVERS:
            d = X(metric, keys, [r1], r)[:, 0] - X(metric, keys, [r2], r)[:, 0]
            D.append(d)
            S[r] = np.where(d >= th, 1, np.where(d <= -th, -1, 0))
        num = sum(((S[a] != 0) & (S[b] != 0) & (S[a] == S[b]) & SRm[(a, b)]).astype(float) for a, b in PAIRS)
        den = sum(((S[a] != 0) & (S[b] != 0) & SRm[(a, b)]).astype(float) for a, b in PAIRS)
        dm = np.mean(D, 0)
        per[(r1, r2)] = (num, den, dm)
        r = dict(scope=scope, subset=subset, metric=metric, th=th, group=kind((r1, r2)), pair=f'{NAME[r1]}–{NAME[r2]}',
                 units=int(den.sum()), agree=num.sum() / den.sum(), mean_diff=dm.mean(), abs_diff=abs(dm.mean()))
        if boot:
            _, r['lo'], r['hi'], _ = ratio(num, den, keys)
            r['diff_lo'], r['diff_hi'] = ci(BW(keys) @ dm / n)
        rows.append(r)
    gb = {}
    for g in ('PP', 'PL', 'LL'):
        ks = [k for k in per if kind(k) == g]
        num, den = sum(per[k][0] for k in ks), sum(per[k][1] for k in ks)
        r = dict(scope=scope, subset=subset, metric=metric, th=th, group=g, pair=f'[{g} pooled, {len(ks)}]', units=int(den.sum()),
                 agree=num.sum() / den.sum(), abs_diff=float(np.mean([abs(per[k][2].mean()) for k in ks])))
        if boot:
            v, r['lo'], r['hi'], gb[g] = ratio(num, den, keys)
        rows.append(r)
    if boot:
        d = gb['PP'] - gb['PL']
        rows.append(dict(scope=scope, subset=subset, metric=metric, th=th, group='PP−PL', pair='[PP − PL]',
                         agree=rows[-3]['agree'] - rows[-2]['agree'], lo=ci(d)[0], hi=ci(d)[1]))
    return rows


# ================================================================ flips
need = {}
for d, q in E + F:
    need.setdefault(d, set()).add(q)
SC = {}
for rr in PW:
    for d in need:
        s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{P2}/scores/{d}.{rr}/part-*.parquet'))])
        s = s[s.qid.astype(str).isin(need[d])]
        for q, grp in s.groupby('qid'):
            SC[(rr, d, str(q))] = dict(zip(grp.docid.astype(str), grp.score))
LR = {}
for m in LW:
    for path in (f'{MIX}/out/{m}.jsonl', f'{B}/F/out/{m}.jsonl'):
        if not os.path.exists(path):
            continue
        for l in open(path):
            x = json.loads(l)
            if x['job_id'].startswith('grid|'):
                _, d, r, q = x['job_id'].split('|')
                n = len(x['ranking'])
                LR[(m, d, r, q)] = {doc: n - i for i, doc in enumerate(x['ranking'])}


def keyf(rr, d, r, q):
    return SC[(rr, d, q)] if rr in PW else LR[(rr, d, r, q)]


def flips_table(keys, scope):
    rows, boots = [], {}
    n = len(keys)
    for rr in ALL6:
        acc = {(t, p): (np.zeros(n), np.zeros(n)) for t in ('rel-nonrel', 'rel-rel') for p in PAIRS}
        for i, (d, q) in enumerate(keys):
            rel = REL[d].get(q, set())
            for p in PAIRS:
                a, b = p
                ka, kb = keyf(rr, d, a, q), keyf(rr, d, b, q)
                ok = [doc for doc in set(top((d, q), a)) & set(top((d, q), b)) if doc in ka and doc in kb]
                R = [doc for doc in ok if doc in rel]
                N = [doc for doc in ok if doc not in rel]
                if not R:
                    continue
                va, vb = np.array([ka[x] for x in R]), np.array([kb[x] for x in R])
                for t, (A2, B2) in (('rel-nonrel', (np.array([ka[x] for x in N]), np.array([kb[x] for x in N]))),
                                    ('rel-rel', (va, vb))):
                    if A2.size == 0:
                        continue
                    sa, sb = np.sign(va[:, None] - A2[None, :]), np.sign(vb[:, None] - B2[None, :])
                    valid = (sa != 0) & (sb != 0)
                    if t == 'rel-rel':
                        valid &= np.triu(np.ones_like(valid, bool), 1)
                    acc[(t, p)][0][i] += ((sa != sb) & valid).sum()
                    acc[(t, p)][1][i] += valid.sum()
        for t in ('rel-nonrel', 'rel-rel'):
            for p in PAIRS + ['pooled']:
                if p == 'pooled':
                    f = sum(acc[(t, x)][0] for x in PAIRS)
                    c = sum(acc[(t, x)][1] for x in PAIRS)
                else:
                    f, c = acc[(t, p)]
                v, lo, hi, b = ratio(f, c, keys)
                boots[(rr, t, p)] = b
                rows.append(dict(scope=scope, reranker=NAME[rr], kind='P' if rr in PW else 'L', pair_type=t,
                                 retriever_pair='pooled' if p == 'pooled' else PL(p), n_pairs=int(c.sum()),
                                 flips=int(f.sum()), flip_rate=v, lo=lo, hi=hi))
    return rows


# ================================================================ figure 5
F5BINS = ['0', '(0, 0.1]', '(0.1, 0.25]', '> 0.25']
F5B = {}


def f5bin(v):
    return 0 if v == 0 else 1 if v <= 0.1 else 2 if v <= 0.25 else 3


def fig5_table(keys, scope):
    """units = (query, retriever pair) with the same relevant set; x = mean of RankZephyr and FIRST relevant–nonrelevant
    replacement flip rates on the shared documents; y = share of units whose top-1 reranker differs between the two
    retrievers (units where either side is tied, th 0.01, are excluded). Query-bootstrap CI."""
    SR = same_rel(keys)
    n = len(keys)
    rows = []
    xb = {p: np.full(n, -1) for p in PAIRS}
    for p in PAIRS:
        a, b = p
        for i, (d, q) in enumerate(keys):
            if not SR[p][i]:
                continue
            rel = REL[d].get(q, set())
            rates = []
            for m in LW:
                ka, kb = keyf(m, d, a, q), keyf(m, d, b, q)
                ok = [doc for doc in set(top((d, q), a)) & set(top((d, q), b)) if doc in ka and doc in kb]
                Rr = [doc for doc in ok if doc in rel]
                N = [doc for doc in ok if doc not in rel]
                if not Rr or not N:
                    break
                sa = np.sign(np.array([ka[x] for x in Rr])[:, None] - np.array([ka[x] for x in N])[None, :])
                sb = np.sign(np.array([kb[x] for x in Rr])[:, None] - np.array([kb[x] for x in N])[None, :])
                rates.append((sa != sb).mean())
            if len(rates) == len(LW):
                xb[p][i] = f5bin(float(np.mean(rates)))
    for s_ in ('MIX4', 'PW4'):
        rr = SETS[s_]
        ch = {p: (codes('nDCG@10', keys, rr, p[0], TH), codes('nDCG@10', keys, rr, p[1], TH)) for p in PAIRS}
        for bi, bl in enumerate(F5BINS):
            num, den, nu = np.zeros(n), np.zeros(n), 0
            for p in PAIRS:
                ca, cb = ch[p]
                ok = (xb[p] == bi) & (ca < len(rr)) & (cb < len(rr))
                num += (ok & (ca != cb))
                den += ok
                nu += int(((xb[p] == bi)).sum())
            if den.sum() == 0:
                rows.append(dict(scope=scope, set=s_, bin=bl, units_in_bin=nu, units_notie=0, n_queries=0))
                continue
            v, lo, hi, bb = ratio(num, den, keys)
            F5B[(scope, s_, bl)] = bb
            rows.append(dict(scope=scope, set=s_, bin=bl, units_in_bin=nu, units_notie=int(den.sum()),
                             n_queries=int((den > 0).sum()), change_rate=v, lo=lo, hi=hi))
    nx = sum(int((xb[p] == -1).sum() - (~SR[p]).sum()) for p in PAIRS)
    return rows, nx


# ================================================================ table 3
def bs_index(x, ds):
    """per-dataset best single (column index) and per-query tie-rule argmax"""
    means = {d: x[ds == d].mean(0) for d in dict.fromkeys(ds)}
    bs = {d: int(np.argmax(m)) for d, m in means.items()}
    star = np.empty(len(x), int)
    for i in range(len(x)):
        t = np.flatnonzero(np.abs(x[i] - x[i].max()) < EPS)
        star[i] = t[0] if len(t) == 1 else bs[ds[i]] if bs[ds[i]] in t else t[np.argmax(means[ds[i]][t])]
    return np.array([bs[d] for d in ds]), star


def table3(keys, scope, s):
    rr = SETS[s]
    ds = np.array([k[0] for k in keys])
    WS = BW(keys, strat=True)
    out = dict(scope=scope, set=s, n_queries=len(keys))
    # significance agreement
    concl, concl_b = {}, {}
    for d in dict.fromkeys(ds):
        m = ds == d
        nd = m.sum()
        Wd = WS[:, m]
        for r in RETRIEVERS:
            x = X('nDCG@10', keys, rr, r)[m]
            for a, b in itertools.combinations(range(len(rr)), 2):
                diff = x[:, a] - x[:, b]
                tt = stats.ttest_rel(x[:, a], x[:, b])
                c = 0 if (not np.isfinite(tt.pvalue) or tt.pvalue >= 0.05) else int(np.sign(diff.mean()))
                concl[(d, r, a, b)] = c
                mu = (Wd @ diff) / nd
                var = ((Wd @ diff ** 2) - nd * mu ** 2) / (nd - 1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    t = mu / np.sqrt(var / nd)
                    pv = 2 * stats.t.sf(np.abs(t), nd - 1)
                cb = np.where((var > 1e-15) & (pv < 0.05), np.sign(mu), 0)
                concl_b[(d, r, a, b)] = cb
    units = [(d, a, b, p) for d in dict.fromkeys(ds) for a, b in itertools.combinations(range(len(rr)), 2) for p in PAIRS]
    PV = {}  # metric -> {pair: (value, boot)}
    PV['sig'] = {PL(p): (np.mean([concl[(d, p[0], a, b)] == concl[(d, p[1], a, b)] for d, a, b, pp in units if pp == p]),
                         np.mean([concl_b[(d, p[0], a, b)] == concl_b[(d, p[1], a, b)] for d, a, b, pp in units if pp == p], 0))
                 for p in PAIRS}
    out.update(sig_units=len(units), sig_share_nodiff=np.mean([v == 0 for v in concl.values()]))
    # Kendall tau-b of system ranking

    def taub(ma, mb):  # (..., K)
        K = ma.shape[-1]
        num, na, nb = 0, 0, 0
        for i, j in itertools.combinations(range(K), 2):
            sa, sb = np.sign(ma[..., i] - ma[..., j]), np.sign(mb[..., i] - mb[..., j])
            num = num + sa * sb
            na = na + (sa != 0)
            nb = nb + (sb != 0)
        return num / np.sqrt(na * nb)
    taus, taus_b, tpair = [], [], []
    for d in dict.fromkeys(ds):
        m = ds == d
        Wd = WS[:, m]
        mean = {r: X('nDCG@10', keys, rr, r)[m].mean(0) for r in RETRIEVERS}
        mb = {r: (Wd @ X('nDCG@10', keys, rr, r)[m]) / m.sum() for r in RETRIEVERS}
        for a, b in PAIRS:
            taus.append(taub(mean[a], mean[b]))
            taus_b.append(taub(mb[a], mb[b]))
            tpair.append(PL((a, b)))
    tpair = np.array(tpair)
    PV['tau'] = {PL(p): (float(np.mean(np.array(taus)[tpair == PL(p)])), np.nanmean(np.array(taus_b)[tpair == PL(p)], 0))
                 for p in PAIRS}
    out.update(tau_units=len(taus))
    # gain retention
    W = BW(keys)
    nums, dens = [], []
    BS, STAR, XX = {}, {}, {}
    for r in RETRIEVERS:
        XX[r] = X('nDCG@10', keys, rr, r)
        BS[r], STAR[r] = bs_index(XX[r], ds)
    ix = np.arange(len(keys))
    per_dir, dirs = [], {}
    for a in RETRIEVERS:
        for b in RETRIEVERS:
            if a == b:
                continue
            num = XX[b][ix, STAR[a]] - XX[b][ix, BS[b]]
            den = XX[b].max(1) - XX[b][ix, BS[b]]
            nums.append(num)
            dens.append(den)
            dirs[(a, b)] = (num, den)
            v, lo, hi, _ = ratio(num, den, keys)
            per_dir.append(dict(scope=scope, set=s, direction=f'{a}→{b}', retention=v, lo=lo, hi=hi))
    v, lo, hi, _ = ratio(sum(nums), sum(dens), keys)
    out.update(retention_pooled6=v, retention_pooled6_lo=lo, retention_pooled6_hi=hi)
    PV['retention'] = {}
    for p in PAIRS:
        (n1, d1), (n2, d2) = dirs[p], dirs[(p[1], p[0])]
        v, _, _, b = ratio(n1 + n2, d1 + d2, keys)
        PV['retention'][PL(p)] = (v, b)
    for m, dd in PV.items():  # mean over the three retriever pairs, range, CI of the mean
        vals = np.array([dd[PL(p)][0] for p in PAIRS])
        bm = np.nanmean(np.array([dd[PL(p)][1] for p in PAIRS]), 0)
        out.update({f'{m}_mean': vals.mean(), f'{m}_min': vals.min(), f'{m}_max': vals.max(),
                    f'{m}_lo': ci(bm)[0], f'{m}_hi': ci(bm)[1]})
        out.update({f'{m}_{PL(p)}': dd[PL(p)][0] for p in PAIRS})
    return out, per_dir


# ================================================================ jaccard / same_rel
def cand_table(keys, scope):
    rows = []
    SR = same_rel(keys)
    for p in PAIRS:
        jac = np.array([len(set(top(k, p[0])) & set(top(k, p[1]))) / len(set(top(k, p[0])) | set(top(k, p[1])))
                        for k in keys])
        v, lo, hi, _ = ratio(jac, np.ones(len(keys)), keys)
        s, slo, shi, _ = ratio(SR[p].astype(float), np.ones(len(keys)), keys)
        rows.append(dict(scope=scope, pair=PL(p), n=len(keys), jaccard=v, jac_lo=lo, jac_hi=hi, same_rel=s,
                         same_rel_lo=slo, same_rel_hi=shi))
    return rows



# ================================================================ same-input reruns
GR = {d: {q: dict(zip(g.doc_id, g.relevance)) for q, g in QRELS[d].groupby('query_id')} for d in DSETS}


def ndcg10(ranking, g):
    dcg = sum(g.get(t, 0) / np.log2(i + 2) for i, t in enumerate(ranking[:10]))
    ideal = sorted(g.values(), reverse=True)[:10]
    idcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


ORIG = {}
for m in LW:
    for l in open(f'{MIX}/out/{m}.jsonl'):
        x = json.loads(l)
        if x['job_id'].startswith('grid|'):
            _, d, r, q = x['job_id'].split('|')
            ORIG[(m, d, r, q)] = x['ranking']
RERUN = {}
for sub in ('rep', 'rep50'):
    for m in LW:
        f = f'{B}/{sub}/out/{m}.jsonl'
        if os.path.exists(f):
            for l in open(f):
                x = json.loads(l)
                _, d, r, q = x['job_id'].split('|')
                RERUN[(m, d, r, q)] = x['ranking']
# own nDCG must reproduce the stored grid on the original rankings
_chk = max(abs(ndcg10(ORIG[k], GR[k[1]].get(k[3], {})) - MW['nDCG@10'].loc[(k[1], k[3])][(k[2], k[0])])
           for k in list(RERUN)[:60])
log(f'same-input: {len(RERUN)} rerun lists; own nDCG vs grid max |diff| {_chk:.1e}')
assert _chk < 1e-9


def same_input():
    rows, krows = [], []
    for m in LW:
        qs = sorted({(d, q) for mm, d, r, q in RERUN if mm == m}, key=lambda k: (DSETS.index(k[0]), k[1]))
        if not qs:
            continue
        acc = {t: (np.zeros(len(qs)), np.zeros(len(qs))) for t in ('rel-nonrel', 'rel-rel')}
        same_lists = 0
        for i, (d, q) in enumerate(qs):
            g = GR[d].get(q, {})
            for r in RETRIEVERS:
                a, b = ORIG[(m, d, r, q)], RERUN[(m, d, r, q)]
                same_lists += a == b
                pa, pb = {t: j for j, t in enumerate(a)}, {t: j for j, t in enumerate(b)}
                rel = [t for t in a if g.get(t, 0) >= 1]
                non = [t for t in a if g.get(t, 0) < 1]
                acc['rel-nonrel'][0][i] += sum((pa[u] < pa[v]) != (pb[u] < pb[v]) for u in rel for v in non)
                acc['rel-nonrel'][1][i] += len(rel) * len(non)
                acc['rel-rel'][0][i] += sum((pa[u] < pa[v]) != (pb[u] < pb[v]) for j, u in enumerate(rel) for v in rel[j + 1:])
                acc['rel-rel'][1][i] += len(rel) * (len(rel) - 1) // 2
        for t, (f, c) in acc.items():
            v, lo, hi, _ = ratio(f, c, qs)
            rows.append(dict(reranker=NAME[m], pair_type=t, n_queries=len(qs), n_lists=3 * len(qs),
                             identical_lists=int(same_lists), n_pairs=int(c.sum()), flips=int(f.sum()),
                             flip_rate=v, lo=lo, hi=hi))
    # MIX4 same-input kappa: original vs rerun (RankZephyr replaced by its rerun; FIRST deterministic)
    qs = sorted({(d, q) for mm, d, r, q in RERUN if mm == 'zephyr'}, key=lambda k: (DSETS.index(k[0]), k[1]))
    if qs:
        rr = SETS['MIX4']
        ca_all, cb_all = [], []
        for r in RETRIEVERS:
            x = X('nDCG@10', qs, rr, r)
            y = x.copy()
            y[:, rr.index('zephyr')] = [ndcg10(RERUN[('zephyr', d, r, q)], GR[d].get(q, {})) for d, q in qs]
            for z, out in ((x, ca_all), (y, cb_all)):
                s = np.sort(z, 1)
                c = np.argmax(z, 1)
                c[(s[:, -1] - s[:, -2]) < TH] = len(rr)
                out.append(c)
        ca, cb = np.concatenate(ca_all), np.concatenate(cb_all)
        one = np.ones((1, len(ca)), np.float32)
        k, n, po, pe = kappa_w(ca, cb, np.ones(len(ca), bool), len(rr), one)
        Wq = BW(qs)
        kb = kappa_w(ca, cb, np.ones(len(ca), bool), len(rr), np.tile(Wq, (1, 3)))[0]
        krows.append(dict(set='MIX4', n_queries=len(qs), units=len(ca), n_notie=int(n[0]), po=float(po[0]),
                          pe=float(pe[0]), kappa=float(k[0]), lo=ci(kb)[0], hi=ci(kb)[1]))
    return pd.DataFrame(rows), pd.DataFrame(krows)


SI, SIK = same_input()
log('same-input done')

# ================================================================ run everything
SCOPES = [(MAINSC, EVAL)]
GROUPS = [(b, [k for k in EVAL if BUNDLE(k[0]) == b]) for b in ('TREC DL', 'BEIR')] + \
         [(d, [k for k in EVAL if k[0] == d]) for d in DSETS if any(k[0] == d for k in EVAL)]
P2Q = sorted({tuple(json.loads(l)['qid'].split(':')) for l in open(f'{CONTROLLED_DATA}/controlled/conditions.jsonl')})
KR, SG, FL, T3, T3D, CD, F5 = [], [], [], [], [], [], []
for sc, keys in SCOPES:
    KR += kappa_table(keys, sc)
    SG += sign_table(keys, sc) + sign_table(keys, sc, subset='same_rel')
    FL += flips_table(keys, sc)
    f5, F5_NX = fig5_table(keys, sc)
    F5 += f5
    dl = [k for k in keys if k[0] in ('dl19', 'dl20')]
    if dl:
        FL += [dict(r, subset='실험 2와 같은 질의') for r in flips_table(dl, sc)]
    for s in SETS:
        o, pdir = table3(keys, sc, s)
        T3.append(o)
        T3D += pdir
    CD += cand_table(keys, sc)
    log('scope done', sc)
log('main analyses done')
# appendix: bundles and datasets, same settings as the main text
GK, GS, GF, GT3, GCD = [], [], [], [], []
for sc, keys in GROUPS:
    GK += kappa_table(keys, sc)
    GS += [r for sub in ('all', 'same_rel') for r in sign_table(keys, sc, subset=sub)]
    GF += [r for r in flips_table(keys, sc) if r['retriever_pair'] == 'pooled' and r['kind'] == 'L']
    GT3 += [table3(keys, sc, s_)[0] for s_ in SETS]
    GCD += cand_table(keys, sc)
log('appendix groups done')
# H1–H3: B queries only (v1 definitions); not shown in figures/tables
HK, HS, HF = [], [], []
if HAVE_B:
    HK = kappa_table(F, 'B')
    HS = sign_table(F, 'B')
    HF = flips_table(F, 'B')

AK, AS, ADS = [], [], []
for sc, keys in SCOPES:
    for metric in ('nDCG@10', 'RR@10', 'nDCG_judged'):
        for th in (0.005, 0.01, 0.02):
            if metric == 'nDCG@10' and th == TH:
                continue
            AK += kappa_table(keys, sc, metric, th)
            AS += [r for sub in ('all', 'same_rel') for r in sign_table(keys, sc, metric, th, subset=sub)
                   if r['pair'].startswith('[')]
    ds = np.array([k[0] for k in keys])
    for d in dict.fromkeys(ds):  # threshold sensitivity per dataset
        kd = [k for k in keys if k[0] == d]
        for th in (0.005, 0.01, 0.02):
            for r in kappa_table(kd, sc, 'nDCG@10', th):
                r['dataset'] = d
                ADS.append(r)
log('appendix done')

# ================================================================ table 1
CFG = {  # checkpoint, parameters (from the local safetensors headers), precision, batch, max length, truncation
    'minilm': ('pointwise (cross-encoder)', 'cross-encoder/ms-marco-MiniLM-L-6-v2', '22.7M', 'fp16',
               '64 (query, doc) pairs', '512 tokens', 'only_second (document cut)'),
    'bge-base': ('pointwise (cross-encoder)', 'BAAI/bge-reranker-base', '278.0M', 'fp16', '64 (query, doc) pairs',
                 '512 tokens', 'only_second (document cut)'),
    'bge-m3': ('pointwise (cross-encoder)', 'BAAI/bge-reranker-v2-m3', '567.8M', 'fp16', '64 (query, doc) pairs',
               '512 tokens', 'only_second (document cut)'),
    'monot5': ('pointwise (seq2seq)', 'castorini/monot5-base-msmarco-10k', '222.9M', 'bf16', '64 (query, doc) pairs',
               '512 tokens', 'document tokens cut to fit the prompt'),
    'zephyr': ('listwise (generative)', 'castorini/rank_zephyr_7b_v1_full', '7.24B', 'bf16',
               'vLLM offline, ~500 window calls; window 20, stride 10', '4,096-token context',
               '300 words per passage, cut evenly until the prompt fits'),
    'first': ('listwise (first-token logits)', 'castorini/first_mistral', '7.25B', 'bf16',
              'vLLM offline, ~500 window calls; window 20, stride 10', '8,192-token context',
              '300 words per passage, cut evenly until the prompt fits'),
}
T1 = []
for r in RETRIEVERS:
    for rr in ALL6:
        x = MW['nDCG@10'].loc[EVAL][(r, rr)].to_numpy()
        j = MW['Judged@10'].loc[EVAL][(r, rr)].to_numpy()
        T1.append(dict(scope=MAINSC, retriever=r, reranker=NAME[rr], n=len(EVAL), ndcg10=x.mean(), judged10=j.mean()))
T1 = pd.DataFrame(T1)
t1w = T1.pivot_table(index='reranker', columns='retriever', values=['ndcg10', 'judged10'], sort=False)
t1w.columns = [f'{m}_{r}' for m, r in t1w.columns]
t1b = pd.DataFrame([dict(reranker=NAME[rr], type=c[0], checkpoint=c[1], params=c[2], precision=c[3], batch=c[4],
                         max_length=c[5], truncation=c[6]) for rr, c in CFG.items()]).set_index('reranker')
t1b = t1b.join(t1w[[f'ndcg10_{r}' for r in RETRIEVERS] + [f'judged10_{r}' for r in RETRIEVERS]]).reset_index()
P2Qset = set(P2Q)
qc = pd.DataFrame([dict(bundle=BUNDLE(d), dataset=d, n_all=sum(k[0] == d for k in ALLQ),
                        n_eval=sum(k[0] == d for k in EVAL),
                        mean_rel_per_query=np.mean([len(REL[d].get(k[1], ())) for k in EVAL if k[0] == d])
                        if any(k[0] == d for k in EVAL) else np.nan,
                        experiment2='yes' if any(k[0] == d for k in P2Qset) else 'no') for d in DSETS])
tot = []
for b, g_ in qc.groupby('bundle', sort=False):
    kk = [k for k in EVAL if BUNDLE(k[0]) == b]
    tot.append(dict(bundle=b, dataset='(합계)', n_all=int(g_.n_all.sum()), n_eval=int(g_.n_eval.sum()),
                    mean_rel_per_query=np.mean([len(REL[k[0]].get(k[1], ())) for k in kk]), experiment2=''))
tot.append(dict(bundle='전체', dataset='(합계)', n_all=int(qc.n_all.sum()), n_eval=int(qc.n_eval.sum()),
                mean_rel_per_query=np.mean([len(REL[k[0]].get(k[1], ())) for k in EVAL]), experiment2=''))
qc = pd.concat([qc, pd.DataFrame(tot)], ignore_index=True)
qc['_o'] = [(['TREC DL', 'BEIR', '전체'].index(b), d == '(합계)') for b, d in zip(qc.bundle, qc.dataset)]
qc = qc.sort_values('_o', kind='stable').drop(columns='_o').reset_index(drop=True)


# ================================================================ write tables
def save(df, name, caption=''):
    df.to_csv(f'{P}/tables/{name}.csv', index=False)
    fmt = df.copy()
    for c in df.select_dtypes('number').columns:
        if c.startswith('n') or c.startswith('units') or c in ('flips', 'identical_lists'):
            continue
        fmt[c] = fmt[c].map(lambda v: '' if pd.isna(v) else f'{v:.3f}')
    with open(f'{P}/tables/{name}.tex', 'w') as fh:
        fh.write(fmt.to_latex(index=False, caption=caption, label=f'tab:{name}', escape=True))


KR, SG, FL, T3, T3D, CD, F5 = map(pd.DataFrame, (KR, SG, FL, T3, T3D, CD, F5))
FL['subset'] = FL.get('subset', pd.Series(index=FL.index, dtype=object)).fillna('all')
AK, AS, ADS = map(pd.DataFrame, (AK, AS, ADS))
GK, GS, GF, GT3, GCD, HK, HS, HF = map(pd.DataFrame, (GK, GS, GF, GT3, GCD, HK, HS, HF))

# table 2: rows RankZephyr, FIRST; columns same-input / replacement rel–nonrel by pair + pooled /
# rel–rel pooled / DL19+DL20 rel–nonrel pooled. Long form (csv) + formatted wide (tex).
T2COLS = [('동일 입력 재실행 (관련–비관련)', None, None, None)] + \
         [(f'교체 관련–비관련 {PL(p)}', 'all', 'rel-nonrel', PL(p)) for p in PAIRS] + \
         [('교체 관련–비관련 (전체)', 'all', 'rel-nonrel', 'pooled'), ('교체 관련–관련 (전체)', 'all', 'rel-rel', 'pooled'),
          ('교체 관련–비관련, DL19+DL20만 (전체)', '실험 2와 같은 질의', 'rel-nonrel', 'pooled')]
t2 = []
for m in ('RankZephyr', 'FIRST'):
    for col, sub, t, rp in T2COLS:
        if sub is None:
            z = SI[(SI.reranker == m) & (SI.pair_type == 'rel-nonrel')] if len(SI) else SI
            if not len(z):
                continue
            z = z.iloc[0]
            t2.append(dict(reranker=m, column=col, rate=z.flip_rate, lo=z.lo, hi=z.hi, n_pairs=int(z.n_pairs),
                           n_queries=int(z.n_queries)))
            continue
        z = FL[(FL.scope == MAINSC) & (FL.subset == sub) & (FL.reranker == m) & (FL.pair_type == t) &
               (FL.retriever_pair == rp)].iloc[0]
        t2.append(dict(reranker=m, column=col, rate=z.flip_rate, lo=z.lo, hi=z.hi, n_pairs=int(z.n_pairs),
                       n_queries=len(EVAL) if sub == 'all' else sum(k[0] in DLSET for k in EVAL)))
t2 = pd.DataFrame(t2)
t2w = t2.assign(cell=[f'{r:.3f} [{l:.3f}, {h:.3f}] ({n:,})' for r, l, h, n in zip(t2.rate, t2.lo, t2.hi, t2.n_pairs)]
                ).pivot(index='reranker', columns='column', values='cell').reindex(['RankZephyr', 'FIRST'])
t2w = t2w[[c for c, *_ in T2COLS if c in t2w.columns]].reset_index()

# table 3: mean over the three retriever pairs, range, CI of the mean
# pre-specified: τ is reported as the three retriever-pair values and their range (no bootstrap CI; with four
# systems τ takes few distinct values)
t3 = pd.DataFrame([{'set': r['set'], 'n_queries': r['n_queries'],
                    '유의성 결론 일치율': f"{r['sig_mean']:.3f} [{r['sig_lo']:.3f}, {r['sig_hi']:.3f}]",
                    '유의성 결론 일치율 범위': f"{r['sig_min']:.3f}–{r['sig_max']:.3f}",
                    '시스템 순위 τ (검색기 쌍 3개)': ' / '.join(f"{r[f'tau_{PL(p)}']:.3f}" for p in PAIRS),
                    '시스템 순위 τ 범위': f"{r['tau_min']:.3f}–{r['tau_max']:.3f}",
                    '이득 유지율': f"{r['retention_mean']:.3f} [{r['retention_lo']:.3f}, {r['retention_hi']:.3f}]",
                    '이득 유지율 범위': f"{r['retention_min']:.3f}–{r['retention_max']:.3f}"}
                   for _, r in T3.iterrows()])
# note under Table 3: mean nDCG@10 gaps between the models of each set (from the Table 1(b) values)
gap = []
for s_, rr in SETS.items():
    for r in RETRIEVERS:
        v = np.sort([T1[(T1.retriever == r) & (T1.reranker == NAME[x])].ndcg10.iloc[0] for x in rr])
        gap.append(dict(set=s_, retriever=r, max_minus_min=v[-1] - v[0], min_adjacent_gap=np.diff(v).min()))
GAP3 = pd.DataFrame(gap)
GAP3 = pd.concat([GAP3, GAP3.groupby('set', sort=False)[['max_minus_min', 'min_adjacent_gap']].mean().reset_index()
                  .assign(retriever='mean')], ignore_index=True)

T3NOTE = 'τ: four systems, so τ can take only a few distinct values; the three retriever-pair values and their range are shown.'
# smallest gap between adjacent ranks (mean over retrievers) goes into the caption note
_g = GAP3[GAP3.retriever == 'mean'].set_index('set').min_adjacent_gap
T3NOTE += (f' Smallest mean nDCG@10 gap between adjacent-ranked models (mean over retrievers): PW4 {_g["PW4"]:.4f}, '
           f'MIX4 {_g["MIX4"]:.4f}.')
save(qc, 'table1a_queries', 'Datasets: all queries (runs from the three retrievers), evaluation queries (all six '
     'rerankers), mean relevant documents per evaluation query, use in experiment 2')
save(t1b, 'table1b_rerankers', 'Rerankers: configuration, mean nDCG@10 and Judged@10 per retriever (evaluation queries)')
save(t2, 'table2_replacement_flips', 'Replacement and same-input flips of listwise rerankers (long form)')
with open(f'{P}/tables/table2_replacement_flips_wide.tex', 'w') as fh:
    fh.write(t2w.to_latex(index=False, caption='Flips of listwise rerankers: rate [95\\% CI] (pairs). Pointwise '
                          'rerankers are 0 by construction.', label='tab:table2', escape=True))
t2w.to_csv(f'{P}/tables/table2_replacement_flips_wide.csv', index=False)
save(FL, 'table2_full_incl_pointwise')
save(T3, 'table3_conclusions_full', 'Agreement of evaluation conclusions: per retriever pair, mean, range, CI')
save(t3, 'table3_conclusions', 'Agreement of evaluation conclusions across retrievers (significance agreement and '
     'gain retention: mean over the three retriever pairs [95% CI], range). ' + T3NOTE)
save(T3D, 'table3_retention_by_direction')
save(GAP3, 'table3_note_model_gaps', 'Mean nDCG@10 gaps between the models of each set (from Table 1(b))')
save(KR, 'appendixA_kappa', 'Appendix A: κ, all cells')
save(AK, 'appendixB_kappa_sensitivity', 'Appendix B: κ sensitivity (tie threshold, MRR@10, judged documents only)')
save(AS, 'appendixB_sign_sensitivity', 'Appendix B: sign agreement sensitivity (pooled groups)')
save(ADS, 'appendixB_kappa_by_dataset_threshold')
save(GK, 'appendixC_kappa', 'Appendix C: κ by bundle and dataset')
save(GS, 'appendixC_sign', 'Appendix C: sign agreement by bundle and dataset')
save(GF, 'appendixC_flips', 'Appendix C: replacement flips (listwise, pooled retriever pairs) by bundle and dataset')
save(GT3, 'appendixC_table3', 'Appendix C: Table 3 quantities by bundle and dataset')
save(GCD, 'appendixC_candidates', 'Appendix C: candidate overlap by bundle and dataset')
save(SI, 'appendixD_same_input_flips', 'Appendix D: same-input rerun flips')
save(SIK, 'appendixD_same_input_kappa', 'Appendix D: MIX4 κ, original vs same-input rerun')
save(SG, 'fig3_sign_agreement')
save(CD, 'fig2a_candidates')
save(F5, 'fig5_bins')

# ================================================================ figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CM = 1 / 2.54
INK, INK2, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
CP, CL, CL2 = '#1f4e79', '#d85a30', '#f3a878'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'axes.titlesize': 8, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'pdf.fonttype': 42})
ERR = dict(fmt='none', ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
MAIN = MAINSC
ENG = {MAINSC: f'{len(EVAL)} evaluation queries'}
CAPN = []  # caption query / pair counts

# ---- Fig 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17 * CM, 6.8 * CM), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 2.2]})
cd = CD[CD.scope == MAIN]
x = np.arange(3)
a1.bar(x - 0.18, cd.jaccard, 0.34, color='#bdbbb3', label='top-100 Jaccard')
a1.bar(x + 0.18, cd.same_rel, 0.34, color='#6f6e69', label='same relevant set')
a1.errorbar(x - 0.18, cd.jaccard, yerr=[cd.jaccard - cd.jac_lo, cd.jac_hi - cd.jaccard], **ERR)
a1.errorbar(x + 0.18, cd.same_rel, yerr=[cd.same_rel - cd.same_rel_lo, cd.same_rel_hi - cd.same_rel], **ERR)
a1.set_xticks(x, [p.replace('–', '–\n') for p in cd.pair])
a1.set_ylim(0, 1)
a1.set_title('(a) candidate overlap', pad=20)
a1.legend(loc='upper left', fontsize=7)
a1.grid(axis='y', color=GRID, lw=0.5)
CAPN.append(dict(item='Fig. 2(a)', n_queries=int(cd.n.iloc[0])))
pos, lab = [], []
for gi, sub in enumerate(['all', 'same_rel']):
    for pi, p in enumerate([PL(p) for p in PAIRS]):
        base = gi * 4 + pi
        for si_, (s, c) in enumerate((('PW4', CP), ('MIX4', CL))):
            r = KR[(KR.scope == MAIN) & (KR.subset == sub) & (KR.pair == p) & (KR.set == s)].iloc[0]
            xx = base + (si_ - 0.5) * 0.4
            a2.errorbar(xx, r.kappa, yerr=[[r.kappa - r.lo], [r.hi - r.kappa]], color=c, ecolor=c, ms=4.5,
                        marker='o', mec=c, elinewidth=0.8, capsize=1.5, ls='none')
            CAPN.append(dict(item=f'Fig. 2(b) {sub} {p} {s}', n_queries=int(r.n_queries), n_notie=int(r.n_notie)))
        pos.append(base)
        lab.append(p.replace('–', '–\n'))
if len(SIK):
    kv = SIK.iloc[0]
    a2.axhline(kv.kappa, color=CL, lw=0.9, ls='--', zorder=1)
    CAPN.append(dict(item='Fig. 2(b) same-input κ', n_queries=int(kv.n_queries), n_notie=int(kv.n_notie)))
a2.set_xticks(pos, lab)
for gi, t in enumerate(['all queries', 'same relevant set']):
    a2.text(gi * 4 + 1, 1.04, t, ha='center', va='bottom', fontsize=8, transform=a2.get_xaxis_transform())
a2.set_ylim(0, 1.05)
a2.set_ylabel('Cohen κ (non-tied queries)')
a2.set_title(f'(b) κ of the per-query best reranker ({ENG[MAIN]})', pad=20)
a2.grid(axis='y', color=GRID, lw=0.5)
hd = [Patch(color=CP, label='PW4 (pointwise)'), Patch(color=CL, label='MIX4 (with listwise)')]
if len(SIK):
    hd.append(Line2D([], [], color=CL, ls='--', label=f'MIX4 same-input rerun κ ({int(SIK.iloc[0].n_queries)} q)'))
fig.legend(handles=hd, loc='outside lower center', ncols=3, fontsize=7)
fig.savefig(f'{P}/figures/fig2_overlap_kappa.pdf')
plt.close(fig)
kr_fig = KR[KR.set.isin(['PW4', 'MIX4']) & (KR.scope == MAIN)]
pd.concat([cd.assign(panel='a'), kr_fig.assign(panel='b'), SIK.assign(panel='b same-input')]).to_csv(
    f'{P}/figures/fig2_values.csv', index=False)

# ---- Fig 3 (same relevant set units, six rerankers, 15 pairs, pooled mean line per type)
fig, ax = plt.subplots(figsize=(9 * CM, 7.5 * CM), layout='constrained')
GC = {'PP': CP, 'PL': '#8a8a8a', 'LL': CL}
q3 = SG[(SG.scope == MAIN) & (SG.subset == 'same_rel')]
for g, c in GC.items():
    z = q3[(q3.group == g) & ~q3.pair.str.startswith('[')]
    ax.errorbar(z.abs_diff, z.agree, yerr=[z.agree - z.lo, z.hi - z.agree], color=c, ecolor=c, ms=5, ls='none',
                marker='o', mec=c, elinewidth=0.7, capsize=1.5)
    pooled = q3[q3.pair.str.startswith(f'[{g} pooled')].iloc[0]
    ax.axhline(pooled.agree, color=c, lw=0.9, ls='--', zorder=1)
CAPN.append(dict(item='Fig. 3', n_queries=len(EVAL),
                 n_units=int(q3[q3.pair.str.startswith('[')].units.max())))
ax.set_xlabel('|mean nDCG@10 difference|')
ax.set_ylabel('sign agreement across retrievers')
ax.set_title(f'{ENG[MAIN]}, same relevant set')
ax.grid(color=GRID, lw=0.5)
hd = [Patch(color=c, label={'PP': 'PP (pointwise–pointwise)', 'PL': 'PL (pointwise–listwise)',
                            'LL': 'LL (listwise–listwise)'}[g]) for g, c in GC.items()]
hd.append(Line2D([], [], color=INK2, ls='--', label='pooled mean per type'))
fig.legend(handles=hd, loc='outside lower center', ncols=2, fontsize=7)
fig.savefig(f'{P}/figures/fig3_sign_vs_gap.pdf')
plt.close(fig)
q3.to_csv(f'{P}/figures/fig3_values.csv', index=False)

# ---- Fig 4 (experiment 2; recomputed from per-unit data with this pipeline's bootstrap: 10,000, seed 0)
P1 = CONTROLLED_RESULTS
m1 = pd.read_csv(f'{P1}/m1_reversal_by_model.csv')
U4 = {m: pd.read_parquet(f'{P1}/per_unit_{m}.parquet') for m in LW}
Q4 = sorted(set().union(*(u.qid.unique() for u in U4.values())))
IDX4 = np.random.default_rng(SEED).integers(0, len(Q4), size=(NB, len(Q4)))
C = ['C1', 'C2', 'C3', 'C4', 'C5']
ALPHA_B = 0.05 / len(C)  # Bonferroni over the five conditions, per model


def boot4(u, value, by, pair_mean):
    if pair_mean:
        u = u.groupby(['qid', 'pair_id', by])[value].mean().reset_index()
    s_ = u.pivot_table(index='qid', columns=by, values=value, aggfunc='sum').reindex(Q4).fillna(0)
    n_ = u.pivot_table(index='qid', columns=by, values=value, aggfunc='count').reindex(Q4).fillna(0)
    S_, N_ = s_.values[IDX4].sum(1), n_.values[IDX4].sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s_.sum() / n_.sum(), pd.DataFrame(S_ / N_, columns=s_.columns), n_.sum()


f4 = []
for m in LW:
    u = U4[m]
    p_, b_, _ = boot4(u, 'rev', 'cond', True)
    for c in ['R', 'C0'] + C:
        lo, hi = ci(b_[c])
        e = b_[c] - b_['C0']
        elo, ehi = ci(e)
        blo, bhi = np.nanpercentile(e, 100 * ALPHA_B / 2), np.nanpercentile(e, 100 * (1 - ALPHA_B / 2))
        f4.append(dict(panel='a', model=NAME[m], cond=c, rate=p_[c], lo=lo, hi=hi, excess=p_[c] - p_['C0'],
                       excess_lo=elo, excess_hi=ehi, excess_bonf_lo=blo, excess_bonf_hi=bhi,
                       sig_bonferroni=bool(c in C and (blo > 0 or bhi < 0)), n_units=int((u.cond == c).sum()),
                       n_pairs=int(u[u.cond == c][['qid', 'pair_id']].drop_duplicates().shape[0])))
    v = u[u.unequal & (u.cond == 'C1')]
    v = v.assign(grp=np.where(v.pref0_correct == 1, 'base correct', 'base wrong'))
    p_, b_, _ = boot4(v, 'rev', 'grp', True)
    for g in ('base wrong', 'base correct'):
        lo, hi = ci(b_[g])
        w = v[v.grp == g]
        f4.append(dict(panel='b', model=NAME[m], cond='C1', base=g, rate=p_[g], lo=lo, hi=hi, n_units=len(w),
                       n_pairs=int(w[['qid', 'pair_id']].drop_duplicates().shape[0])))
f4 = pd.DataFrame(f4)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(17 * CM, 6.4 * CM), layout='constrained', gridspec_kw={'width_ratios': [1.5, 1]})
xs = np.arange(len(C))
for i, (m, c) in enumerate((('RankZephyr', CL), ('FIRST', CL2))):
    q = f4[(f4.panel == 'a') & (f4.model == m)].set_index('cond').loc[C]
    xx = xs + (i - 0.5) * 0.36
    b1.bar(xx, 100 * q.excess, 0.34, color=c, label=m)
    b1.errorbar(xx, 100 * q.excess, yerr=[100 * (q.excess - q.excess_lo), 100 * (q.excess_hi - q.excess)], **ERR)
    for x_, (_, r) in zip(xx, q.iterrows()):
        if r.sig_bonferroni:
            b1.text(x_, 100 * max(r.excess_hi, 0) + 0.3, '*', ha='center', va='bottom', fontsize=9, color=INK)
b1.axhline(0, color=INK, lw=0.7)
b1.set_ylim(top=b1.get_ylim()[1] * 1.12)
b1.set_xticks(xs, C)
b1.set_ylabel('controlled flips − C0 (%p)')
b1.set_title('(a) excess controlled flips over C0')
b1.legend(fontsize=7, loc='upper left')
b1.grid(axis='y', color=GRID, lw=0.5)
xs2 = np.arange(2)
for i, (m, c) in enumerate((('RankZephyr', CL), ('FIRST', CL2))):
    q = f4[(f4.panel == 'b') & (f4.model == m)].set_index('base').loc[['base wrong', 'base correct']]
    xx = xs2 + (i - 0.5) * 0.36
    b2.bar(xx, 100 * q.rate, 0.34, color=c, label=m)
    b2.errorbar(xx, 100 * q.rate, yerr=[100 * (q.rate - q.lo), 100 * (q.hi - q.rate)], **ERR)
    for x_, (_, r) in zip(xx, q.iterrows()):
        b2.text(x_, 100 * r.hi + 0.6, f'n={r.n_pairs}', ha='center', va='bottom', fontsize=6, color=INK2)
b2.set_xticks(xs2, ['base call wrong', 'base call correct'])
b2.set_ylabel('controlled flip rate in C1 (%)')
b2.set_title('(b) C1, by base-call correctness')
b2.legend(fontsize=7, loc='upper right')
b2.grid(axis='y', color=GRID, lw=0.5)
fig.savefig(f'{P}/figures/fig4_controlled.pdf')
plt.close(fig)
f4.to_csv(f'{P}/figures/fig4_values.csv', index=False)
n_exp2 = len(Q4)
CAPN.append(dict(item='Fig. 4', n_queries=n_exp2,
                 n_pairs=int(U4['zephyr'][['qid', 'pair_id']].drop_duplicates().shape[0])))

# ---- Fig 5 (pair-level flips vs query-level disagreement)
fig, ax = plt.subplots(figsize=(10 * CM, 7.5 * CM), layout='constrained')
xs = np.arange(len(F5BINS))
f5m = F5[F5.scope == MAIN]
for i, (s_, c) in enumerate((('PW4', CP), ('MIX4', CL))):
    q = f5m[f5m.set == s_].set_index('bin').loc[F5BINS]
    ok = q.units_notie > 0
    xx = xs + (i - 0.5) * 0.3
    ax.errorbar(xx[ok], q.change_rate[ok], yerr=[(q.change_rate - q.lo)[ok], (q.hi - q.change_rate)[ok]], color=c,
                ecolor=c, marker='o', ms=5, ls='-', lw=1, elinewidth=0.8, capsize=1.5, label=s_)
nu = {bl: int(f5m[(f5m.set == 'MIX4') & (f5m.bin == bl)].units_in_bin.iloc[0]) for bl in F5BINS}
ax.set_xticks(xs, [f'{bl}\n({nu[bl]} units)' for bl in F5BINS])
ax.set_xlabel('mean listwise replacement flip rate\n(relevant–nonrelevant, RankZephyr and FIRST)')
ax.set_ylabel('top-1 reranker changes between retrievers')
ax.set_title(f'{ENG[MAIN]}, same relevant set')
ax.set_ylim(0, None)
ax.grid(axis='y', color=GRID, lw=0.5)
ax.legend(fontsize=7, loc='upper left')
fig.savefig(f'{P}/figures/fig5_flip_vs_top1.pdf')
plt.close(fig)
f5m.to_csv(f'{P}/figures/fig5_values.csv', index=False)
# pre-specified placement rule: main text iff Spearman(bin order, MIX4 change rate) > 0 and the 95% CI lower
# bound of (rate in "> 0.25") − (rate in "0") is > 0; otherwise appendix
_m = f5m[(f5m.set == 'MIX4') & (f5m.units_notie > 0)].set_index('bin').reindex(F5BINS)
_ok = _m.change_rate.notna().to_numpy()
F5RHO = float(stats.spearmanr(np.arange(len(F5BINS))[_ok], _m.change_rate.to_numpy()[_ok]).statistic) if _ok.sum() > 1 else np.nan
if (MAIN, 'MIX4', '> 0.25') in F5B and (MAIN, 'MIX4', '0') in F5B:
    _d = F5B[(MAIN, 'MIX4', '> 0.25')] - F5B[(MAIN, 'MIX4', '0')]
    F5DIFF = (float(_m.change_rate['> 0.25'] - _m.change_rate['0']), *ci(_d))
else:
    F5DIFF = (np.nan, np.nan, np.nan)
F5PLACE = 'main text' if (F5RHO > 0 and F5DIFF[1] > 0) else 'appendix'
pd.DataFrame([dict(spearman_bin_vs_mix4=F5RHO, diff_gt025_minus_0=F5DIFF[0], diff_lo=F5DIFF[1], diff_hi=F5DIFF[2],
                   placement=F5PLACE)]).to_csv(f'{P}/figures/fig5_placement.csv', index=False)
CAPN += [dict(item=f'Fig. 5 {r.set} {r.bin}', n_queries=int(r.n_queries), n_units=int(r.units_notie))
         for _, r in f5m.iterrows()]
log('figures done')

# ================================================================ §8 consistency checks
CHK = []
CAPN += [dict(item='Table 1', n_queries=len(EVAL)), dict(item='Table 3', n_queries=len(EVAL))]
CAPN += [dict(item=f'Table 2 {r.reranker} {r.column}', n_queries=int(r.n_queries), n_pairs=int(r.n_pairs))
         for _, r in t2.iterrows()]
CAPN += [dict(item=f'Appendix C {sc}', n_queries=len(k)) for sc, k in GROUPS]
pd.DataFrame(CAPN).to_csv(f'{P}/tables/caption_counts.csv', index=False)
# 7.4 model identity (experiment 1 vs experiment 2 code)
s1 = open(f'{REPO}/experiments/cross_retriever/listwise/run_listwise.py').read()
s2 = open(f'{REPO}/experiments/controlled_context/run_listwise.py').read()
FEAT = {'checkpoint RankZephyr': 'castorini/rank_zephyr_7b_v1_full', 'checkpoint FIRST': 'castorini/first_mistral',
        'RankZephyr prompt (rank_llm, context 4096)': 'context_size=4096',
        'FIRST prompt (alpha template)': 'rank_zephyr_alpha_template.yaml',
        'FIRST system message': 'You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.',
        'FIRST logits': 'use_alpha=True, use_logits=True', 'FIRST variable passages': 'variable_passages=True',
        'FIRST "[" prefix': '+ "[" if first', 'FIRST sampling (1 token, logprobs 30)': 'min_tokens=1, max_tokens=1, logprobs=30',
        'temperature 0': 'temperature=0.0', 'vLLM enforce_eager False / max_logprobs 30': 'enforce_eager=False, max_logprobs=30',
        'vLLM seed': 'seed=SEED', 'max_model_len from rank_llm': '_get_max_model_len'}
for k, pat in FEAT.items():
    a, b = pat.replace('"', "'") in s1 or pat in s1, pat in s2 or pat.replace('"', "'") in s2
    CHK.append(dict(check='7.4 model identity', item=k, exp1=a, exp2=b, ok=a and b))
CHK.append(dict(check='7.4 model identity', item='vLLM version (both .venv_llm)', exp1='0.26.0', exp2='0.26.0', ok=True))
CHK.append(dict(check='7.4 model identity', item='window (design difference, reported)', exp1='20 / stride 10 over 100',
                exp2='single window of 20', ok=True))
# 7.5 experiment-2 queries
dlq = {k for k in EVAL if k[0] in DLSET}
CHK.append(dict(check='7.5 experiment-2 queries', item='DL19+DL20 evaluation queries = experiment-2 queries',
                exp1=len(dlq), exp2=len(P2Q), ok=dlq == set(P2Q)))
CHK.append(dict(check='8.1 evaluation set', item='E ∪ F, all six rerankers present', exp1=len(EVAL),
                exp2=len(E) + len(F), ok=len(EVAL) == len(E) + len(F) and not set(E) & set(F)))
for sc, keys in GROUPS[:2]:
    nn = sum(len(k) for d, k in GROUPS[2:] if BUNDLE(d) == sc)
    CHK.append(dict(check='8.1 evaluation set', item=f'bundle {sc} = sum of its datasets', exp1=len(keys), exp2=nn,
                    ok=len(keys) == nn))


def cmp(item, a, b, tol=1e-9):
    CHK.append(dict(check='8.3 cross-check', item=item, exp1=a, exp2=b, ok=bool(abs(a - b) <= tol)))


f2 = pd.read_csv(f'{P}/figures/fig2_values.csv')
for _, r in f2[f2.panel == 'b'].iterrows():
    ap = KR[(KR.scope == r.scope) & (KR.subset == r.subset) & (KR.pair == r.pair) & (KR.set == r.set)].iloc[0]
    cmp(f'Fig2b vs appendix A κ {r.subset} {r.pair} {r.set}', r.kappa, ap.kappa)
for _, r in pd.read_csv(f'{P}/figures/fig3_values.csv').iterrows():
    cmp(f'Fig3 vs sign table {r.subset} {r.pair}', r.agree,
        SG[(SG.scope == r.scope) & (SG.subset == r.subset) & (SG.pair == r.pair)].agree.iloc[0])
for _, r in t2.iterrows():
    if r.column.startswith('동일'):
        ref = SI[(SI.reranker == r.reranker) & (SI.pair_type == 'rel-nonrel')].flip_rate.iloc[0]
        cmp(f'Table2 vs appendix D {r.reranker}', r.rate, ref)
for m in LW:  # recomputed Fig 4 point values equal the experiment-2 table
    for c in ['R', 'C0'] + C:
        a = f4[(f4.panel == 'a') & (f4.model == NAME[m]) & (f4.cond == c)].iloc[0]
        b = m1[(m1.model == m) & (m1.cond == c)].iloc[0]
        cmp(f'Fig4a vs experiment-2 table {NAME[m]} {c} rate', a.rate, b.reversal)
        cmp(f'Fig4a vs experiment-2 table {NAME[m]} {c} excess', a.excess, b.excess_vs_C0)
for _, r in T1.iterrows():
    inv = {v: k for k, v in NAME.items()}
    src = g0 if inv[r.reranker] in PW else gm
    kk = pd.MultiIndex.from_tuples(EVAL, names=['dataset', 'qid'])
    ref = src[(src.retriever == r.retriever) & (src.reranker == inv[r.reranker])].set_index(['dataset', 'qid']).ndcg10
    if inv[r.reranker] in PW or not HAVE_B:
        cmp(f'Table1 nDCG vs stored grid {r.retriever} {r.reranker}', r.ndcg10, ref.reindex(kk).mean(), 1e-9)
for _, r in T3.iterrows():  # mean of the per-pair values for sig and tau equals the pooled value over all units
    cmp(f'Table3 {r.set} sig mean = mean of pairs', r.sig_mean, np.mean([r[f'sig_{PL(p)}'] for p in PAIRS]))
pw0 = FL[(FL.kind == 'P')].flips.sum()
CHK.append(dict(check='8.3 cross-check', item='pointwise replacement flips (must be 0 by construction)', exp1=int(pw0),
                exp2=0, ok=pw0 == 0))
CHK = pd.DataFrame(CHK)
CHK.to_csv(f'{P}/tables/consistency_checks.csv', index=False)
log(f'consistency checks: {len(CHK)} items, {int((~CHK.ok.astype(bool)).sum())} failed')

# ================================================================ judgment-gap check (6.A.4)
JG = T1.groupby('reranker', sort=False).judged10.agg(['mean', 'min', 'max']).reset_index()
JK = AK[(AK.metric == 'nDCG_judged') & (AK.th == TH)]
save(JG, 'judgment_gap_judged10')

# ================================================================ hypotheses (B queries only)
HYP = []
if HAVE_B:
    kf = HK[(HK.subset == 'same_rel') & (HK.set == 'PW4−MIX4')]
    n_ok = int((kf.lo > 0).sum())
    HYP.append(('H1', n_ok >= 2, '; '.join(f'{r.pair} {r.kappa:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]' for _, r in kf.iterrows())
                + f' → CI lower bound > 0: {n_ok}/3'))
    h2 = HS[HS.group == 'PP−PL'].iloc[0]
    HYP.append(('H2', h2.lo > 0, f'PP − PL {h2.agree:+.3f} [{h2.lo:+.3f}, {h2.hi:+.3f}] (all units, original definition)'))
    for m in ('RankZephyr', 'FIRST'):
        h3 = HF[(HF.reranker == m) & (HF.pair_type == 'rel-nonrel') & (HF.retriever_pair == 'pooled')].iloc[0]
        HYP.append((f'H3 {m}', h3.lo > 0.05, f'{h3.flip_rate:.3f} [{h3.lo:.3f}, {h3.hi:.3f}]'))

# ================================================================ report
fails = CHK[~CHK.ok.astype(bool)]
# B execution record (n recomputations, lock bypass, FIRST check) for RESULTS.md
BREC = []
if os.path.exists(f'{B}/orchestrate.log'):
    BREC += [l.rstrip() for l in open(f'{B}/orchestrate.log') if any(t in l for t in (
        'n = ', 'recompute', 'FIRST check', 'DEADLINE', 'measured', 'SPEED_NOT', 'F done', 'stopped v5b'))]
for f_ in (f'{B}/n_recompute_v8.json', f'{B}/lock_bypass.jsonl', f'{B}/F/out/v8_first_check.txt'):
    if os.path.exists(f_):
        BREC += [f'{os.path.basename(f_)}: {l.rstrip()}' for l in open(f_)]
md = [f'# {"RESULTS" if HAVE_B else "RESULTS_existing"} — cross-retriever reranker transfer', '',
      f'Generated {time.strftime("%Y-%m-%d %H:%M:%S")}. Evaluation queries {len(EVAL)} (set E {len(E)} + set F {len(F)}); '
      f'queries with runs of all three retrievers {len(ALLQ)}. ' + ' / '.join(LOG[:2]), '']
md += ['## Set F execution record (n recomputation, FIRST check)', '', '```'] + (BREC or ['(none)']) + ['```', '']
if HYP:
    md += [f'## Hypotheses H1–H3 (set F only, {len(F)} queries; not used in figures or tables)', '', '| hypothesis | verdict | evidence |', '|---|---|---|'] + \
          [f'| {h} | {"supported" if ok else "rejected"} | {t} |' for h, ok, t in HYP] + ['']
md += ['## Table 1(a). Datasets', '', qc.round(2).to_markdown(index=False), '',
       '## Table 1(b). Rerankers', '', t1b.round(3).to_markdown(index=False), '',
       '## Fig. 2(a). Candidate overlap', '', cd.round(3).to_markdown(index=False), '',
       '## Fig. 2(b) / Appendix A. κ (ties excluded, threshold 0.01, nDCG@10)', '', KR.round(3).to_markdown(index=False), '',
       SIK.round(3).to_markdown(index=False) if len(SIK) else '', '',
       '## Fig. 3. Sign agreement vs mean effectiveness gap (relevant-set-identical units; all units in the table files)', '',
       q3.round(3).to_markdown(index=False), '',
       '## Table 2. Listwise flips (zero for pointwise rerankers by construction)', '', t2w.to_markdown(index=False), '',
       '## Fig. 4. Controlled flips (Experiment 2, recomputed here: 10,000 bootstrap resamples)', '', f4.round(4).to_markdown(index=False), '',
       f'## Fig. 5. Top-1 change rate by pair-flip bin ({F5_NX} units without a shared relevant–non-relevant pair for either listwise reranker excluded)', '',
       f5m.round(3).to_markdown(index=False), '',
       f'Placement rule: Spearman(bin order, MIX4 change rate) = {F5RHO:.3f}; (>0.25) − (0) = {F5DIFF[0]:.3f} '
       f'[{F5DIFF[1]:.3f}, {F5DIFF[2]:.3f}] → **{F5PLACE}**', '',
       '## Table 3. Agreement of evaluation conclusions across retrievers (significance, retention: mean of three retriever pairs [95% CI], range; τ: three values and range)', '',
       t3.to_markdown(index=False), '', f'Note: {T3NOTE}', '',
       'Mean nDCG@10 gaps between the models of each set (Table 1(b) values)', '', GAP3.round(4).to_markdown(index=False), '',
       T3.round(3).to_markdown(index=False), '',
       '## Judgment-gap check', '', 'Share of judged documents in the top 10 (Judged@10, mean and range over the three retrievers):', '', JG.round(3).to_markdown(index=False), '',
       'κ recomputed with nDCG@10 on judged documents only (threshold 0.01):', '', JK.round(3).to_markdown(index=False), '',
       '## Construction and consistency checks', '', f'{len(CHK)} checks, {len(fails)} failed.', '',
       (fails.to_markdown(index=False) if len(fails) else 'No failed checks.'), '',
       CHK[CHK.check != '8.3 cross-check'].to_markdown(index=False), '',
       '### Query and pair counts', '', pd.DataFrame(CAPN).drop_duplicates().to_markdown(index=False), '',
       '## Appendix B. Sensitivity — sign agreement by pair type', '', AS.round(3).to_markdown(index=False), '',
       '## Appendix B. Sensitivity — κ', '', AK.round(3).to_markdown(index=False), '',
       '## Appendix C. By collection and dataset', '', '### κ', '', GK.round(3).to_markdown(index=False), '',
       '### Sign agreement (pooled pair types)', '', GS[GS.pair.str.startswith('[')].round(3).to_markdown(index=False), '',
       '### Replacement flips (listwise, retriever pairs pooled)', '', GF.round(4).to_markdown(index=False), '',
       '### Table 3', '', GT3.round(3).to_markdown(index=False), '', '### Candidate overlap', '', GCD.round(3).to_markdown(index=False), '',
       '## Appendix D. Same-input reruns', '', SI.round(4).to_markdown(index=False) if len(SI) else '(none)', '',
       f'(Per-dataset κ threshold sensitivity: tables/appendixB_kappa_by_dataset_threshold.csv, {len(ADS)} rows)', '',
       f'Total time {time.time() - T0:.0f} s']
open(f'{P}/{"RESULTS" if HAVE_B else "RESULTS_existing"}.md', 'w').write('\n'.join(md))
log('report written')
