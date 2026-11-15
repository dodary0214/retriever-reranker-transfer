"""Four-reranker sets and k groups (definitions fixed before computing). All 15 four-reranker subsets of the six rerankers, grouped by the number of listwise rerankers k (0, 1, 2).
CPU only, existing scores only. Executes analyze.py's source up to '# ... run everything' (definitions, data loading
and the same-input reruns; writes no files), with two textual changes: table3() also returns its bootstrap arrays, and
fig5_table() loops over all 15 sets. Writes $RRT_OUT/derived/ksets_*.csv.
Stage 2 (margin) runs only if stage 1 is 'keep' or 'partial'."""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from common import OUT as OUT_ROOT, QWEN, REPO  # noqa: E402
AN = os.path.join(REPO, 'analysis', 'analyze.py')
OUT = os.path.join(OUT_ROOT, 'derived')
os.makedirs(OUT, exist_ok=True)
src = open(AN).read()
pre = src[:src.index('# ================================================================ run everything')]
a = "        out.update({f'{m}_mean': vals.mean(),"
assert pre.count(a) == 1
pre = pre.replace(a, "        out[f'{m}_boot'] = bm\n" + a)
b = "    for s_ in ('MIX4', 'PW4'):"
assert pre.count(b) == 1
pre = pre.replace(b, '    for s_ in F5SETS:')
g = {'__file__': AN, '__name__': 'analyze_prefix'}
exec(compile(pre, AN, 'exec'), g)
EVAL, E, PAIRS, SETS, PL, ALL6, LW, NAME = (g[x] for x in ('EVAL', 'E', 'PAIRS', 'SETS', 'PL', 'ALL6', 'LW', 'NAME'))
X, codes, kappa_w, BW, ci, same_rel, TH, RETRIEVERS = (g[x] for x in ('X', 'codes', 'kappa_w', 'BW', 'ci', 'same_rel',
                                                                      'TH', 'RETRIEVERS'))
RERUN, GR, ndcg10, DSETS = g['RERUN'], g['GR'], g['ndcg10'], g['DSETS']
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)

# ---- the 15 sets
COMBOS = [list(c) for c in itertools.combinations(ALL6, 4)]
SNAME = {}
for c in COMBOS:
    nm = '+'.join(NAME[x] for x in c)
    SETS[nm] = c
    SNAME[nm] = sum(x in LW for x in c)
assert len(SNAME) == 15
PW4N = '+'.join(NAME[x] for x in SETS['PW4'])
MIX4N = [nm for nm in SNAME if set(SETS[nm]) == set(SETS['MIX4'])][0]
GRP = {k: [nm for nm in SNAME if SNAME[nm] == k] for k in (0, 1, 2)}
assert GRP[0] == [PW4N] and len(GRP[1]) == 8 and len(GRP[2]) == 6 and MIX4N in GRP[2]
print({k: len(v) for k, v in GRP.items()}, 'MIX4 =', MIX4N)


def kpool(keys, rr, sub, metric='nDCG@10', th=TH):
    """kappa over (query, retriever pair) units pooled over the three retriever pairs"""
    SR = same_rel(keys)
    n = len(keys)
    ca = np.concatenate([codes(metric, keys, rr, p[0], th) for p in PAIRS])
    cb = np.concatenate([codes(metric, keys, rr, p[1], th) for p in PAIRS])
    mk = np.concatenate([SR[p] if sub == 'same_rel' else np.ones(n, bool) for p in PAIRS])
    k, nn, _, _ = kappa_w(ca, cb, mk, len(rr), np.ones((1, 3 * n), np.float32))
    kb = kappa_w(ca, cb, mk, len(rr), np.tile(BW(keys), (1, 3)))[0]
    return float(k[0]), int(nn[0]), int(mk.sum()), kb


def gstat(vals, boots):
    v = float(np.mean(vals))
    b = np.nanmean(np.vstack(boots), 0)
    return v, *ci(b), b


def same_input_kappa(rr):
    qs = sorted({(d, q) for mm, d, r, q in RERUN if mm == 'zephyr'}, key=lambda k: (DSETS.index(k[0]), k[1]))
    ca_all, cb_all = [], []
    for r in RETRIEVERS:
        x = X('nDCG@10', qs, rr, r)
        y = x.copy()
        if 'zephyr' in rr:
            y[:, rr.index('zephyr')] = [ndcg10(RERUN[('zephyr', d, r, q)], GR[d].get(q, {})) for d, q in qs]
        for z, o in ((x, ca_all), (y, cb_all)):
            s = np.sort(z, 1)
            c = np.argmax(z, 1)
            c[(s[:, -1] - s[:, -2]) < TH] = len(rr)
            o.append(c)
    ca, cb = np.concatenate(ca_all), np.concatenate(cb_all)
    k, nn, _, _ = kappa_w(ca, cb, np.ones(len(ca), bool), len(rr), np.ones((1, len(ca)), np.float32))
    kb = kappa_w(ca, cb, np.ones(len(ca), bool), len(rr), np.tile(BW(qs), (1, 3)))[0]
    return float(k[0]), int(nn[0]), len(qs), *ci(kb)


# ================================================================ stage 1
def stage1(keys, scope, full=True):
    rows, boots = [], {}
    for nm, k in SNAME.items():
        rr = SETS[nm]
        r = dict(scope=scope, set=nm, k=k, members=','.join(rr))
        r['mean_ndcg10'] = float(np.mean([X('nDCG@10', keys, rr, rt).mean() for rt in RETRIEVERS]))
        for sub in ('all', 'same_rel'):
            kv, nn, nu, kb = kpool(keys, rr, sub)
            r.update({f'kappa_{sub}': kv, f'n_notie_{sub}': nn, f'units_{sub}': nu,
                      f'lo_{sub}': ci(kb)[0], f'hi_{sub}': ci(kb)[1]})
            boots[(nm, sub)] = kb
        rows.append(r)
    S = pd.DataFrame(rows)
    G, D = [], []
    gb = {}
    for sub in ('all', 'same_rel'):
        for k, mem in GRP.items():
            v, lo, hi, b = gstat(S.set_index('set').loc[mem, f'kappa_{sub}'], [boots[(m, sub)] for m in mem])
            gb[(k, sub)] = b
            G.append(dict(scope=scope, subset=sub, k=k, n_sets=len(mem), kappa=v, lo=lo, hi=hi,
                          set_min=S.set_index('set').loc[mem, f'kappa_{sub}'].min(),
                          set_max=S.set_index('set').loc[mem, f'kappa_{sub}'].max()))
        for a_, b_ in ((0, 1), (0, 2), (1, 2)):
            gv = {x['k']: x['kappa'] for x in G if x['subset'] == sub}
            d = gb[(a_, sub)] - gb[(b_, sub)]
            D.append(dict(scope=scope, subset=sub, diff=f'k{a_}-k{b_}', value=gv[a_] - gv[b_], lo=ci(d)[0], hi=ci(d)[1]))
    return S, pd.DataFrame(G), pd.DataFrame(D), gb


S1, G1, D1, GB1 = stage1(EVAL, 'evaluation')
# same-input kappa (k >= 1 sets)
si = {nm: same_input_kappa(SETS[nm]) for nm in SNAME if SNAME[nm] >= 1}
S1['same_input_kappa'] = S1.set.map(lambda nm: si[nm][0] if nm in si else np.nan)
S1['same_input_lo'] = S1.set.map(lambda nm: si[nm][3] if nm in si else np.nan)
S1['same_input_hi'] = S1.set.map(lambda nm: si[nm][4] if nm in si else np.nan)
S1['same_input_n_notie'] = S1.set.map(lambda nm: si[nm][1] if nm in si else np.nan)
S1['has_zephyr'] = S1.members.str.contains('zephyr')
# per retriever pair kappa (all 15 sets), for Appendix A
kt = pd.DataFrame(g['kappa_table'](EVAL, 'evaluation', sets=tuple(SNAME)))
kt['k'] = kt.set.map(SNAME)
kt.to_csv(f'{OUT}/ksets_kappa_by_pair.csv', index=False)
# Spearman within k groups (set mean nDCG@10 vs pooled relevant-set-identical kappa)
SP = []
for k in (1, 2):
    z = S1[S1.k == k]
    rho = stats.spearmanr(z.mean_ndcg10, z.kappa_same_rel).statistic
    SP.append(dict(k=k, n_sets=len(z), spearman=float(rho), confound_flag=bool(abs(rho) >= 0.5)))
SP = pd.DataFrame(SP)
# rank of MIX4 among the six k=2 sets (1 = highest kappa)
z2 = S1[S1.k == 2].sort_values('kappa_same_rel', ascending=False).reset_index(drop=True)
mix_rank = int(z2.index[z2.set == MIX4N][0]) + 1

# ---- Table 3 quantities, 15 sets and k groups
T3, T3B = [], {}
for nm in SNAME:
    o, _ = g['table3'](EVAL, 'evaluation', nm)
    T3B[nm] = {m: o.pop(f'{m}_boot') for m in ('sig', 'tau', 'retention')}
    o['k'] = SNAME[nm]
    T3.append(o)
T3 = pd.DataFrame(T3)
T3G = []
for k, mem in GRP.items():
    z = T3.set_index('set').loc[mem]
    r = dict(k=k, n_sets=len(mem))
    for m in ('sig', 'retention'):
        v, lo, hi, _ = gstat(z[f'{m}_mean'], [T3B[x][m] for x in mem])
        r.update({f'{m}': v, f'{m}_lo': lo, f'{m}_hi': hi, f'{m}_set_min': z[f'{m}_mean'].min(),
                  f'{m}_set_max': z[f'{m}_mean'].max()})
    r.update(tau=z.tau_mean.mean(), tau_set_min=z.tau_mean.min(), tau_set_max=z.tau_mean.max(),
             tau_pair_min=z.tau_min.min(), tau_pair_max=z.tau_max.max())
    T3G.append(r)
T3G = pd.DataFrame(T3G)
T3D = []
for m in ('sig', 'retention'):
    gbm = {k: np.nanmean(np.vstack([T3B[x][m] for x in mem]), 0) for k, mem in GRP.items()}
    for a_, b_ in ((0, 1), (0, 2), (1, 2)):
        v = T3G.set_index('k').loc[a_, m] - T3G.set_index('k').loc[b_, m]
        T3D.append(dict(metric=m, diff=f'k{a_}-k{b_}', value=v, lo=ci(gbm[a_] - gbm[b_])[0], hi=ci(gbm[a_] - gbm[b_])[1]))
T3D = pd.DataFrame(T3D)

# ---- Fig. 5 by k group
g['F5SETS'] = list(SNAME)
F5rows, _ = g['fig5_table'](EVAL, 'evaluation')
F5 = pd.DataFrame(F5rows)
F5B = g['F5B']
F5G = []
for k, mem in GRP.items():
    for bl in g['F5BINS']:
        z = F5[(F5.set.isin(mem)) & (F5.bin == bl)]
        bb = [F5B[('evaluation', m, bl)] for m in mem if ('evaluation', m, bl) in F5B]
        if not bb:
            continue
        v, lo, hi, b = gstat(z.change_rate, bb)
        F5G.append(dict(k=k, bin=bl, change_rate=v, lo=lo, hi=hi, units_in_bin=int(z.units_in_bin.iloc[0]),
                        units_notie_mean=float(z.units_notie.mean()), _b=b))
F5G = pd.DataFrame(F5G)
z2f = F5G[F5G.k == 2].set_index('bin').reindex(g['F5BINS'])
ok = z2f.change_rate.notna().to_numpy()
F5RHO = float(stats.spearmanr(np.arange(4)[ok], z2f.change_rate.to_numpy()[ok]).statistic)
dd = z2f.loc['> 0.25', '_b'] - z2f.loc['0', '_b']
F5DIFF = (float(z2f.loc['> 0.25', 'change_rate'] - z2f.loc['0', 'change_rate']), *ci(dd))
F5PLACE = 'main text' if (F5RHO > 0 and F5DIFF[1] > 0) else 'appendix'
F5G = F5G.drop(columns='_b')

# ---- 547-query direction check (pooled kappa groups and Table 3 groups)
S5, G5, D5, _ = stage1(E, 'first547')
T35 = []
for nm in SNAME:
    o, _ = g['table3'](E, 'first547', nm)
    T35.append(dict(set=nm, k=SNAME[nm], sig=o['sig_mean'], tau=o['tau_mean'], retention=o['retention_mean']))
T35 = pd.DataFrame(T35).groupby('k')[['sig', 'tau', 'retention']].mean().reset_index()

for nm_, df in (('sets', S1), ('groups', pd.concat([G1, G5])), ('group_diffs', pd.concat([D1, D5])), ('spearman', SP),
                ('table3_sets', T3), ('table3_groups', T3G), ('table3_group_diffs', T3D), ('fig5_sets', F5),
                ('fig5_groups', F5G), ('table3_groups_547', T35), ('sets_547', S5)):
    df.to_csv(f'{OUT}/ksets_{nm_}.csv', index=False)
pd.DataFrame([dict(spearman_bin_vs_k2=F5RHO, diff_gt025_minus_0=F5DIFF[0], lo=F5DIFF[1], hi=F5DIFF[2],
                   placement=F5PLACE)]).to_csv(f'{OUT}/ksets_fig5_placement.csv', index=False)

print('\n=== stage 1: sets (evaluation, pooled kappa)')
print(S1[['set', 'k', 'mean_ndcg10', 'kappa_all', 'kappa_same_rel', 'lo_same_rel', 'hi_same_rel', 'n_notie_same_rel',
          'same_input_kappa']].round(3).to_string())
print(G1.round(3).to_string())
print(D1.round(3).to_string())
print(SP.round(3).to_string())
print('MIX4 rank among k=2 sets:', mix_rank, 'of 6')
print(T3G.round(3).to_string())
print(T3D.round(3).to_string())
print(F5G.round(3).to_string())
print('fig5 k2 rule: rho', round(F5RHO, 3), 'diff', [round(x, 3) for x in F5DIFF], '->', F5PLACE)
print('547:', G5[G5.subset == 'same_rel'].round(3).to_string(), D5.round(3).to_string(), T35.round(3).to_string(), sep='\n')

gv = G1[G1.subset == 'same_rel'].set_index('k').kappa
d02 = D1[(D1.subset == 'same_rel') & (D1['diff'] == 'k0-k2')].iloc[0]
mono = gv[0] > gv[1] > gv[2]
V1 = 'reject' if d02.lo <= 0 else ('keep' if mono else 'partial')
print(f'STAGE1 VERDICT {V1} (monotone {mono}; k0-k2 {d02.value:.3f} [{d02.lo:.3f}, {d02.hi:.3f}]); '
      f'confound flags {SP.confound_flag.tolist()}')
pd.DataFrame([dict(stage=1, verdict=V1, monotone=mono, k0_k2=d02.value, lo=d02.lo, hi=d02.hi, mix4_rank_k2=mix_rank,
                   confound_k1=SP.confound_flag[0], confound_k2=SP.confound_flag[1])]).to_csv(
    f'{OUT}/ksets_verdict_stage1.csv', index=False)
if V1 == 'reject':
    sys.exit(0)

# ================================================================ stage 2: margin
n = len(EVAL)
SR = same_rel(EVAL)
W = BW(EVAL).astype(np.float64)
EDGES = [0.01, 0.05, 0.1, np.inf]


def units(rr):
    qi, mg, co, cx = [], [], [], []
    for p in PAIRS:
        idx = np.flatnonzero(SR[p])
        for a_, b_ in ((p[0], p[1]), (p[1], p[0])):
            x = np.sort(X('nDCG@10', EVAL, rr, a_), 1)
            ca, cb = codes('nDCG@10', EVAL, rr, a_, TH), codes('nDCG@10', EVAL, rr, b_, TH)
            qi.append(idx); mg.append((x[:, -1] - x[:, -2])[idx]); co.append(ca[idx]); cx.append(cb[idx])
    return np.concatenate(qi), np.concatenate(mg), np.concatenate(co), np.concatenate(cx)


U = {nm: units(SETS[nm]) for nm in GRP[0] + GRP[2]}


def kbin(nm, lo, hi, boot=True):
    qi, mg, co, cx = U[nm]
    K = 4
    ok = (co < K) & (cx < K) & (mg >= lo) & (mg < hi)
    q, a_, b_ = qi[ok], co[ok], cx[ok]

    def stat(w):
        Aq = np.zeros((n, K)); Bq = np.zeros((n, K)); Eq = np.zeros(n); Nq = np.zeros(n)
        np.add.at(Aq, (q, a_), 1); np.add.at(Bq, (q, b_), 1); np.add.at(Eq, q, (a_ == b_).astype(float)); np.add.at(Nq, q, 1)
        N = w @ Nq
        with np.errstate(invalid='ignore', divide='ignore'):
            po = (w @ Eq) / N
            pe = (((w @ Aq) / N[:, None]) * ((w @ Bq) / N[:, None])).sum(1)
            return (po - pe) / (1 - pe)
    return float(stat(np.ones((1, n)))[0]), int(ok.sum()), (stat(W) if boot else None)


bins = [(EDGES[i], EDGES[i + 1]) for i in range(3)]
merges = []
while len(bins) > 1:
    ns = [min(kbin(nm, lo, hi, False)[1] for nm in U) for lo, hi in bins]
    bad = [i for i, v in enumerate(ns) if v < 30]
    if not bad:
        break
    i = bad[0]
    j = i - 1 if i == len(bins) - 1 else i + 1
    a_, b_ = sorted((i, j))
    merges.append(f'{bins[a_]} + {bins[b_]} (min n {ns[i]})')
    bins = bins[:a_] + [(bins[a_][0], bins[b_][1])] + bins[b_ + 1:]
MR = []
for lo, hi in bins + [(0.01, np.inf)]:
    k0v, k0n, k0b = kbin(GRP[0][0], lo, hi)
    res = [kbin(nm, lo, hi) for nm in GRP[2]]
    k2v = float(np.mean([r[0] for r in res]))
    k2b = np.nanmean(np.vstack([r[2] for r in res]), 0)
    d = k0b - k2b
    MR.append(dict(bin=f'[{lo}, {hi})' if np.isfinite(hi) else f'>= {lo}', kappa_k0=k0v, n_k0=k0n,
                   k0_lo=ci(k0b)[0], k0_hi=ci(k0b)[1], kappa_k2=k2v, k2_lo=ci(k2b)[0], k2_hi=ci(k2b)[1],
                   n_k2_mean=float(np.mean([r[1] for r in res])), n_k2_min=min(r[1] for r in res),
                   n_k2_max=max(r[1] for r in res), kappa_k2_set_min=min(r[0] for r in res),
                   kappa_k2_set_max=max(r[0] for r in res), diff=k0v - k2v, diff_lo=ci(d)[0], diff_hi=ci(d)[1],
                   is_reference_all=not np.isfinite(hi) and lo == 0.01 and len(bins) > 0 and (lo, hi) not in bins))
MR = pd.DataFrame(MR)
MR['merges'] = '; '.join(merges) if merges else 'none'
MD = []
for lab, mem in (('k0', GRP[0]), ('k2', GRP[2])):
    mg = np.concatenate([U[nm][1] for nm in mem])
    for sub, v in (('all same_rel units', mg), ('margin >= 0.01', mg[mg >= TH])):
        MD.append(dict(group=lab, subset=sub, n=len(v), q1=np.percentile(v, 25), median=np.median(v),
                       q3=np.percentile(v, 75), share_lt_001=float((mg < TH).mean())))
MD = pd.DataFrame(MD)
MR.to_csv(f'{OUT}/ksets_margin_kappa.csv', index=False)
MD.to_csv(f'{OUT}/ksets_margin_dist.csv', index=False)
print('\n=== stage 2: margin; merges:', merges or 'none')
print(MR.round(3).to_string())
print(MD.round(4).to_string())
body = MR[~MR.is_reference_all]
allpos = bool((body['diff'] > 0).all())
big = body.loc[(body.n_k0 + body.n_k2_mean).idxmax()]
V2 = 'pass' if allpos and big.diff_lo > 0 else 'fail'
print(f'STAGE2 VERDICT {V2} (all diffs > 0 {allpos}; largest-n bin {big.bin}: {big["diff"]:.3f} '
      f'[{big.diff_lo:.3f}, {big.diff_hi:.3f}])')
pd.DataFrame([dict(stage=2, verdict=V2, all_pos=allpos, big_bin=big.bin, diff=big['diff'], lo=big.diff_lo,
                   hi=big.diff_hi)]).to_csv(f'{OUT}/ksets_verdict_stage2.csv', index=False)
