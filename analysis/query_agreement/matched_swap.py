"""Matched set-swap analysis: does replacing one pointwise reranker of a four-reranker set by a listwise reranker
lower the cross-retriever agreement (Cohen's kappa) of the per-query top-1 reranker, on relevant-set-identical units?

Definitions are taken unchanged from analysis/analyze.py, executed read-only up to '# ... run everything'
(as analysis/query_agreement/ksets.py does): evaluation queries (EVAL, 1,289), nDCG@10 per (query, retriever, reranker),
top-1 codes with tie threshold TH = 0.01 (a unit whose best two rerankers differ by < TH counts as a tie and is excluded),
relevant-set-identical mask same_rel(), units = (query, retriever pair) pooled over the three retriever pairs, and the
query bootstrap BW(EVAL) (10,000 resamples, numpy default_rng(0)), tiled over the three retriever pairs so that every set
and every swap uses the same resampled queries (dependence between sets preserved).

Swaps (A -> B share three rerankers; B = A with one pointwise reranker replaced by a listwise one):
  k0->1: 4 removed pointwise x 2 added listwise = 8 swaps;  k1->2: 24 swaps (every k=2 set from every k=1 set it contains).
Delta = kappa(A) - kappa(B) (positive = agreement drops when the listwise reranker comes in).
Unit definitions:
  own    : each set's kappa on its own untied units (the definition used for the reported kappa values)
  common : both sets on the same units, those untied for A and for B under both retrievers
Subsets: same_rel (main) and all (all evaluation units, for contrast).
Writes $RRT_OUT/derived/matched_swap_*.csv; the comparison table reads the reference values in
results/summary/derived/ksets_{sets,groups,group_diffs}.csv."""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from common import OUT as OUT_ROOT, REPO  # noqa: E402
AN = os.path.join(REPO, 'analysis', 'analyze.py')
OUT = os.path.join(OUT_ROOT, 'derived')
os.makedirs(OUT, exist_ok=True)
src = open(AN).read()
pre = src[:src.index('# ================================================================ run everything')]
g = {'__file__': AN, '__name__': 'analyze_prefix'}
exec(compile(pre, AN, 'exec'), g)
EVAL, PAIRS, PW, LW, NAME = g['EVAL'], g['PAIRS'], g['PW'], g['LW'], g['NAME']
codes, kappa_w, BW, ci, same_rel, TH = g['codes'], g['kappa_w'], g['BW'], g['ci'], g['same_rel'], g['TH']
assert len(EVAL) == 1289 and TH == 0.01 and g['NB'] == 10000 and g['SEED'] == 0

n = len(EVAL)
SR = same_rel(EVAL)
WT = np.tile(BW(EVAL), (1, 3)).astype(np.float64)   # (10000, 3n): query resamples shared by all sets
ONE = np.ones((1, 3 * n))
MASK = {'same_rel': np.concatenate([SR[p] for p in PAIRS]), 'all': np.ones(3 * n, bool)}
ALL6 = PW + LW
SETS = [tuple(c) for c in itertools.combinations(ALL6, 4)]
K = lambda s: sum(x in LW for x in s)
sname = lambda s: '+'.join(NAME[x] for x in s)

CODE = {}
for s in SETS:
    rr = list(s)
    CODE[s] = (np.concatenate([codes('nDCG@10', EVAL, rr, p[0], TH) for p in PAIRS]),
               np.concatenate([codes('nDCG@10', EVAL, rr, p[1], TH) for p in PAIRS]))
untied = {s: (CODE[s][0] < 4) & (CODE[s][1] < 4) for s in SETS}


def kap(s, mask):
    ca, cb = CODE[s]
    k, nn, _, _ = kappa_w(ca, cb, mask, 4, ONE)
    kb = kappa_w(ca, cb, mask, 4, WT)[0]
    return float(k[0]), int(nn[0]), kb


# ---------------------------------------------------------------- 1. per-set kappa (reproduction of the reported values)
SETROWS, SB = [], {}
for sub, m in MASK.items():
    for s in SETS:
        k, nn, kb = kap(s, m)
        SB[(sub, s)] = kb
        SETROWS.append(dict(subset=sub, set=sname(s), k=K(s), kappa=k, lo=ci(kb)[0], hi=ci(kb)[1], n_untied=nn,
                            units=int(m.sum())))
SETS_DF = pd.DataFrame(SETROWS)
GRP = []
for sub in MASK:
    for k in (0, 1, 2):
        mem = [s for s in SETS if K(s) == k]
        b = np.nanmean(np.vstack([SB[(sub, s)] for s in mem]), 0)
        v = float(np.mean([SETS_DF[(SETS_DF.subset == sub) & (SETS_DF.set == sname(s))].kappa.iloc[0] for s in mem]))
        GRP.append(dict(subset=sub, k=k, n_sets=len(mem), kappa=v, lo=ci(b)[0], hi=ci(b)[1]))
GRP = pd.DataFrame(GRP)

# ---------------------------------------------------------------- 2. matched swaps
SWAPS = []
for a in SETS:
    for b in SETS:
        if K(b) == K(a) + 1 and len(set(a) & set(b)) == 3:
            removed, = set(a) - set(b)
            added, = set(b) - set(a)
            assert removed in PW and added in LW
            SWAPS.append((a, b, removed, added))
assert sum(K(a) == 0 for a, *_ in SWAPS) == 8 and sum(K(a) == 1 for a, *_ in SWAPS) == 24

ROWS, DB = [], {}
for sub, m in MASK.items():
    for a, b, rem, add in SWAPS:
        for unit in ('own', 'common'):
            if unit == 'own':
                ka, na, ba = kap(a, m)
                kb_, nb, bb = kap(b, m)
                ncommon = int((m & untied[a] & untied[b]).sum())
            else:
                cm = m & untied[a] & untied[b]
                ka, na, ba = kap(a, cm)
                kb_, nb, bb = kap(b, cm)
                ncommon = int(cm.sum())
            d = ba - bb
            DB[(sub, unit, a, b)] = d
            ROWS.append(dict(subset=sub, units=unit, step=f'k{K(a)}->{K(b)}', set_A=sname(a), set_B=sname(b),
                             removed=NAME[rem], added=NAME[add],
                             other_listwise_in_A=next((NAME[x] for x in a if x in LW), ''),
                             kappa_A=ka, kappa_B=kb_, n_untied_A=na, n_untied_B=nb, n_common=ncommon,
                             delta=ka - kb_, lo=ci(d)[0], hi=ci(d)[1]))
SW = pd.DataFrame(ROWS)


def summarize(sel, label, sub, unit):
    keys = [(sub, unit, a, b) for a, b, rem, add in SWAPS if sel(a, b, rem, add)]
    rows = SW[(SW.subset == sub) & (SW.units == unit)]
    rows = rows[[sel(a, b, r, d) for a, b, r, d in SWAPS]]
    mb = np.nanmean(np.vstack([DB[k] for k in keys]), 0)
    return dict(subset=sub, units=unit, comparison=label, n_swaps=len(keys), mean_delta=float(rows.delta.mean()),
                lo=ci(mb)[0], hi=ci(mb)[1], min_delta=float(rows.delta.min()), max_delta=float(rows.delta.max()),
                swaps_delta_pos=int((rows.delta > 0).sum()), swaps_ci_above_0=int((rows.lo > 0).sum()),
                swaps_ci_below_0=int((rows.hi < 0).sum()), mean_n_common=float(rows.n_common.mean()))


SUM = []
for sub in MASK:
    for unit in ('own', 'common'):
        for step in (0, 1):
            SUM.append(summarize(lambda a, b, r, d, st=step: K(a) == st, f'k{step}->{step + 1}: all swaps', sub, unit))
            for m in LW:
                SUM.append(summarize(lambda a, b, r, d, st=step, m=m: K(a) == st and d == m,
                                     f'k{step}->{step + 1}: add {NAME[m]}', sub, unit))
SUM = pd.DataFrame(SUM)

# ---------------------------------------------------------------- 3. model identity, paired contrasts
# k0->1: kappa(PW4 - p + FIRST) - kappa(PW4 - p + RankZephyr), paired by the removed pointwise reranker p (4 pairs).
# k1->2: for each k=2 set B and each pointwise p not in B: A_F = B - RankZephyr + p (contains FIRST),
#        A_Z = B - FIRST + p (contains RankZephyr); Delta(add RZ to a FIRST set) - Delta(add FIRST to a RZ set)
#        = kappa(A_F) - kappa(A_Z) (B cancels), 12 pairs.
IDR = []
for sub, m in MASK.items():
    for unit in ('own', 'common'):
        for step in (0, 1):
            pairs = []
            if step == 0:
                pw4 = tuple(PW)
                for p in PW:
                    rest = [x for x in pw4 if x != p]
                    sf, sz = tuple(sorted(rest + ['first'], key=ALL6.index)), tuple(sorted(rest + ['zephyr'], key=ALL6.index))
                    pairs.append((sf, sz, pw4, NAME[p]))
            else:
                for B in [s for s in SETS if K(s) == 2]:
                    pws = [x for x in B if x in PW]
                    for p in [x for x in PW if x not in B]:
                        af = tuple(sorted(pws + [p, 'first'], key=ALL6.index))
                        az = tuple(sorted(pws + [p, 'zephyr'], key=ALL6.index))
                        pairs.append((af, az, B, f'{sname(B)} / {NAME[p]}'))
            vals, boots = [], []
            for sf, sz, ref, lab in pairs:
                if unit == 'own':
                    kf, _, bf = kap(sf, m)
                    kz, _, bz = kap(sz, m)
                else:
                    cm = m & untied[sf] & untied[sz] & untied[ref]
                    kf, _, bf = kap(sf, cm)
                    kz, _, bz = kap(sz, cm)
                vals.append(kf - kz)
                boots.append(bf - bz)
            mb = np.nanmean(np.vstack(boots), 0)
            IDR.append(dict(subset=sub, units=unit, step=f'k{step}->{step + 1}',
                            contrast='kappa(set with FIRST) - kappa(set with RankZephyr), same other rerankers',
                            note=('B cancels: with own units this equals the k0->1 contrast (same k=1 sets); with common '
                                  'units it differs only through the unit mask') if step == 1 else
                                 'paired by the removed pointwise reranker',
                            n_pairs=len(pairs), mean=float(np.mean(vals)), lo=ci(mb)[0], hi=ci(mb)[1],
                            min=float(np.min(vals)), max=float(np.max(vals)), pairs_pos=int(np.sum(np.array(vals) > 0))))
IDR = pd.DataFrame(IDR)

# ---------------------------------------------------------------- 4. comparison with the reported values
KD = os.path.join(REPO, 'results', 'summary', 'derived')
rep_sets = pd.read_csv(f'{KD}/ksets_sets.csv')
rep_grp = pd.read_csv(f'{KD}/ksets_groups.csv')
rep_grp = rep_grp[rep_grp.scope == 'evaluation']
rep_diff = pd.read_csv(f'{KD}/ksets_group_diffs.csv')
rep_diff = rep_diff[rep_diff.scope == 'evaluation']
CMP = []
for _, r in GRP.iterrows():
    o = rep_grp[(rep_grp.subset == r.subset) & (rep_grp.k == r.k)].iloc[0]
    CMP.append(dict(item=f'group kappa k={r.k} ({r.subset})', reported=o.kappa, reported_lo=o.lo, reported_hi=o.hi,
                    this=r.kappa, this_lo=r.lo, this_hi=r.hi, abs_diff=abs(o.kappa - r.kappa)))
for sub in MASK:
    for _, r in SETS_DF[SETS_DF.subset == sub].iterrows():
        o = rep_sets[rep_sets.set == r.set].iloc[0]
        CMP.append(dict(item=f'set kappa {r.set} ({sub})', reported=o[f'kappa_{sub}'], reported_lo=o[f'lo_{sub}'],
                        reported_hi=o[f'hi_{sub}'], this=r.kappa, this_lo=r.lo, this_hi=r.hi,
                        abs_diff=abs(o[f'kappa_{sub}'] - r.kappa)))
for sub in MASK:
    for st, dname in ((0, 'k0-k1'), (1, 'k1-k2')):
        s = SUM[(SUM.subset == sub) & (SUM.units == 'own') & (SUM.comparison == f'k{st}->{st + 1}: all swaps')].iloc[0]
        o = rep_diff[(rep_diff.subset == sub) & (rep_diff['diff'] == dname)].iloc[0]
        CMP.append(dict(item=f'mean matched Delta k{st}->{st + 1} (own units) vs reported group difference {dname} ({sub})',
                        reported=o.value, reported_lo=o.lo, reported_hi=o.hi, this=s.mean_delta, this_lo=s.lo,
                        this_hi=s.hi, abs_diff=abs(o.value - s.mean_delta)))
CMP = pd.DataFrame(CMP)

SETS_DF.to_csv(f'{OUT}/matched_swap_sets_kappa.csv', index=False)
GRP.to_csv(f'{OUT}/matched_swap_groups_kappa.csv', index=False)
SW.to_csv(f'{OUT}/matched_swap_swaps.csv', index=False)
SUM.to_csv(f'{OUT}/matched_swap_swaps_summary.csv', index=False)
IDR.to_csv(f'{OUT}/matched_swap_model_identity.csv', index=False)
CMP.to_csv(f'{OUT}/matched_swap_comparison_with_reported.csv', index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
print(GRP.round(4).to_string())
print('max |diff| vs reported (kappa values):', CMP.abs_diff.max())
print(SUM.round(4).to_string())
print(IDR.round(4).to_string())
