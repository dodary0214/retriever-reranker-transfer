"""Within-type Spearman between |mean nDCG@10 difference| and sign
agreement (relevant-set-identical units) over the reranker pairs of a type (PP 6, PL 8); CI from the same query bootstrap
(both quantities recomputed per resample). Definition fixed before computing. Writes $RRT_OUT/derived/fig3_spearman.csv."""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from common import OUT as OUT_ROOT, QWEN, REPO  # noqa: E402
AN = os.path.join(REPO, 'analysis', 'analyze.py')
OUT = os.path.join(OUT_ROOT, 'derived')
os.makedirs(OUT, exist_ok=True)
src = open(AN).read()
g = {'__file__': AN, '__name__': 'analyze_prefix'}
exec(compile(src[:src.index('# ================================================================ figure 5')], AN, 'exec'), g)
EVAL, PAIRS, ALL6, X, BW, ci, same_rel, TH, RET, kind = (g[x] for x in ('EVAL', 'PAIRS', 'ALL6', 'X', 'BW', 'ci', 'same_rel', 'TH', 'RETRIEVERS', 'kind'))
n = len(EVAL); SR = same_rel(EVAL); W = BW(EVAL).astype(np.float64)
rows = {}
for r1, r2 in itertools.combinations(ALL6, 2):
    S, D = {}, []
    for r in RET:
        d = X('nDCG@10', EVAL, [r1], r)[:, 0] - X('nDCG@10', EVAL, [r2], r)[:, 0]
        D.append(d); S[r] = np.where(d >= TH, 1, np.where(d <= -TH, -1, 0))
    num = sum(((S[a] != 0) & (S[b] != 0) & (S[a] == S[b]) & SR[(a, b)]).astype(float) for a, b in PAIRS)
    den = sum(((S[a] != 0) & (S[b] != 0) & SR[(a, b)]).astype(float) for a, b in PAIRS)
    dm = np.mean(D, 0)
    rows[(r1, r2)] = (kind((r1, r2)), num.sum() / den.sum(), abs(dm.mean()), (W @ num) / (W @ den), np.abs(W @ dm / n))
out = []
for t in ('PP', 'PL'):
    ks = [k for k in rows if rows[k][0] == t]
    agree = np.array([rows[k][1] for k in ks]); gap = np.array([rows[k][2] for k in ks])
    rho = stats.spearmanr(gap, agree).statistic
    A = np.vstack([rows[k][3] for k in ks]).T; G = np.vstack([rows[k][4] for k in ks]).T
    rb = np.array([stats.spearmanr(G[i], A[i]).statistic for i in range(len(A))])
    lo, hi = ci(rb)
    out.append(dict(type=t, n_pairs=len(ks), spearman=rho, lo=lo, hi=hi))
o = pd.DataFrame(out); o.to_csv(f'{OUT}/fig3_spearman.csv', index=False); print(o.round(3).to_string())
