"""Raw nDCG@10 behind the oracle gain retention, per four-reranker set and k group (definition fixed before computing).
CPU only, existing scores only; executes analyze.py's definitions (no files written there).
Writes $RRT_OUT/derived/ksets_selection_ndcg.csv."""
import itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from common import OUT as OUT_ROOT, QWEN, REPO  # noqa: E402
AN = os.path.join(REPO, 'analysis', 'analyze.py')
OUT = os.path.join(OUT_ROOT, 'derived')
os.makedirs(OUT, exist_ok=True)
src = open(AN).read()
g = {'__file__': AN, '__name__': 'analyze_prefix'}
exec(compile(src[:src.index('# ================================================================ jaccard / same_rel')], AN, 'exec'), g)
EVAL, ALL6, LW, X, bs_index, RET = (g[x] for x in ('EVAL', 'ALL6', 'LW', 'X', 'bs_index', 'RETRIEVERS'))
ds = np.array([k[0] for k in EVAL]); ix = np.arange(len(EVAL))
rows = []
for c in itertools.combinations(ALL6, 4):
    rr = list(c)
    XX = {r: X('nDCG@10', EVAL, rr, r) for r in RET}
    BS, ST = {}, {}
    for r in RET:
        BS[r], ST[r] = bs_index(XX[r], ds)
    o, t, b, ret = [], [], [], []
    for a in RET:
        for y in RET:
            if a == y:
                continue
            o.append(XX[y].max(1).mean()); t.append(XX[y][ix, ST[a]].mean()); b.append(XX[y][ix, BS[y]].mean())
            ret.append((t[-1] - b[-1]) / (o[-1] - b[-1]))
    rows.append(dict(set='+'.join(c), k=sum(x in LW for x in c), oracle=np.mean(o), transferred=np.mean(t),
                     best_single=np.mean(b), retention_from_means=np.mean(ret)))
S = pd.DataFrame(rows)
G = S.groupby('k')[['oracle', 'transferred', 'best_single', 'retention_from_means']].mean().reset_index()
pd.concat([S.assign(level='set'), G.assign(level='group')]).to_csv(f'{OUT}/ksets_selection_ndcg.csv', index=False)
print(S.round(4).to_string()); print(G.round(4).to_string())
