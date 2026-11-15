"""Query set E: listwise outputs (out/<model>.jsonl) -> runs/<dataset>.<retriever>.<model>.txt (TREC, score
1000 - rank), swap_listwise.csv (dataset, qid, cond, seed, reranker, rank, docid), grid_mix.parquet (= pointwise/grid.parquet
plus zephyr/first rows for the 547 queries), grid_mix_summary.md (means, gate, malformed/missing counts)."""
import json
import os
import sys

import ir_measures
import pandas as pd
from ir_measures import nDCG

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import EXP, LISTWISE_E, POINTWISE, RETRIEVERS

P, P2 = LISTWISE_E, POINTWISE

LW = ['zephyr', 'first']
GATE, TOL = {'zephyr': 0.74, 'first': 0.72}, 0.03
Q = pd.read_csv(f'{P}/queries_mix.csv', dtype={'qid': str})
os.makedirs(f'{P}/runs', exist_ok=True)

grid = pd.read_parquet(f'{P2}/grid.parquet')
rows, swap, stats = [], [], []
for m in LW:
    path = f'{P}/out/{m}.jsonl'
    if not os.path.exists(path):
        continue
    out = {}
    for l in open(path):
        x = json.loads(l)
        out[x['job_id']] = x
    stats.append(dict(reranker=m, lists=len(out), calls=sum(x['n_calls'] for x in out.values()),
                      bad=sum(x['n_bad'] for x in out.values()),
                      bad_unit='malformed responses' if m == 'zephyr' else 'identifiers missing from top-30 logprobs'))
    for ds in Q.dataset.unique():
        qrels = pd.read_csv(f'{EXP}/common/qrels/{ds}.tsv', sep='\t', names=['query_id', 'doc_id', 'relevance'],
                            dtype={'query_id': str, 'doc_id': str, 'relevance': int})
        for r in RETRIEVERS:
            qids = [q for q in Q[Q.dataset == ds].qid if f'grid|{ds}|{r}|{q}' in out]
            if not qids:
                continue
            with open(f'{P}/runs/{ds}.{r}.{m}.txt', 'w') as f:
                for q in qids:
                    for i, d in enumerate(out[f'grid|{ds}|{r}|{q}']['ranking']):
                        f.write(f'{q} Q0 {d} {i + 1} {1000 - i} {m}\n')
            per_q = {x.query_id: x.value for x in ir_measures.iter_calc(
                [nDCG @ 10], qrels, ir_measures.read_trec_run(f'{P}/runs/{ds}.{r}.{m}.txt'))}
            rows += [(ds, q, r, m, float(per_q.get(q, 0.0))) for q in qids]
    for k, x in out.items():
        if k.startswith('swap|'):
            _, ds, cond, seed, qid = k.split('|')
            swap += [(ds, qid, cond, int(seed), m, i + 1, d) for i, d in enumerate(x['ranking'])]

new = pd.DataFrame(rows, columns=grid.columns)
gm = pd.concat([grid, new], ignore_index=True)
gm.to_parquet(f'{P}/grid_mix.parquet', index=False)
pd.DataFrame(swap, columns=['dataset', 'qid', 'cond', 'seed', 'reranker', 'rank', 'docid']).to_csv(f'{P}/swap_listwise.csv', index=False)

sub = gm.merge(Q[['dataset', 'qid']], on=['dataset', 'qid'])
sub = sub[sub.reranker.isin(['none', 'bge-m3', 'monot5', *LW])]
cnt = sub.groupby(['dataset', 'retriever', 'reranker']).size().unstack('reranker')
summ = sub.groupby(['dataset', 'retriever', 'reranker'], sort=False).ndcg10.mean().unstack('reranker')
summ = summ[[c for c in ['none', 'bge-m3', 'monot5', *LW] if c in summ]].round(4)
gate, ok = [], True
for m, ref in GATE.items():
    if m in summ:
        v = summ.loc[('dl19', 'bm25'), m]
        good = abs(v - ref) <= TOL
        ok &= good
        gate.append(f'| dl19 bm25+{m} | {v:.4f} | ≈{ref} ±{TOL} | {"PASS" if good else "FAIL"} |')
md = ['# Mixed grid summary (mean nDCG@10, 547-query set)', '',
      f'Queries per dataset: {Q.groupby("dataset").size().to_dict()}. Rows per cell (min over cells): '
      f'{cnt.min().to_dict()}.', '', summ.reset_index().to_markdown(index=False), '',
      '## Gate', '', '| combination | nDCG@10 | expected | status |', '|---|---|---|---|', *gate, '',
      f'Overall: **{"PASS" if ok and len(gate) == 2 else "FAIL" if not ok else "INCOMPLETE"}**', '',
      '## Listwise output checks', '', pd.DataFrame(stats).to_markdown(index=False) if stats else '(no output)', '']
open(f'{P}/grid_mix_summary.md', 'w').write('\n'.join(md))
print('\n'.join(md))
