"""Per-query nDCG@10 grid -> grid.parquet (dataset, qid, retriever, reranker, ndcg10),
reranker='none' = first stage. Mean table + sanity gate -> grid_summary.md."""
import os
import sys

import ir_measures
import pandas as pd
from ir_measures import nDCG

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import DATASETS, EXP, POINTWISE, RETRIEVERS, read_queries
from run_rerankers import RERANKERS

P2 = POINTWISE
GATE = {('dl19', 'bm25', 'bge-m3'): 0.72, ('dl19', 'bm25', 'monot5'): 0.71}
TOL = 0.03

rows = []
for ds in DATASETS:
    qrels = pd.read_csv(f'{EXP}/common/qrels/{ds}.tsv', sep='\t', names=['query_id', 'doc_id', 'relevance'],
                        dtype={'query_id': str, 'doc_id': str, 'relevance': int})
    qids = list(read_queries(ds))
    for r in RETRIEVERS:
        for rr in ['none', *RERANKERS]:
            path = f'{EXP}/common/runs/{ds}.{r}.top100.txt' if rr == 'none' else f'{P2}/runs/{ds}.{r}.{rr}.txt'
            per_q = {m.query_id: m.value for m in ir_measures.iter_calc([nDCG @ 10], qrels, ir_measures.read_trec_run(path))}
            rows += [(ds, q, r, rr, float(per_q.get(q, 0.0))) for q in qids]  # no hits -> 0
grid = pd.DataFrame(rows, columns=['dataset', 'qid', 'retriever', 'reranker', 'ndcg10'])
grid.to_parquet(f'{P2}/grid.parquet', index=False)

summ = grid.groupby(['dataset', 'retriever', 'reranker'], sort=False).ndcg10.mean().unstack('reranker')
summ = summ[['none', *RERANKERS]].round(4)
gate_lines, ok = [], True
for (ds, r, rr), ref in GATE.items():
    v = summ.loc[(ds, r), rr]
    good = abs(v - ref) <= TOL
    ok &= good
    gate_lines.append(f'| {ds} {r}+{rr} | {v:.4f} | ≈{ref} ±{TOL} | {"PASS" if good else "FAIL"} |')
md = ['# Grid summary (mean nDCG@10)', '',
      f'{len(grid)} rows, {grid.qid.nunique()} unique qids; reranker `none` = first stage.', '',
      summ.reset_index().to_markdown(index=False), '',
      '## Sanity gate', '', '| combination | nDCG@10 | expected | status |', '|---|---|---|---|',
      *gate_lines, '', f'Overall: **{"PASS" if ok else "FAIL"}**', '']
open(f'{P2}/grid_summary.md', 'w').write('\n'.join(md))
print('\n'.join(md))
sys.exit(0 if ok else 1)
