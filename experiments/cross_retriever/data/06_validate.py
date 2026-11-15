"""nDCG@10 vs Pyserini 2CR references, depth/docs checks -> validation.md. Exit 1 on any FAIL."""
import json
import os
import sys

import ir_measures
import pandas as pd
from ir_measures import nDCG, R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import DATASETS, DEPTH, EXP, RETRIEVERS, read_run, update_manifest

TOL = 0.02
# Pyserini 2CR nDCG@10 (msmarco-v1-passage.yaml / beir.yaml); the dl19 rows are a hard gate.
REF = {
    ('dl19', 'bm25'): 0.5058, ('dl19', 'splade'): 0.7308, ('dl19', 'dense'): 0.7016,
    ('dl20', 'bm25'): 0.4796, ('dl20', 'splade'): 0.7197, ('dl20', 'dense'): 0.6768,
    ('nfcorpus', 'bm25'): 0.3218, ('nfcorpus', 'splade'): 0.3473, ('nfcorpus', 'dense'): 0.3735,
    ('fiqa', 'bm25'): 0.2361, ('fiqa', 'splade'): 0.3473, ('fiqa', 'dense'): 0.4065,
    ('scifact', 'bm25'): 0.6789, ('scifact', 'splade'): 0.7036, ('scifact', 'dense'): 0.7408,
    ('dbpedia', 'bm25'): 0.3180, ('dbpedia', 'splade'): 0.4374, ('dbpedia', 'dense'): 0.4073,
}
GATE = {('dl19', 'bm25'), ('dl19', 'splade'), ('dl19', 'dense')}  # stop if outside ±0.02

rows, fails = [], []
for ds in DATASETS:
    qrels = pd.read_csv(f'{EXP}/common/qrels/{ds}.tsv', sep='\t', names=['query_id', 'doc_id', 'relevance'],
                        dtype={'query_id': str, 'doc_id': str, 'relevance': int})
    nq = sum(1 for _ in open(f'{EXP}/common/queries/{ds}.tsv'))
    docs_path = f'{EXP}/common/docs/{ds}.jsonl'
    cached = {json.loads(l)['docid'] for l in open(docs_path)} if os.path.exists(docs_path) else None
    for r in RETRIEVERS:
        path = f'{EXP}/common/runs/{ds}.{r}.top100.txt'
        if not os.path.exists(path):
            rows.append(dict(dataset=ds, retriever=r, status='MISSING'))
            continue
        run = read_run(path)
        trec = pd.DataFrame([(q, d, s) for q, hs in run.items() for d, s in hs], columns=['query_id', 'doc_id', 'score'])
        agg = ir_measures.calc_aggregate([nDCG @ 10, R @ 100], qrels, trec)
        ndcg = agg[nDCG @ 10]
        ref = REF[(ds, r)]
        n_short = sum(len(h) < DEPTH for h in run.values()) + (nq - len(run))
        # BM25 returns < k only when no further document shares a query term: report, do not pad
        depth_ok = n_short == 0 or r == 'bm25'
        missing_docs = None if cached is None else len({d for hs in run.values() for d, _ in hs} - cached)
        ok_ref = abs(ndcg - ref) <= TOL
        if not (depth_ok and missing_docs in (0, None)) or (not ok_ref and (ds, r) in GATE):
            status = 'FAIL'
            fails.append((ds, r))
        elif not ok_ref or n_short:
            status = 'WARN'
        else:
            status = 'PASS'
        rows.append(dict(dataset=ds, retriever=r, nDCG10=round(ndcg, 4), ref=ref, diff=round(ndcg - ref, 4),
                         R100=round(agg[R @ 100], 4), queries=f'{len(run)}/{nq}',
                         min_depth=min(len(h) for h in run.values()), short_queries=n_short,
                         docs_missing='n/a' if missing_docs is None else missing_docs,
                         gate='yes' if (ds, r) in GATE else '', status=status))

df = pd.DataFrame(rows)
md = ['# Validation (first-stage runs)', '',
      f'nDCG@10 by ir_measures vs Pyserini 2CR reference, tolerance ±{TOL}. Gate rows (dl19) stop the pipeline on FAIL; '
      'other rows outside the tolerance are WARN. Depth = 100 per query; short_queries = queries with < 100 candidates '
      '(BM25 only: lexical exhaustion, no doc left sharing a query term; runs are not padded, WARN). '
      'docs_missing = run docids absent from docs/ (must be 0).', '',
      df.to_markdown(index=False), '',
      f'Overall: **{"FAIL " + str(fails) if fails else "PASS"}**', '']
open(f'{EXP}/common/validation.md', 'w').write('\n'.join(md))
print('\n'.join(md))
update_manifest({'validation': {'overall': 'FAIL' if fails else 'PASS', 'fails': [list(f) for f in fails]}})
sys.exit(1 if fails else 0)
