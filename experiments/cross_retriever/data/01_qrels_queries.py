"""Write qrels/<ds>.tsv (qid, docid, grade) and queries/<ds>.tsv (qid, text) from ir_datasets."""
import json
import os
import sys

import ir_datasets
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import DATASETS, EXP, update_manifest

stats = {}
for ds, cfg in DATASETS.items():
    d = ir_datasets.load(cfg['irds'])
    qrels = pd.DataFrame([(q.query_id, q.doc_id, int(q.relevance)) for q in d.qrels_iter()],
                         columns=['qid', 'docid', 'grade']).drop_duplicates(['qid', 'docid'])
    queries = pd.DataFrame([(q.query_id, q.text) for q in d.queries_iter()], columns=['qid', 'text'])
    # keep only queries that have judgments (judged subsets already do this; BEIR test splits too)
    queries = queries[queries.qid.isin(set(qrels.qid))]
    queries['text'] = queries.text.str.replace(r'[\t\n\r]+', ' ', regex=True).str.strip()
    qrels.to_csv(f'{EXP}/common/qrels/{ds}.tsv', sep='\t', index=False, header=False)
    queries.to_csv(f'{EXP}/common/queries/{ds}.tsv', sep='\t', index=False, header=False)
    stats[ds] = {
        'irds_id': cfg['irds'],
        'n_queries': len(queries),
        'n_judged': len(qrels),
        'n_judged_pos': int((qrels.grade > 0).sum()),
        'grades': sorted(qrels.grade.unique().tolist()),
    }
    print(ds, stats[ds], flush=True)

update_manifest({'datasets': stats})
