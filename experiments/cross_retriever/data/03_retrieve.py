"""Top-100 runs for bm25 / splade / dense -> runs/<ds>.<retriever>.top100.txt.
BEIR runs drop docid == qid (Pyserini 2CR --remove-query)."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import DATASETS, DEPTH, EXP, index_name, mem_share, read_queries, start_watchdog, write_run

ap = argparse.ArgumentParser()
ap.add_argument('retriever', choices=['bm25', 'splade', 'dense'])
ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
args = ap.parse_args()
start_watchdog()


def finish(ds, raw):
    beir = not DATASETS[ds]['pyserini'].startswith('msmarco')
    res = {}
    for qid, hits in raw.items():
        if beir:
            hits = [h for h in hits if h[0] != qid]
        res[qid] = hits[:DEPTH]
    path = f'{EXP}/common/runs/{ds}.{args.retriever}.top100.txt'
    write_run(path, res, args.retriever)
    print(ds, args.retriever, len(res), 'queries ->', path, flush=True)


if args.retriever in ('bm25', 'splade'):
    from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher
    for ds in args.datasets:
        q = read_queries(ds)
        ids = list(q)
        if args.retriever == 'bm25':
            s = LuceneSearcher.from_prebuilt_index(index_name(ds, 'bm25'))
            s.set_bm25(0.9, 0.4)
        else:
            s = LuceneImpactSearcher.from_prebuilt_index(index_name(ds, 'splade'), 'naver/splade-cocondenser-ensembledistil')
        out = s.batch_search([q[i] for i in ids], ids, k=DEPTH + 1, threads=16)
        finish(ds, {qid: [(h.docid, h.score) for h in out[qid]] for qid in ids})
else:
    import faiss
    from pyserini.util import download_prebuilt_index
    faiss.omp_set_num_threads(16)
    by_index = {}
    for ds in args.datasets:
        by_index.setdefault(index_name(ds, 'dense'), []).append(ds)
    for name, dss in by_index.items():  # dl19/dl20 share one 27 GB index: load once
        d = download_prebuilt_index(name)
        need_gb = os.path.getsize(os.path.join(d, 'index')) / 2**30 + 4  # flat index is read fully into RAM
        with mem_share().reserve(need_gb, f'rrt/faiss-{name}'):
            index = faiss.read_index(os.path.join(d, 'index'))
        docids = [l.strip() for l in open(os.path.join(d, 'docid'))]
        assert index.ntotal == len(docids)
        for ds in dss:
            qemb = np.load(f'{EXP}/common/emb/{ds}.queries.npy')
            qids = open(f'{EXP}/common/emb/{ds}.queries.ids.txt').read().split()
            scores, idx = index.search(qemb, DEPTH + 1)
            finish(ds, {qid: [(docids[j], float(s)) for s, j in zip(scores[r], idx[r]) if j >= 0]
                        for r, qid in enumerate(qids)})
        del index
