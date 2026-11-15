"""Top-100 candidates per (dataset, retriever) -> runs/<dataset>.<retriever>.top100.txt (skips existing files).
usage: 02_retrieve.py <bm25|splade|dense> [dataset ...]"""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import (DATASETS, COMMON, DEPTH, SEED, BGE, BGE_QUERY_PREFIX, SPLADE_ENCODER, bm25_index, splade_index,
                    dense_index, mem_watchdog, read_tsv_queries, write_run)
import numpy as np

retriever = sys.argv[1]
codes = sys.argv[2:] or list(DATASETS)
mem_watchdog()
todo = [c for c in codes if not (COMMON / "runs" / f"{c}.{retriever}.top100.txt").exists()]
by_corpus = {}
for c in todo:
    by_corpus.setdefault(DATASETS[c][1], []).append(c)


def lucene_search(searcher, q):
    res = searcher.batch_search(q.text.tolist(), q.qid.tolist(), k=DEPTH, threads=16)
    return {qid: [(h.docid, h.score) for h in res[qid]] for qid in q.qid}


if retriever == "dense":
    import torch
    from sentence_transformers import SentenceTransformer
    from host_resources import reserve
    torch.manual_seed(SEED)
    with reserve(4, "rrt/bge-query-encode"):
        model = SentenceTransformer(BGE, device="cuda")
    qemb = {}
    for c in todo:
        q = read_tsv_queries(c)
        qemb[c] = model.encode([BGE_QUERY_PREFIX + t for t in q.text], batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True).astype(np.float32)
    del model; torch.cuda.empty_cache()

for corpus, cs in by_corpus.items():
    t = time.time()
    if retriever == "bm25":
        from pyserini.search.lucene import LuceneSearcher
        s = LuceneSearcher.from_prebuilt_index(bm25_index(corpus))
        s.set_bm25(0.9, 0.4)
    elif retriever == "splade":
        from pyserini.search.lucene import LuceneImpactSearcher
        s = LuceneImpactSearcher.from_prebuilt_index(splade_index(corpus), SPLADE_ENCODER)
    elif retriever == "dense":
        from pyserini.search.faiss import FaissSearcher
        from pyserini.util import download_prebuilt_index
        from host_resources import reserve
        gb = 30 if corpus == "msmarco-v1-passage" else (16 if "dbpedia" in corpus else 2)
        with reserve(gb, f"rrt/faiss-{corpus}"):
            s = FaissSearcher(download_prebuilt_index(dense_index(corpus)), None)
    print(f"[{retriever}] loaded {corpus} in {time.time()-t:.0f}s", flush=True)
    for c in cs:
        q = read_tsv_queries(c)
        if retriever == "dense":
            res = s.batch_search(qemb[c], q.qid.tolist(), k=DEPTH, threads=16)
            out = {qid: [(h.docid, float(h.score)) for h in res[qid]] for qid in q.qid}
        else:
            out = lucene_search(s, q)
        path = COMMON / "runs" / f"{c}.{retriever}.top100.txt"
        write_run(str(path) + ".tmp", out, retriever)
        (COMMON / "runs" / f"{c}.{retriever}.top100.txt.tmp").rename(path)
        print(f"[{retriever}] wrote {path.name} ({len(out)} queries)", flush=True)
    del s
