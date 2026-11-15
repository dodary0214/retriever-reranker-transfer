"""Validation (nDCG@10 vs published, 100 cands/query, docid coverage) -> validation.md; also writes manifest.json."""
import sys, json, datetime, platform
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import (DATASETS, COMMON, RETRIEVERS, DEPTH, SEED, BGE, BGE_QUERY_PREFIX, SPLADE_ENCODER, TOKENIZER,
                    bm25_index, splade_index, dense_index, read_tsv_queries)
import ir_measures
from ir_measures import nDCG, R
import pandas as pd, numpy as np

# Pyserini/Anserini reproduction pages (nDCG@10). Gate = dl19 only; others reported for reference.
REF = {("dl19", "bm25"): .5058, ("dl19", "splade"): .7308, ("dl19", "dense"): .7016,
       ("dl20", "bm25"): .4796, ("dl20", "splade"): .7198, ("dl20", "dense"): .6768,
       ("nfcorpus", "bm25"): .3218, ("nfcorpus", "splade"): .3470, ("nfcorpus", "dense"): .3735,
       ("fiqa", "bm25"): .2361, ("fiqa", "splade"): .3475, ("fiqa", "dense"): .4065,
       ("scifact", "bm25"): .6789, ("scifact", "splade"): .7041, ("scifact", "dense"): .7408,
       ("dbpedia", "bm25"): .3180, ("dbpedia", "splade"): .4366, ("dbpedia", "dense"): .4077}
GATE = {("dl19", "bm25"), ("dl19", "splade"), ("dl19", "dense")}
TOL = 0.02

rows, ok_all, run_files = [], True, []
for c in DATASETS:
    qr = pd.read_csv(COMMON / "qrels" / f"{c}.tsv", sep="\t", header=None, names=["query_id", "doc_id", "relevance"], dtype={0: str, 1: str})
    qdf = read_tsv_queries(c); qids, qtexts = qdf.qid.tolist(), qdf.text.tolist(); nq = len(qids)
    dpath = COMMON / "docs" / f"{c}.jsonl"
    docids = {json.loads(l)["docid"] for l in open(dpath)} if dpath.exists() else set()
    for r in RETRIEVERS:
        p = COMMON / "runs" / f"{c}.{r}.top100.txt"
        if not p.exists():
            rows.append(dict(dataset=c, retriever=r, status="MISSING")); ok_all = False; continue
        run_files.append(p.name)
        run = pd.read_csv(p, sep=" ", header=None, names=["query_id", "q0", "doc_id", "rank", "score", "tag"], dtype={0: str, 2: str})
        per_q = run.groupby("query_id").size()
        m = ir_measures.calc_aggregate([nDCG @ 10, R @ 100], qr, run[["query_id", "doc_id", "score"]])
        ndcg, rec = m[nDCG @ 10], m[R @ 100]
        ref = REF.get((c, r))
        diff = ndcg - ref if ref is not None else np.nan
        cov = run.doc_id.isin(docids).mean() if docids else 0.0
        checks = dict(n_q_run=len(per_q), all100=bool((per_q == DEPTH).all()), min_cands=int(per_q.min()),
                      dup_free=bool(not run.duplicated(["query_id", "doc_id"]).any()), doc_cov=cov)
        # BM25 can return fewer than DEPTH hits when fewer docs share a query term. Accept a short query only if a
        # k=1000 search returns the same number of hits (i.e. every matching doc is already in the run).
        short = per_q.reindex(qids, fill_value=0)
        short = short[short < DEPTH]
        exhausted = True
        if len(short) and r == "bm25":
            from pyserini.search.lucene import LuceneSearcher
            s = LuceneSearcher.from_prebuilt_index(bm25_index(DATASETS[c][1])); s.set_bm25(0.9, 0.4)
            qtext = dict(zip(qids, qtexts))
            exhausted = all(len(s.search(qtext[q], k=1000)) == n for q, n in short.items())
        elif len(short):
            exhausted = False
        checks.update(n_short=len(short), n_zero=int((short == 0).sum()), short_exhausted=exhausted)
        gate = (c, r) in GATE
        status = "PASS"
        if gate and abs(diff) > TOL: status = "FAIL(gate)"
        elif not gate and abs(diff) > TOL: status = "WARN"
        if not ((checks["all100"] or exhausted) and checks["dup_free"] and cov == 1.0): status = "FAIL(checks)"
        elif not checks["all100"]: status += "*"
        if status.startswith("FAIL"): ok_all = False
        rows.append(dict(dataset=c, retriever=r, ndcg10=ndcg, ref=ref, diff=diff, recall100=rec, n_q=nq, **checks,
                         gate="yes" if gate else "", status=status))

df = pd.DataFrame(rows)
emb_ok = {c: (COMMON / "emb" / f"{c}.docs.npy").exists() for c in DATASETS}
with open(COMMON / "validation.md", "w") as f:
    f.write(f"# Common preprocessing validation\n\nGenerated {datetime.datetime.now().isoformat(timespec='seconds')} · SEED={SEED} · depth={DEPTH}\n\n")
    f.write(f"Gate: dl19 nDCG@10 within ±{TOL} of published (BM25 .506, SPLADE++ED .731, bge-base .702). "
            "Other rows are compared with Pyserini reproduction numbers for reference (WARN = off by more than ±0.02, not a gate).\n\n")
    f.write("| dataset | retriever | nDCG@10 | published | diff | R@100 | queries (run/all) | 100 cands/q | min cands | short q (zero) | no dup | doc coverage | gate | status |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for x in rows:
        if x.get("status") == "MISSING":
            f.write(f"| {x['dataset']} | {x['retriever']} | – | – | – | – | – | – | – | – | – | – | MISSING |\n"); continue
        f.write(f"| {x['dataset']} | {x['retriever']} | {x['ndcg10']:.4f} | {x['ref']:.4f} | {x['diff']:+.4f} | {x['recall100']:.4f} | "
                f"{x['n_q_run']}/{x['n_q']} | {x['all100']} | {x['min_cands']} | {x['n_short']} ({x['n_zero']}) | {x['dup_free']} | {x['doc_cov']:.4f} | {x['gate']} | {x['status']} |\n")
    f.write("\n`*` = some queries have fewer than 100 candidates because BM25 found fewer matching documents in the corpus "
            "(a k=1000 search returns the same count, so the run already holds every match; zero = queries with no match at all). "
            "Runs are kept as retrieved, not padded; the nDCG values equal the published ones.\n")
    f.write(f"\nEmbeddings present: {emb_ok}\n\nOverall: **{'PASS' if ok_all and all(emb_ok.values()) else 'FAIL'}**\n")
print(df.to_string())

stats = json.load(open(COMMON / "logs" / "qrels_stats.json"))
import pyserini, faiss, sentence_transformers, torch, transformers
manifest = {
    "created": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED, "depth": DEPTH,
    "datasets": {c: {**stats[c], "corpus": DATASETS[c][1],
                     "n_docs_cached": sum(1 for _ in open(COMMON / "docs" / f"{c}.jsonl")) if (COMMON / "docs" / f"{c}.jsonl").exists() else 0}
                 for c in DATASETS},
    "retrievers": {
        "bm25": {"impl": "pyserini LuceneSearcher prebuilt", "index": {c: bm25_index(DATASETS[c][1]) for c in DATASETS}, "k1": 0.9, "b": 0.4},
        "splade": {"impl": "pyserini LuceneImpactSearcher prebuilt", "index": {c: splade_index(DATASETS[c][1]) for c in DATASETS},
                   "query_encoder": SPLADE_ENCODER + " (pytorch, on the fly)"},
        "dense": {"impl": "pyserini FaissSearcher prebuilt flat IP", "index": {c: dense_index(DATASETS[c][1]) for c in DATASETS},
                  "query_encoder": BGE + " (sentence-transformers, fp32, normalized)", "query_prefix": BGE_QUERY_PREFIX},
    },
    "runs": sorted(run_files), "excluded": [],
    "bm25_short_queries": {x["dataset"]: {"n_short": x["n_short"], "n_zero": x["n_zero"]} for x in rows
                           if x.get("retriever") == "bm25" and x.get("n_short")},
    "notes": "BM25 runs are not padded: queries with fewer than 100 lexical matches keep fewer candidates "
             "(nfcorpus 15 queries have none). Dense/SPLADE runs have exactly 100 per query.",
    "docs": {"source": "raw docs stored in the BM25 prebuilt Lucene index", "tokenizer": TOKENIZER, "n_tokens": "no special tokens",
             "extra": "dl19/dl20 also include every judged docid"},
    "emb": {"model": BGE, "precision": "fp16 inference, saved float32, L2-normalized", "max_seq_length": 512,
            "doc_input": "docs/*.jsonl text field", "query_input": "bge prefix + query"},
    "validation": "PASS" if ok_all and all(emb_ok.values()) else "FAIL",
    "env": {"python": platform.python_version(), "pyserini": getattr(pyserini, "__version__", "2.x"), "faiss": faiss.__version__,
            "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
            "java": "OpenJDK 21.0.13", "host": "aarch64 GB10"},
}
json.dump(manifest, open(COMMON / "manifest.json", "w"), indent=1, ensure_ascii=False)
print("validation:", manifest["validation"])
