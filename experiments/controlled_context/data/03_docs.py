"""Doc text cache: every candidate docid in runs/ (+ all judged docids for dl19/dl20) -> docs/<dataset>.jsonl
fields: docid, title, text (as indexed: title + body), n_tokens (bert-base-uncased, no special tokens)."""
import sys, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import DATASETS, COMMON, RETRIEVERS, TOKENIZER, bm25_index
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer
import pandas as pd

tok = AutoTokenizer.from_pretrained(TOKENIZER)
codes = sys.argv[1:] or list(DATASETS)
searchers = {}
for c in codes:
    corpus = DATASETS[c][1]
    ids = set()
    for r in RETRIEVERS:
        p = COMMON / "runs" / f"{c}.{r}.top100.txt"
        if p.exists():
            ids |= set(pd.read_csv(p, sep=" ", header=None, dtype=str)[2])
    if c in ("dl19", "dl20"):
        ids |= set(pd.read_csv(COMMON / "qrels" / f"{c}.tsv", sep="\t", header=None, dtype=str)[1])
    if corpus not in searchers:
        searchers[corpus] = LuceneSearcher.from_prebuilt_index(bm25_index(corpus))
    s = searchers[corpus]
    rows, missing = [], []
    for d in sorted(ids):
        doc = s.doc(d)
        if doc is None:
            missing.append(d); continue
        raw = json.loads(doc.raw())
        title = raw.get("title", "") or ""
        body = raw.get("text", raw.get("contents", "")) or ""
        text = f"{title} {body}".strip() if title and not body.startswith(title) else body
        rows.append({"docid": d, "title": title, "text": text})
    ntok = [len(x) for x in tok([r["text"] for r in rows], add_special_tokens=False)["input_ids"]]
    with open(COMMON / "docs" / f"{c}.jsonl", "w") as f:
        for r, n in zip(rows, ntok):
            r["n_tokens"] = n
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{c}: {len(rows)} docs, missing {len(missing)} {missing[:5]}", flush=True)
