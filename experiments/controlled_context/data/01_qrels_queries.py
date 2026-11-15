"""Write qrels (qid, docid, grade) and queries (qid, text) TSVs from ir_datasets, keeping original grades."""
import sys, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import DATASETS, COMMON
import ir_datasets
import pandas as pd

stats = {}
for code, (irds_id, _) in DATASETS.items():
    ds = ir_datasets.load(irds_id)
    q = pd.DataFrame([(x.query_id, x.text) for x in ds.queries_iter()], columns=["qid", "text"])
    r = pd.DataFrame([(x.query_id, x.doc_id, int(x.relevance)) for x in ds.qrels_iter()], columns=["qid", "docid", "grade"])
    r = r.drop_duplicates(["qid", "docid"])
    q = q[q.qid.isin(set(r.qid))]  # judged queries only
    q.to_csv(COMMON / "queries" / f"{code}.tsv", sep="\t", index=False, header=False)
    r.to_csv(COMMON / "qrels" / f"{code}.tsv", sep="\t", index=False, header=False)
    stats[code] = {
        "irds_id": irds_id, "n_queries": len(q), "n_judged": len(r),
        "n_judged_docs_unique": int(r.docid.nunique()),
        "grade_counts": {str(k): int(v) for k, v in r.grade.value_counts().sort_index().items()},
        "n_pos": int((r.grade > 0).sum()),
    }
    print(code, stats[code], flush=True)
json.dump(stats, open(COMMON / "logs" / "qrels_stats.json", "w"), indent=1)
