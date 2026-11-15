"""bge-base-en-v1.5 embeddings (L2-normalized, float32) for docs/<dataset>.jsonl and queries/<dataset>.tsv.
emb/<dataset>.docs.npy + .docs.ids.txt, emb/<dataset>.queries.npy + .queries.ids.txt (queries use the bge prefix)."""
import sys, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import DATASETS, COMMON, SEED, BGE, BGE_QUERY_PREFIX, mem_watchdog, read_tsv_queries
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from host_resources import reserve, claim

mem_watchdog()
torch.manual_seed(SEED)
codes = sys.argv[1:] or list(DATASETS)
with reserve(8, "rrt/bge-embed") as _:
    model = SentenceTransformer(BGE, device="cuda")
    model.half()
claim("rrt/bge-embed", 8)
model.max_seq_length = 512
E = COMMON / "emb"
for c in codes:
    if (E / f"{c}.docs.npy").exists():
        print(f"{c}: exists, skip", flush=True); continue
    docs = [json.loads(l) for l in open(COMMON / "docs" / f"{c}.jsonl")]
    order = np.argsort([-d["n_tokens"] for d in docs])  # length-sorted batches
    v = model.encode([docs[i]["text"] for i in order], batch_size=128, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    out = np.empty_like(v); out[order] = v
    q = read_tsv_queries(c)
    qv = model.encode([BGE_QUERY_PREFIX + t for t in q.text], batch_size=128, normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)
    np.save(E / f"{c}.queries.npy", qv)
    (E / f"{c}.queries.ids.txt").write_text("\n".join(q.qid) + "\n")
    (E / f"{c}.docs.ids.txt").write_text("\n".join(d["docid"] for d in docs) + "\n")
    np.save(E / f"{c}.docs.npy", out)
    print(f"{c}: docs {out.shape}, queries {qv.shape}", flush=True)
