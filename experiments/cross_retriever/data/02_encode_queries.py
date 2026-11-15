"""Encode all queries with bge-base-en-v1.5 (query prefix, L2-normalised) -> emb/<ds>.queries.{npy,ids.txt}.
These vectors are also the query side of the dense retriever."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import BGE, BGE_QUERY_PREFIX, DATASETS, EXP, SEED, mem_share, read_queries, start_watchdog

torch.manual_seed(SEED)
start_watchdog()
ms = mem_share()
with ms.reserve(16, 'rrt/bge-query-encode'):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE, device='cuda')
for ds in DATASETS:
    q = read_queries(ds)
    ids = list(q)
    emb = model.encode([BGE_QUERY_PREFIX + q[i] for i in ids], batch_size=128,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(f'{EXP}/common/emb/{ds}.queries.npy', emb)
    open(f'{EXP}/common/emb/{ds}.queries.ids.txt', 'w').write('\n'.join(ids) + '\n')
    print(ds, emb.shape, flush=True)
