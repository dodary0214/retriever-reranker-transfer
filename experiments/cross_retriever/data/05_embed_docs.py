"""Encode docs/<ds>.jsonl text with bge-base-en-v1.5 (no prefix, L2-normalised, max 512 tokens)
-> emb/<ds>.docs.npy (float32, row i = line i of docs jsonl) + emb/<ds>.docs.ids.txt. Skips finished datasets."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import BGE, DATASETS, EXP, SEED, mem_share, start_watchdog, update_manifest

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
args = ap.parse_args()
torch.manual_seed(SEED)
start_watchdog()
todo = [ds for ds in args.datasets if not os.path.exists(f'{EXP}/common/emb/{ds}.docs.npy')]
if todo:
    with mem_share().reserve(16, 'rrt/bge-doc-encode'):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(BGE, device='cuda')
        model.max_seq_length = 512
    for ds in todo:
        rows = [json.loads(l) for l in open(f'{EXP}/common/docs/{ds}.jsonl')]
        # length-sorted batches for speed, restored to file order afterwards
        order = np.argsort([r['n_tokens'] for r in rows])
        emb = model.encode([rows[i]['text'] for i in order], batch_size=128, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        out = np.empty_like(emb)
        out[order] = emb
        np.save(f'{EXP}/common/emb/{ds}.docs.npy', out)
        open(f'{EXP}/common/emb/{ds}.docs.ids.txt', 'w').write('\n'.join(r['docid'] for r in rows) + '\n')
        update_manifest({'emb': {ds: {'n_docs': len(rows), 'dim': int(out.shape[1]), 'model': BGE}}})
        print(ds, out.shape, flush=True)
