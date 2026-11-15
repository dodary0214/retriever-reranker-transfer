"""Rescore each dataset's top-100 candidates with 4 rerankers.

Each unique (qid, docid) pair over the three retriever runs is scored once (cross-encoders score pairs independently),
cached in scores/<ds>.<reranker>/part-*.parquet (resumable checkpoints), then written out as
runs/<ds>.<retriever>.<reranker>.txt for every retriever.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import DATASETS, EXP, POINTWISE, RETRIEVERS, SEED, mem_share, read_queries, read_run, start_watchdog, write_run

RERANKERS = {
    'minilm': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'bge-base': 'BAAI/bge-reranker-base',
    'bge-m3': 'BAAI/bge-reranker-v2-m3',
    'monot5': 'castorini/monot5-base-msmarco-10k',
}
MAX_LEN, BATCH, PART = 512, 64, 16384
P2 = POINTWISE


class CrossEncoderScorer:
    def __init__(self, name):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def __call__(self, queries, docs):
        enc = self.tok(queries, docs, truncation='only_second', max_length=MAX_LEN, padding=True,
                       return_tensors='pt').to('cuda')
        return self.model(**enc).logits[:, 0].float().cpu().numpy()


class MonoT5Scorer:
    def __init__(self, name):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = T5ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
        self.true_id, self.false_id = self.tok.convert_tokens_to_ids(['▁true', '▁false'])
        self.suffix = self.tok.encode(' Relevant:', add_special_tokens=False) + [self.tok.eos_token_id]

    @torch.inference_mode()
    def __call__(self, queries, docs):
        ids = []
        for q, d in zip(queries, docs):
            head = self.tok.encode(f'Query: {q} Document:', add_special_tokens=False)
            body = self.tok.encode(d, add_special_tokens=False)
            budget = max(MAX_LEN - len(head) - len(self.suffix), 0)
            ids.append((head + body[:budget] + self.suffix)[:MAX_LEN])
        width = max(map(len, ids))
        input_ids = torch.full((len(ids), width), self.tok.pad_token_id)
        mask = torch.zeros_like(input_ids)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        input_ids, mask = input_ids.cuda(), mask.cuda()
        dec = torch.full((len(ids), 1), self.model.config.decoder_start_token_id, device='cuda')
        logits = self.model(input_ids=input_ids, attention_mask=mask, decoder_input_ids=dec).logits[:, 0]
        tf = torch.stack([logits[:, self.true_id], logits[:, self.false_id]], 1).float()
        return torch.log_softmax(tf, 1)[:, 0].cpu().numpy()


def unique_pairs(ds):
    runs = {r: read_run(f'{EXP}/common/runs/{ds}.{r}.top100.txt') for r in RETRIEVERS}
    pairs = sorted({(q, d) for run in runs.values() for q, hs in run.items() for d, _ in hs})
    return runs, pairs


def score_dataset(ds, rr, scorer):
    out_dir = f'{P2}/scores/{ds}.{rr}'
    os.makedirs(out_dir, exist_ok=True)
    runs, pairs = unique_pairs(ds)
    queries = read_queries(ds)
    docs = {}
    for line in open(f'{EXP}/common/docs/{ds}.jsonl'):
        x = json.loads(line)
        docs[x['docid']] = (x['text'], x['n_tokens'])
    n_parts = (len(pairs) + PART - 1) // PART
    for p in range(n_parts):
        path = f'{out_dir}/part-{p:04d}.parquet'
        if os.path.exists(path):
            continue
        chunk = pairs[p * PART:(p + 1) * PART]
        order = np.argsort([docs[d][1] for _, d in chunk])  # length-sorted batches; restored below
        scores = np.empty(len(chunk), dtype=np.float32)
        for i in range(0, len(chunk), BATCH):
            idx = order[i:i + BATCH]
            scores[idx] = scorer([queries[chunk[j][0]] for j in idx], [docs[chunk[j][1]][0] for j in idx])
        pd.DataFrame({'qid': [q for q, _ in chunk], 'docid': [d for _, d in chunk], 'score': scores}) \
            .to_parquet(path + '.tmp', index=False)
        os.replace(path + '.tmp', path)
        print(f'{ds} {rr} part {p + 1}/{n_parts} {time.strftime("%H:%M:%S")}', flush=True)
    s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{out_dir}/part-*.parquet'))])
    assert len(s) == len(pairs), (ds, rr, len(s), len(pairs))
    lookup = dict(zip(zip(s.qid, s.docid), s.score.astype(float)))
    for r, run in runs.items():
        res = {q: sorted(((d, lookup[(q, d)]) for d, _ in hs), key=lambda x: (-x[1], x[0])) for q, hs in run.items()}
        write_run(f'{P2}/runs/{ds}.{r}.{rr}.txt', res, rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rerankers', nargs='*', default=list(RERANKERS))
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
    args = ap.parse_args()
    torch.manual_seed(SEED)
    start_watchdog()
    for rr in args.rerankers:
        todo = [ds for ds in args.datasets
                if not all(os.path.exists(f'{P2}/runs/{ds}.{r}.{rr}.txt') for r in RETRIEVERS)]
        if not todo:
            continue
        with mem_share().reserve(16, f'rrt/pointwise-{rr}'):  # one model at a time
            scorer = (MonoT5Scorer if rr == 'monot5' else CrossEncoderScorer)(RERANKERS[rr])
        for ds in todo:
            score_dataset(ds, rr, scorer)
        del scorer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
