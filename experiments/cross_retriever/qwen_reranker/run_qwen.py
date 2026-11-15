"""Qwen3-Reranker-4B pointwise scores for evaluation queries x 3 retrievers (top-100 each).
usage: run_qwen.py sample   -> 100 evaluation queries (dataset share, largest remainder, seed-0 permutation), timing
       run_qwen.py rest     -> remaining evaluation queries
Prompt / yes-no scoring as in the Qwen3-Reranker model card (default instruction); document tokens are cut so the
prompt fits 8,192 tokens; bf16, length-sorted batches of 8. Writes scores/<phase>-<ds>.parquet, timing_<phase>.json."""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import EXP, LISTWISE_E, LISTWISE_F, QWEN, RETRIEVERS, SEED, mem_share, read_queries, read_run, start_watchdog

MODEL = 'Qwen/Qwen3-Reranker-4B'
MAXLEN, BATCH = 8192, 8
INSTR = 'Given a web search query, retrieve relevant passages that answer the query'
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
P, B = QWEN, LISTWISE_F
os.makedirs(P, exist_ok=True)


def eval_queries():
    E = pd.read_csv(f'{LISTWISE_E}/queries_mix.csv', dtype={'qid': str})[['dataset', 'qid']]
    F = pd.read_csv(f'{B}/queries_F.csv', dtype={'qid': str})[['dataset', 'qid']]  # final (complete) B set
    return sorted(set(map(tuple, E.values)) | set(map(tuple, F.values)))


def sample100(keys):
    byds = {}
    for d, q in keys:
        byds.setdefault(d, []).append(q)
    quota = {d: 100 * len(v) / len(keys) for d, v in byds.items()}
    k = {d: math.floor(x) for d, x in quota.items()}
    for d in sorted(quota, key=lambda d: quota[d] - k[d], reverse=True)[:100 - sum(k.values())]:
        k[d] += 1
    out = []
    for d, v in byds.items():
        v = sorted(v)
        out += [(d, v[i]) for i in np.random.default_rng(SEED).permutation(len(v))[:k[d]]]
    return sorted(out)


def main():
    phase = sys.argv[1]
    keys = eval_queries()
    s100 = sample100(keys)
    todo = s100 if phase == 'sample' else [k for k in keys if k not in set(s100)]
    pd.DataFrame(todo, columns=['dataset', 'qid']).to_csv(f'{P}/queries_{phase}.csv', index=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    start_watchdog()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    ms = mem_share()
    t_load = time.time()
    with ms.reserve(44, 'rrt/qwen3-reranker'):
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    ms.claim('rrt/qwen3-reranker', 44)
    # cap PyTorch at 44 GB of the 119 GB unified memory of the original host (batch 16 peaked at about 57 GB)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('RRT_TORCH_MEM_FRACTION', 44 / 119)))
    t_load = time.time() - t_load
    yes, no = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    pre, suf = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)
    os.makedirs(f'{P}/scores', exist_ok=True)
    t0, npairs = time.time(), 0
    for d in sorted({k[0] for k in todo}):
        qs = {q for dd, q in todo if dd == d}
        runs = {r: read_run(f'{EXP}/common/runs/{d}.{r}.top100.txt') for r in RETRIEVERS}
        pairs = sorted({(q, doc) for r in RETRIEVERS for q in qs for doc, _ in runs[r][q]})
        need = {doc for _, doc in pairs}
        text = {}
        for l in open(f'{EXP}/common/docs/{d}.jsonl'):
            x = json.loads(l)
            if x['docid'] in need:
                text[x['docid']] = x['text']
        qt = read_queries(d)
        ids = []
        for q, doc in pairs:
            head = tok.encode(f'<Instruct>: {INSTR}\n<Query>: {qt[q]}\n<Document>: ', add_special_tokens=False)
            body = tok.encode(text[doc], add_special_tokens=False)
            budget = max(MAXLEN - len(pre) - len(suf) - len(head), 0)
            ids.append(pre + head + body[:budget] + suf)
        order = np.argsort([len(x) for x in ids])
        sc = np.empty(len(ids), np.float32)
        with torch.inference_mode():
            for i in range(0, len(ids), BATCH):
                idx = order[i:i + BATCH]
                w = max(len(ids[j]) for j in idx)
                inp = torch.full((len(idx), w), tok.pad_token_id)
                att = torch.zeros_like(inp)
                for r_, j in enumerate(idx):
                    inp[r_, w - len(ids[j]):] = torch.tensor(ids[j])
                    att[r_, w - len(ids[j]):] = 1
                lg = model(input_ids=inp.cuda(), attention_mask=att.cuda()).logits[:, -1, :]
                two = torch.stack([lg[:, no], lg[:, yes]], 1).float()
                sc[idx] = torch.log_softmax(two, 1)[:, 1].cpu().numpy()
        pd.DataFrame({'qid': [q for q, _ in pairs], 'docid': [x for _, x in pairs], 'score': sc}).to_parquet(
            f'{P}/scores/{phase}-{d}.parquet', index=False)
        npairs += len(pairs)
        print(f'{phase} {d}: {len(qs)} queries, {len(pairs)} pairs, {time.time() - t0:.0f}s', flush=True)
    el = time.time() - t0
    json.dump(dict(phase=phase, n_queries=len(todo), n_pairs=npairs, score_sec=el, load_sec=t_load,
                   n_eval=len(keys)), open(f'{P}/timing_{phase}.json', 'w'))
    print('DONE', phase, f'{el:.0f}s', flush=True)


if __name__ == '__main__':
    main()
