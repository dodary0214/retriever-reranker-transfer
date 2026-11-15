"""Rerank job lists with RankZephyr-7B or FIRST-7B (rank_llm prompts and parsing, offline vLLM batches, temperature 0),
sliding window 20 / stride 10 from the bottom of the 100-candidate list.
--dir <dir>: reads <dir>/jobs.jsonl and appends to <dir>/out/<model>.jsonl (job_id, ranking, n_calls, n_bad, sec).
Lists are processed in chunks of about CHUNK calls; finished lists are skipped on restart.
n_bad = zephyr: responses whose identifiers are not exactly 1..w; first: identifiers missing from the top-30 logprobs.
The gate phase (dl19 bm25) is checked before anything else runs; a miss writes out/GATE_FAIL_<model> and exits 3.
Used for query set E (--dir $RRT_DATA/listwise_e; phases gate,dl,swap,beir1 then beir2), set F (--dir
$RRT_DATA/listwise_f/F --phases F) and the same-input reruns (--dir $RRT_DATA/listwise_f/rep or rep50 --phases rep).
usage: run_listwise.py --model zephyr|first --phases PHASES --dir DIR"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import EXP, SEED, read_queries, start_watchdog

MODELS = {'zephyr': 'castorini/rank_zephyr_7b_v1_full', 'first': 'castorini/first_mistral'}
GATE = {'zephyr': 0.74, 'first': 0.72}
TOL, CHUNK, MIN_GB, WIN, STRIDE = 0.03, 500, 24, 20, 10


def windows(n):
    out, end, start = [], n, max(n - WIN, 0)
    while True:
        out.append((start, end))
        if start == 0:
            return out
        end, start = end - STRIDE, max(start - STRIDE, 0)


def ndcg10(ranking, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:10]))  # trec_eval linear gain
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def build_reranker(model):
    import rank_llm.rerank.vllm_handler as vh
    from importlib.resources import files
    from transformers import AutoTokenizer
    from rank_llm.rerank.listwise import RankListwiseOSLLM

    class TokenizerOnlyHandler:  # prompts and parsing only; generation runs in our own vLLM engine
        def __init__(self, model, **kw):
            self._tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        def get_tokenizer(self):
            return self._tok
        def close(self):
            pass
    vh.VllmHandler = TokenizerOnlyHandler
    if model == 'zephyr':
        return RankListwiseOSLLM(model=MODELS[model], context_size=4096, window_size=WIN, stride=STRIDE)
    return RankListwiseOSLLM(
        model=MODELS[model], context_size=8192, window_size=WIN, stride=STRIDE, variable_passages=True,
        prompt_template_path=files('rank_llm.rerank.prompt_templates') / 'rank_zephyr_alpha_template.yaml',
        system_message='You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.',
        use_alpha=True, use_logits=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--phases', required=True)
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    phases = args.phases.split(',')
    P = os.path.abspath(args.dir)
    start_watchdog()
    os.makedirs(f'{P}/out', exist_ok=True)
    out = f'{P}/out/{args.model}.jsonl'
    if os.path.exists(f'{P}/out/GATE_FAIL_{args.model}'):
        sys.exit(f'gate failed earlier for {args.model}')
    done = {json.loads(l)['job_id'] for l in open(out)} if os.path.exists(out) else set()
    jobs = [j for j in map(json.loads, open(f'{P}/jobs.jsonl')) if j['phase'] in phases]
    todo = [j for j in jobs if j['job_id'] not in done]
    print(f'{args.model}: {len(jobs) - len(todo)} lists done, {len(todo)} to run '
          f'({sum(len(windows(len(j["docids"]))) for j in todo)} calls)', flush=True)

    def gate_check():
        res = {json.loads(l)['job_id']: json.loads(l)['ranking'] for l in open(out)}
        rel = {}
        for l in open(f'{EXP}/common/qrels/dl19.tsv'):
            q, d, g = l.split()
            rel.setdefault(q, {})[d] = int(g)
        gate = [j for j in map(json.loads, open(f'{P}/jobs.jsonl')) if j['phase'] == 'gate']
        v = sum(ndcg10(res[j['job_id']], rel.get(j['qid'], {})) for j in gate) / len(gate)
        ok = abs(v - GATE[args.model]) <= TOL
        print(f'GATE dl19 bm25+{args.model} nDCG@10 {v:.4f} (expected {GATE[args.model]} ± {TOL}): {"PASS" if ok else "FAIL"}',
              flush=True)
        if not ok:
            open(f'{P}/out/GATE_FAIL_{args.model}', 'w').write(f'{v:.4f}\n')
            sys.exit(3)

    if not todo:
        if 'gate' in phases:
            gate_check()
        print('LISTWISE_DONE', flush=True)
        return

    datasets = sorted({j['dataset'] for j in todo})
    text, qtext = {}, {}
    for ds in datasets:
        need = {d for j in todo if j['dataset'] == ds for d in j['docids']}
        for l in open(f'{EXP}/common/docs/{ds}.jsonl'):
            x = json.loads(l)
            if x['docid'] in need:
                text[(ds, x['docid'])] = x['text']
        qtext[ds] = read_queries(ds)

    import vllm
    from rank_llm.data import Candidate, Query, Result
    from host_resources import claim, reserve
    reranker = build_reranker(args.model)
    first = args.model == 'first'

    def params(w):
        if first:
            return vllm.SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1, logprobs=30, seed=SEED)
        return vllm.SamplingParams(temperature=0.0, min_tokens=reranker.num_output_tokens(w),
                                   max_tokens=reranker._get_max_output_tokens(w), seed=SEED)
    sp = {w: params(w) for w in range(1, WIN + 1)}

    holder = f'rrt/listwise-{args.model}'
    with reserve(MIN_GB, holder) as util:
        print(f'vLLM gpu_memory_utilization={util:.3f}', flush=True)
        llm = vllm.LLM(model=MODELS[args.model], enforce_eager=False, max_logprobs=30, tensor_parallel_size=1,
                       gpu_memory_utilization=util, trust_remote_code=True, seed=SEED,
                       max_model_len=reranker._get_max_model_len(WIN))
    claim(holder, MIN_GB)

    # chunks of lists worth about CHUNK calls; gate lists form their own chunk so the gate is checked first
    chunks, cur, n = [], [], 0
    for j in todo:
        if cur and (n >= CHUNK or (cur[-1]['phase'] == 'gate') != (j['phase'] == 'gate')):
            chunks.append(cur); cur, n = [], 0
        cur.append(j); n += len(windows(len(j['docids'])))
    chunks.append(cur)

    t0, calls_done, total = time.time(), 0, sum(len(windows(len(j['docids']))) for j in todo)
    for ci, chunk in enumerate(chunks):
        tc = time.time()
        res = [Result(query=Query(text=qtext[j['dataset']][j['qid']], qid=j['qid']),
                      candidates=[Candidate(docid=d, score=float(len(j['docids']) - i), doc={'text': text[(j['dataset'], d)]})
                                  for i, d in enumerate(j['docids'])]) for j in chunk]
        wins = [windows(len(j['docids'])) for j in chunk]
        bad = [0] * len(chunk)
        for step in range(max(map(len, wins))):
            act = [i for i in range(len(chunk)) if step < len(wins[i])]
            prompts, ps = [], []
            for i in act:
                s, e = wins[i][step]
                prompt, _ = reranker.create_prompt(res[i], s, e)
                prompts.append(prompt + '[' if first else prompt)
                ps.append(sp[e - s])
            gens = llm.generate(prompts, ps, use_tqdm=False)
            for i, g in zip(act, gens):
                s, e = wins[i][step]
                if first:
                    perm, _ = reranker._evaluate_logits(g.outputs[0].logprobs[0], (1, e - s))
                    bad[i] += (e - s) - len(re.findall(r'\[([A-Z])\]', perm))
                else:
                    perm = g.outputs[0].text
                    ids = [int(x) for x in re.findall(r'\[(\d+)\]', perm)]
                    bad[i] += sorted(ids) != list(range(1, e - s + 1))
                before = sorted(c.docid for c in res[i].candidates)
                res[i] = reranker.receive_permutation(res[i], perm, s, e)
                assert sorted(c.docid for c in res[i].candidates) == before
            calls_done += len(act)
        with open(out, 'a') as f:
            for j, r, w, b in zip(chunk, res, wins, bad):
                f.write(json.dumps(dict(job_id=j['job_id'], ranking=[c.docid for c in r.candidates], n_calls=len(w),
                                        n_bad=b, sec=round((time.time() - tc) / len(chunk), 2))) + '\n')
        el = time.time() - t0
        print(f'chunk {ci + 1}/{len(chunks)} [{chunk[0]["phase"]}] {calls_done}/{total} calls, {el / 60:.1f} min, '
              f'ETA {el / calls_done * (total - calls_done) / 60:.0f} min', flush=True)
        if chunk[0]['phase'] == 'gate':
            gate_check()
    if 'gate' in phases and not any(c[0]['phase'] == 'gate' for c in chunks):
        gate_check()
    print('LISTWISE_DONE', flush=True)


if __name__ == '__main__':
    main()
