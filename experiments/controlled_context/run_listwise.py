"""Run conditions.jsonl through a listwise reranker (rank_llm prompt/parsing,
offline vLLM batches, temperature 0).
models: zephyr = castorini/rank_zephyr_7b_v1_full (original run), first = castorini/first_mistral (single-token logits,
same settings as in Experiment 1), qwen = Qwen/Qwen2.5-7B-Instruct zero-shot with rank_llm's RankGPT prompt (not used
in the reported analysis).
out: controlled/rank_out.jsonl (zephyr) or controlled/models/rank_out_<model>.jsonl
     (call_id, ranking[20 docids], response, prompt_tokens, malformed), appended per chunk of 500 calls;
finished call_ids are skipped on restart. malformed: zephyr/qwen = identifiers not exactly 1..20;
first = identifiers missing from the top-30 logprobs (rank_llm fills them in). Calls are processed in a seeded random
order so that replicate calls (cond R) land in different batches than their base call; C5 permutations 4-5 run last.
usage: run_listwise.py [--model zephyr|first|qwen] [--limit N] [--out FILE] [--fewshot]  (run under setsid; .venv_llm)"""
import sys, os, json, re, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from config import COMMON, EXP, SEED, mem_watchdog
import numpy as np

P1 = EXP / "controlled"
MODELS = {"zephyr": "castorini/rank_zephyr_7b_v1_full", "first": "castorini/first_mistral",
          "qwen": "Qwen/Qwen2.5-7B-Instruct"}
CHUNK = 500
MIN_GB = 24
# qwen pilot: 169/217 malformed outputs were cut at rank_llm's 90-token cap after the 19th id -> 30 extra tokens
QWEN_EXTRA_TOKENS = 30
# one worked format example for qwen, used only if its parse failure rate exceeds 5% (pre-specified fallback)
FEWSHOT_NOTE = (" Example of the required format for 3 passages: [2] > [3] > [1]. "
                "Your answer must list all {num} identifiers exactly once.")


def build_reranker(model, fewshot=False):
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
    tpl = files("rank_llm.rerank.prompt_templates")
    if model == "zephyr":
        return RankListwiseOSLLM(model=MODELS[model], context_size=4096, window_size=20, stride=10)
    if model == "first":
        return RankListwiseOSLLM(
            model=MODELS[model], context_size=8192, window_size=20, stride=10, variable_passages=True,
            prompt_template_path=tpl / "rank_zephyr_alpha_template.yaml",
            system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
            use_alpha=True, use_logits=True)
    # qwen: same 4096 context (hence the same passage truncation) as zephyr
    r = RankListwiseOSLLM(model=MODELS[model], context_size=4096, window_size=20, stride=10,
                          prompt_template_path=tpl / "rank_gpt_template.yaml")
    if fewshot:
        h = r._inference_handler
        h.template["suffix_user"] = h.template["suffix_user"] + FEWSHOT_NOTE
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zephyr", choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fewshot", action="store_true", help="qwen only: add the one-example format note")
    ap.add_argument("--order-seed", type=int, default=SEED, help="call order within the selected set (tests only)")
    args = ap.parse_args()
    mem_watchdog()
    out = Path(args.out or (P1 / "rank_out.jsonl" if args.model == "zephyr" else P1 / "models" / f"rank_out_{args.model}.jsonl"))

    calls = [json.loads(l) for l in open(P1 / "conditions.jsonl")]
    order = np.random.default_rng(SEED).permutation(len(calls))
    calls = [calls[i] for i in order]
    # C5 permutations 4-5 go last, so the run can stop at C5 = 3 permutations if time runs out
    calls.sort(key=lambda c: c["cond"] == "C5" and c["perm"] >= 3)
    if args.limit:
        calls = calls[: args.limit]
    if args.order_seed != SEED:
        calls = [calls[i] for i in np.random.default_rng(args.order_seed).permutation(len(calls))]
    done = {json.loads(l)["call_id"] for l in open(out)} if out.exists() else set()
    todo = [c for c in calls if c["call_id"] not in done]
    print(f"{args.model}: {len(done)} done, {len(todo)} to run -> {out}", flush=True)
    if not todo:
        return

    text = {}
    for ds in ("dl19", "dl20"):
        for l in open(COMMON / "docs" / f"{ds}.jsonl"):
            d = json.loads(l)
            text[(ds, d["docid"])] = d["text"]
    qtext = {}
    for ds in ("dl19", "dl20"):
        for l in open(COMMON / "queries" / f"{ds}.tsv"):
            qid, t = l.rstrip("\n").split("\t", 1)
            qtext[f"{ds}:{qid}"] = t

    # rank_llm builds the prompt (template, 300-word truncation loop, chat template, ftfy) and parses the output;
    # generation runs as offline vLLM batches with rank_llm's own engine and sampling settings, because its async
    # engine path neither batches well here nor survives a second asyncio.run().
    import vllm
    from rank_llm.data import Result, Query, Candidate
    from host_resources import reserve, claim

    first = args.model == "first"
    reranker = build_reranker(args.model, args.fewshot)
    if first:
        sp = vllm.SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1, logprobs=30, seed=SEED)
    else:
        sp = vllm.SamplingParams(temperature=0.0, min_tokens=reranker.num_output_tokens(20),
                                 max_tokens=reranker._get_max_output_tokens(20) + QWEN_EXTRA_TOKENS * (args.model == "qwen"), seed=SEED)

    holder = f"rrt/controlled-{args.model}"
    with reserve(MIN_GB, holder) as util:
        print(f"vLLM gpu_memory_utilization={util:.3f}", flush=True)
        llm = vllm.LLM(model=MODELS[args.model], enforce_eager=False, max_logprobs=30, tensor_parallel_size=1,
                       gpu_memory_utilization=util, trust_remote_code=True, seed=SEED,
                       max_model_len=reranker._get_max_model_len(20))
    claim(holder, MIN_GB)

    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for s in range(0, len(todo), CHUNK):
            chunk = todo[s: s + CHUNK]
            results, prompts = [], []
            for c in chunk:
                ds = c["qid"].split(":")[0]
                cands = [Candidate(docid=d, score=float(20 - i), doc={"text": text[(ds, d)]})
                         for i, d in enumerate(c["doc_list"])]
                r = Result(query=Query(text=qtext[c["qid"]], qid=c["qid"]), candidates=cands)
                prompt, _ = reranker.create_prompt(r, 0, 20)
                results.append(r); prompts.append(prompt + "[" if first else prompt)
            gens = llm.generate(prompts, sp, use_tqdm=False)
            for c, r, g in zip(chunk, results, gens):
                if first:
                    resp, _ = reranker._evaluate_logits(g.outputs[0].logprobs[0], (1, 20))
                    malformed = len(set(re.findall(r"\[([A-Z])\]", resp))) != 20
                else:
                    resp = g.outputs[0].text
                    ids = [int(x) for x in re.findall(r"\[(\d+)\]", resp)]
                    malformed = sorted(ids) != list(range(1, 21))
                ranked = reranker.receive_permutation(r, resp, 0, 20)
                ranking = [x.docid for x in ranked.candidates]
                assert sorted(ranking) == sorted(c["doc_list"])
                f.write(json.dumps(dict(call_id=c["call_id"], ranking=ranking, response=resp,
                                        prompt_tokens=len(g.prompt_token_ids), malformed=malformed)) + "\n")
            f.flush()
            n = s + len(chunk)
            el = time.time() - t0
            print(f"{n}/{len(todo)} calls, {el/60:.1f} min, ETA {el/n*(len(todo)-n)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
