"""Control reranker bge-reranker-v2-m3 on the same conditions.jsonl.
Each call's 20 (query, doc) pairs are scored as one batch, independently of each other (fp32, max_length 512).
out: controlled/ce_out.jsonl (call_id, ranking[20 docids], scores[20, slot order]); resumable."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from config import COMMON, EXP, SEED, mem_watchdog
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

P1 = EXP / "controlled"
MODEL = "BAAI/bge-reranker-v2-m3"


def main():
    mem_watchdog()
    torch.manual_seed(SEED)
    out = P1 / "ce_out.jsonl"
    done = {json.loads(l)["call_id"] for l in open(out)} if out.exists() else set()
    calls = [c for c in map(json.loads, open(P1 / "conditions.jsonl")) if c["call_id"] not in done]
    print(f"{len(done)} done, {len(calls)} to run", flush=True)
    text, qtext = {}, {}
    for ds in ("dl19", "dl20"):
        for l in open(COMMON / "docs" / f"{ds}.jsonl"):
            d = json.loads(l)
            text[(ds, d["docid"])] = d["text"]
        for l in open(COMMON / "queries" / f"{ds}.tsv"):
            qid, t = l.rstrip("\n").split("\t", 1)
            qtext[f"{ds}:{qid}"] = t
    from host_resources import reserve
    with reserve(8, "rrt/bge-reranker-v2-m3"):
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()
    t0 = time.time()
    with open(out, "a") as f, torch.inference_mode():
        for i, c in enumerate(calls, 1):
            ds = c["qid"].split(":")[0]
            q = qtext[c["qid"]]
            enc = tok([q] * 20, [text[(ds, d)] for d in c["doc_list"]], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to("cuda")
            s = model(**enc).logits.view(-1).float().cpu().tolist()
            ranking = [d for _, d in sorted(zip(s, c["doc_list"]), key=lambda x: -x[0])]
            f.write(json.dumps(dict(call_id=c["call_id"], ranking=ranking, scores=s)) + "\n")
            if i % 2000 == 0:
                f.flush()
                el = time.time() - t0
                print(f"{i}/{len(calls)} calls, {el/60:.1f} min, ETA {el/i*(len(calls)-i)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
