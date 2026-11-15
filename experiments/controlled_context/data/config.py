"""Shared config for the controlled candidate-context experiment (Experiment 2). Every script reads SEED and paths
from here. Data root: $RRT_CONTROLLED_DATA (default <repo>/data/controlled_context) with common/ (first-stage data)
and controlled/ (pairs, conditions, model outputs)."""
import os
from pathlib import Path

SEED = int(os.environ.get("SEED", 0))
REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("RRT_CONTROLLED_DATA", REPO / "data" / "controlled_context"))
EXP = ROOT
COMMON = EXP / "common"
DEPTH = 100

# code -> (ir_datasets id, pyserini corpus key)
DATASETS = {
    "dl19": ("msmarco-passage/trec-dl-2019/judged", "msmarco-v1-passage"),
    "dl20": ("msmarco-passage/trec-dl-2020/judged", "msmarco-v1-passage"),
    "nfcorpus": ("beir/nfcorpus/test", "beir-v1.0.0-nfcorpus"),
    "fiqa": ("beir/fiqa/test", "beir-v1.0.0-fiqa"),
    "scifact": ("beir/scifact/test", "beir-v1.0.0-scifact"),
    "dbpedia": ("beir/dbpedia-entity/test", "beir-v1.0.0-dbpedia-entity"),
}
RETRIEVERS = ["bm25", "splade", "dense"]

def bm25_index(corpus):
    return corpus if corpus == "msmarco-v1-passage" else f"{corpus}.flat"

def splade_index(corpus):
    return f"{corpus}.splade-pp-ed"

def dense_index(corpus):
    return f"{corpus}.bge-base-en-v1.5"

BGE = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SPLADE_ENCODER = "naver/splade-cocondenser-ensembledistil"
TOKENIZER = "bert-base-uncased"

import sys, threading, time, signal
sys.path.insert(0, str(REPO / "src"))  # host_resources


def mem_watchdog(guard_gb=float(os.environ.get("RRT_MIN_AVAIL_GB", 12)), every_s=5):
    """Abort own process group when MemAvailable drops below guard_gb (protects hosts with unified CPU/GPU memory)."""

    def avail_gb():
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 2**20

    def loop():
        while True:
            if avail_gb() < guard_gb:
                print(f"[watchdog] MemAvailable < {guard_gb} GB, aborting", flush=True)
                os.killpg(os.getpgid(0), signal.SIGKILL)
            time.sleep(every_s)
    threading.Thread(target=loop, daemon=True).start()


def read_tsv_queries(code):
    import pandas as pd
    return pd.read_csv(COMMON / "queries" / f"{code}.tsv", sep="\t", header=None, names=["qid", "text"],
                       dtype=str, keep_default_na=False, quoting=3)


def write_run(path, results, tag):
    """results: {qid: [(docid, score), ...]} already sorted."""
    with open(path, "w") as f:
        for qid, hits in results.items():
            for r, (d, s) in enumerate(hits, 1):
                f.write(f"{qid} Q0 {d} {r} {s:.6f} {tag}\n")
