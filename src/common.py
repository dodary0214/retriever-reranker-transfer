"""Shared constants, paths and helpers for the cross-retriever experiment (Experiment 1) and the analysis scripts.

Paths are taken from environment variables (defaults are relative to the repository root):
  RRT_DATA                 Experiment 1 data root (see docs/data_layout.md)
  RRT_CONTROLLED_DATA      Experiment 2 data root (used by the analysis to read the Experiment 2 call list)
  RRT_OUT                  output root of the analysis scripts (tables/, figures/, derived/, controlled_context/)
  RRT_CONTROLLED_RESULTS   Experiment 2 analysis outputs (default: $RRT_OUT/controlled_context)
"""
import datetime
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.environ.get('RRT_DATA', os.path.join(REPO, 'data', 'cross_retriever'))
CONTROLLED_DATA = os.environ.get('RRT_CONTROLLED_DATA', os.path.join(REPO, 'data', 'controlled_context'))
OUT = os.environ.get('RRT_OUT', os.path.join(REPO, 'outputs'))
CONTROLLED_RESULTS = os.environ.get('RRT_CONTROLLED_RESULTS', os.path.join(OUT, 'controlled_context'))
POINTWISE = f'{EXP}/pointwise'          # scores/, runs/, grid.parquet
LISTWISE_E = f'{EXP}/listwise_e'        # 547-query set E: queries_mix.csv, jobs.jsonl, out/, runs/, grid_mix.parquet
LISTWISE_F = f'{EXP}/listwise_f'        # 742-query set F and same-input reruns: queries_F.csv, F/, rep/, rep50/, runs/
QWEN = f'{EXP}/qwen_reranker'           # Qwen3-Reranker-4B scores (robustness analysis)
SEED = int(os.environ.get('SEED', 0))
DEPTH = 100

DATASETS = {
    'dl19': {'irds': 'msmarco-passage/trec-dl-2019/judged', 'pyserini': 'msmarco-v1-passage'},
    'dl20': {'irds': 'msmarco-passage/trec-dl-2020/judged', 'pyserini': 'msmarco-v1-passage'},
    'nfcorpus': {'irds': 'beir/nfcorpus/test', 'pyserini': 'beir-v1.0.0-nfcorpus'},
    'fiqa': {'irds': 'beir/fiqa/test', 'pyserini': 'beir-v1.0.0-fiqa'},
    'scifact': {'irds': 'beir/scifact/test', 'pyserini': 'beir-v1.0.0-scifact'},
    'dbpedia': {'irds': 'beir/dbpedia-entity/test', 'pyserini': 'beir-v1.0.0-dbpedia-entity'},
}
RETRIEVERS = ['bm25', 'splade', 'dense']
BGE = 'BAAI/bge-base-en-v1.5'
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '


def index_name(ds, retriever):
    base = DATASETS[ds]['pyserini']
    if retriever == 'bm25':
        return base if base.startswith('msmarco') else f'{base}.flat'
    return f"{base}.{'splade-pp-ed' if retriever == 'splade' else 'bge-base-en-v1.5'}"


def read_queries(ds):
    out = {}
    with open(f'{EXP}/common/queries/{ds}.tsv') as f:
        for line in f:
            qid, text = line.rstrip('\n').split('\t', 1)
            out[qid] = text
    return out


def update_manifest(patch):
    path = f'{EXP}/common/manifest.json'
    m = json.load(open(path)) if os.path.exists(path) else {}
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(m.get(k), dict):
            m[k].update(v)
        else:
            m[k] = v
    m['seed'] = SEED
    m['depth'] = DEPTH
    m['updated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    json.dump(m, open(path, 'w'), indent=2, ensure_ascii=False)


def start_watchdog(min_avail_gb=float(os.environ.get('RRT_MIN_AVAIL_GB', 12)), interval_s=5):
    """Abort this process group if MemAvailable drops below min_avail_gb (protects hosts with unified CPU/GPU memory)."""
    import signal
    import threading
    import time

    def avail_gb():
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) / 2**20

    def loop():
        while True:
            a = avail_gb()
            if a < min_avail_gb:
                print(f'[watchdog] MemAvailable {a:.1f} GB < {min_avail_gb} GB, aborting', flush=True)
                os.killpg(os.getpgid(0), signal.SIGKILL)
            time.sleep(interval_s)

    threading.Thread(target=loop, daemon=True).start()


def mem_share():
    """GPU memory coordinator with reserve(min_gb, holder) and claim(holder, gb); see host_resources.py."""
    import host_resources
    return host_resources


def write_run(path, results, tag):
    """results: {qid: [(docid, score), ...]} sorted by score desc."""
    with open(path, 'w') as f:
        for qid, hits in results.items():
            for rank, (docid, score) in enumerate(hits, 1):
                f.write(f'{qid} Q0 {docid} {rank} {score:.6f} {tag}\n')


def read_run(path):
    out = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            out.setdefault(qid, []).append((docid, float(score)))
    return out
