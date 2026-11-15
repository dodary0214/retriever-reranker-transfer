# Environment

The reported results were produced on one host with an NVIDIA GB10 (aarch64, 119 GB unified CPU/GPU memory,
CUDA 13.0, driver 580). Two Python 3.12 environments were used.

| Environment | Used for | Key packages |
|---|---|---|
| `requirements.txt` | data preparation, first-stage retrieval, pointwise reranking, Qwen3-Reranker-4B scores, Experiment 2 construction and control reranker, all analyses | numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, pyarrow 26.0.0, ir_measures 0.4.3, ir_datasets 0.6.3, pyserini 2.4.0 (Java 21), faiss-cpu 1.15.1, torch 2.14.1 (cu130), transformers 5.21.0, sentence-transformers 6.2.0 |
| `requirements-llm.txt` | listwise reranking with RankZephyr-7B and FIRST-7B (both experiments) | vllm 0.26.0, rank_llm 0.26.0, torch 2.11.0 (cu130), transformers 5.12.1 |

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python3.12 -m venv .venv-llm && . .venv-llm/bin/activate && pip install -r requirements-llm.txt
```
The CPU analysis (`scripts/run_analysis.sh`) needs only the first environment; torch, transformers,
sentence-transformers, pyserini and faiss are not imported by the analysis scripts.

## Settings that are fixed in the code

| Setting | Value | Where |
|---|---|---|
| Random seed | `SEED=0` (environment variable, default 0) for sampling, bootstrap and vLLM | `src/common.py`, `experiments/controlled_context/data/config.py` |
| Candidate depth | top 100 per retriever; BEIR runs drop `docid == qid` | `03_retrieve.py` |
| BM25 | k1 = 0.9, b = 0.4 | `03_retrieve.py` |
| Pointwise reranking | max 512 tokens (document cut), batch 64, fp16 (monoT5 bf16) | `run_rerankers.py` |
| Listwise reranking | window 20, stride 10 from the bottom; temperature 0; RankZephyr context 4,096; FIRST context 8,192, single-token logits (top-30 logprobs) | `experiments/cross_retriever/listwise/run_listwise.py` |
| Experiment 2 calls | one window of 20 documents per call; target pair in slots 3 and 12 | `experiments/controlled_context/` |
| Tie threshold for per-query preferences | θ = 0.01 nDCG@10 (sensitivity 0.005, 0.02) | `analysis/analyze.py` (`TH`) |
| Bootstrap | 10,000 query resamples (`NB`), percentile 95% intervals, `numpy.random.default_rng(0)` | `analysis/analyze.py` |
| Experiment 2 model tables | 1,000 query resamples; the reported Experiment 2 intervals (`figures/fig4_values.csv`) use 10,000 | `experiments/controlled_context/analyze.py`, `analysis/analyze.py` |
| Multiple comparisons, Experiment 2 | Bonferroni over the five conditions C1–C5 per model (two-sided 99% intervals) | `analysis/analyze.py` |

## GPU memory

On the original host, jobs reserved memory through a host-level manager. In this repository that manager is optional
(`src/host_resources.py`): set `RRT_MEM_SHARE_DIR` to its directory to use it, otherwise the vLLM memory fraction is
taken from `RRT_GPU_MEM_UTIL` (default 0.9). The fractions used in the reported runs, from the run logs:

| Run | `gpu_memory_utilization` |
|---|---|
| Experiment 1, set E, RankZephyr-7B | 0.249, 0.298 (two launches) |
| Experiment 1, set E, FIRST-7B | 0.247, 0.253 |
| Experiment 1, set F and same-input reruns | 0.40 |
| Experiment 2, RankZephyr-7B and FIRST-7B | 0.40 |

`RRT_MIN_AVAIL_GB` (default 12) sets the free-memory threshold below which a run aborts itself, and
`RRT_TORCH_MEM_FRACTION` caps PyTorch memory for the Qwen3-Reranker-4B scorer (default 44/119).

## Run-to-run variation

Listwise outputs at temperature 0 were not bit-identical across engine sessions for RankZephyr-7B: in a same-input
rerun of 50 set-E queries (150 lists) in a new session, 36 lists were identical and 1.2% of the relevant–non-relevant
pairs changed order (`results/summary/tables/appendixD_same_input_flips.csv`). FIRST-7B reproduced all 30 rerun lists
(10 queries) exactly. New GPU
runs are therefore expected to reproduce the listwise outputs only up to this variation; the analysis results in
`results/summary/` are reproduced exactly from the stored outputs.
