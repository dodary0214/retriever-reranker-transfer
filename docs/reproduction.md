# Reproduction

Commands are run from the repository root. `SEED=0` is the default everywhere.

```bash
export RRT_DATA=$PWD/data/cross_retriever          # Experiment 1 data root
export RRT_CONTROLLED_DATA=$PWD/data/controlled_context   # Experiment 2 data root
export RRT_OUT=$PWD/outputs                          # analysis outputs
```

There are three levels of reproduction.

| Level | Needs | Reproduces |
|---|---|---|
| A. Analysis | stored model outputs (see README) + benchmark qrels, CPU only | every value in `results/summary/`, exactly |
| B. Inputs | benchmark data, CPU | query sets E and F, job lists, Experiment 2 pairs and conditions, pointwise grid, listwise runs (from stored outputs) |
| C. Model runs | benchmark data, GPU (about 24 GB for each 7B listwise model) | the stored model outputs, up to run-to-run variation of the listwise models (docs/environment.md) |

## A. Analysis from stored outputs (CPU)

1. Place the stored outputs under `$RRT_DATA` and `$RRT_CONTROLLED_DATA` (layout in `docs/data_layout.md`) and create
   the qrels and query files with step C1 (`01_qrels_queries.py` for Experiment 1; it needs no GPU).
2. Run the analysis and compare it with the reference values:

```bash
bash scripts/run_analysis.sh            # about 5 minutes on 20 CPU cores
python scripts/compare_outputs.py       # compares $RRT_OUT with results/summary/
```

`run_analysis.sh` runs, in this order:

| Step | Script | Output (`$RRT_OUT/...`) |
|---|---|---|
| Experiment 2 per model | `experiments/controlled_context/analyze.py --c5-perms 3` | `controlled_context/table*_<model>.csv`, `per_unit_<model>.parquet` |
| Experiment 2 comparison | `experiments/controlled_context/compare_models.py` | `controlled_context/m1_reversal_by_model.csv`, ... |
| Main analysis | `analysis/analyze.py` | `tables/`, `figures/*_values.csv`, `RESULTS.md` |
| Qwen3-Reranker-4B rows | `analysis/robustness/appendix_qwen.py` | `derived/qwen_*.csv` |
| 15 four-reranker sets, k groups | `analysis/query_agreement/ksets.py` | `derived/ksets_*.csv` |
| Within-type Spearman | `analysis/query_agreement/fig3_spearman.py` | `derived/fig3_spearman.csv` |
| Robustness by k group | `analysis/robustness/ksets_robust.py` | `derived/ksets_robust_*.csv` |
| Raw nDCG@10 of selection | `analysis/evaluation_transfer/ksets_selection_ndcg.py` | `derived/ksets_selection_ndcg.csv` |
| Judged-grade-0 flips, extra κ rows | `analysis/robustness/appendixB_extra.py` | `derived/appendixB_extra_*.csv` |
| Rerun window diagnostic | `analysis/pair_reversals/diag_rerun_windows.py` | `derived/diag_rerun_windows.csv` |
| Matched set swaps (one pointwise reranker replaced by a listwise one; own and common untied units) | `analysis/query_agreement/matched_swap.py` | `derived/matched_swap_*.csv` |

The scripts under `analysis/*/` execute the definitions of `analysis/analyze.py` up to a marked line (data loading,
metrics, bootstrap helpers) and then compute their own quantities, so `analysis/analyze.py` must not be moved.

## B. Inputs from benchmark data (CPU)

```bash
# query set E and its job list (the original job file also had 711 "swap" lists that the analysis does not use)
python experiments/cross_retriever/listwise/build_jobs_e.py
# same-input rerun lists and query set F
python experiments/cross_retriever/listwise/build_jobs_f.py rep
python experiments/cross_retriever/listwise/build_jobs_f.py rep50
python experiments/cross_retriever/listwise/build_jobs_f.py Fn 742
# Experiment 2 pairs and conditions (needs the Experiment 2 document vectors, see docs/data_layout.md)
python experiments/controlled_context/build_pairs.py
python experiments/controlled_context/build_conditions.py
```

## C. Full pipeline

### C1. Experiment 1: first-stage retrieval

```bash
mkdir -p $RRT_DATA/common/{qrels,queries,runs,docs,emb}
python experiments/cross_retriever/data/00_download_indexes.py   # Pyserini prebuilt indexes (about 70 GB)
python experiments/cross_retriever/data/01_qrels_queries.py
python experiments/cross_retriever/data/02_encode_queries.py      # GPU
python experiments/cross_retriever/data/03_retrieve.py bm25
python experiments/cross_retriever/data/03_retrieve.py splade
python experiments/cross_retriever/data/03_retrieve.py dense
python experiments/cross_retriever/data/04_docs.py
python experiments/cross_retriever/data/06_validate.py            # nDCG@10 against Pyserini references
```
`00b_download_msmarco.py` fetches the MS MARCO indexes from an alternative Pyserini mirror if the default source is
slow. `05_embed_docs.py` (document vectors) is not needed by later steps of Experiment 1.

### C2. Experiment 1: pointwise reranking

```bash
mkdir -p $RRT_DATA/pointwise/runs
python experiments/cross_retriever/pointwise/run_rerankers.py    # GPU; resumable
python experiments/cross_retriever/pointwise/build_grid.py       # grid.parquet + sanity gate
```

### C3. Experiment 1: listwise reranking (environment of `requirements-llm.txt`)

```bash
L=experiments/cross_retriever/listwise
python $L/build_jobs_e.py
# set E, in the original order
python $L/run_listwise.py --model zephyr --phases gate,dl,swap,beir1 --dir $RRT_DATA/listwise_e
python $L/run_listwise.py --model first  --phases gate,dl,swap,beir1 --dir $RRT_DATA/listwise_e
python $L/run_listwise.py --model zephyr --phases beir2 --dir $RRT_DATA/listwise_e
python $L/run_listwise.py --model first  --phases beir2 --dir $RRT_DATA/listwise_e
python $L/build_grid_e.py
# same-input reruns (zephyr and first on rep; zephyr on rep50)
python $L/build_jobs_f.py rep && python $L/build_jobs_f.py rep50
for m in zephyr first; do python $L/run_listwise.py --model $m --phases rep --dir $RRT_DATA/listwise_f/rep; done
python $L/run_listwise.py --model zephyr --phases rep --dir $RRT_DATA/listwise_f/rep50
python $L/compare_rep.py; python $L/diag_rep.py      # rerun vs original (FAIL = lists differ, expected for RankZephyr)
# set F
python $L/build_jobs_f.py Fn 742
cp $RRT_DATA/listwise_f/queries_F.csv $RRT_DATA/listwise_f/queries_F.planned.csv
for m in zephyr first; do python $L/run_listwise.py --model $m --phases F --dir $RRT_DATA/listwise_f/F; done
python $L/finalize_f.py && touch $RRT_DATA/listwise_f/F/out/B_DONE && python $L/make_runs_f.py
```
A run that fails the dl19 BM25 gate (nDCG@10 outside 0.74 ± 0.03 for RankZephyr, 0.72 ± 0.03 for FIRST) stops with
exit code 3. In the original run, set F was reached in steps (1,058 → 788 → 742 planned queries) as the measured time per
query became known; because F is drawn by a fixed permutation, the final 742 queries are the first 742 of that order, and
all 742 have outputs from both models.

### C4. Experiment 1: Qwen3-Reranker-4B (robustness only)

```bash
python experiments/cross_retriever/qwen_reranker/run_qwen.py sample   # 100 evaluation queries, timing
python experiments/cross_retriever/qwen_reranker/run_qwen.py rest     # the remaining 1,189
```

### C5. Experiment 2

```bash
mkdir -p $RRT_CONTROLLED_DATA/common/{qrels,queries,runs,docs,emb}
D=experiments/controlled_context/data
python $D/00_download.py && python $D/01_qrels_queries.py
python $D/02_retrieve.py bm25 dl19 dl20
python $D/03_docs.py dl19 dl20
python $D/04_embed.py dl19 dl20                      # GPU; vectors for C3/C4
python experiments/controlled_context/build_pairs.py
python experiments/controlled_context/build_conditions.py
python experiments/controlled_context/run_ce.py      # control reranker (bge-reranker-v2-m3)
python experiments/controlled_context/run_listwise.py --model zephyr     # environment of requirements-llm.txt
python experiments/controlled_context/run_listwise.py --model first
```
Then run level A.
