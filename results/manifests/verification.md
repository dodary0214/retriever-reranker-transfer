# Verification record

Date: 2026-11-15. Host: NVIDIA GB10 (aarch64), CPU only (no model inference). Environment of `requirements.txt`.

The code in this repository was checked against the original experiment outputs from which the reported values were
taken. The original outputs were read through a separate data root (symbolic links) and all new outputs were written to a
separate directory; the sha256 checksums of all 1,006 original files were identical before and after the checks.

## 1. Analysis (level A in docs/reproduction.md)

`scripts/run_analysis.sh` was run on the original outputs and its CSV files were compared with the original analysis
outputs, column by column.

| Output | Files | Result |
|---|---|---|
| `tables/*.csv` | 26 | identical values (maximum absolute difference 0) |
| `figures/*_values.csv`, `fig5_placement.csv` | 5 | identical values |
| `derived/*.csv` (15 sets, k groups, robustness, Qwen3-Reranker-4B rows, diagnostics) | 30 | identical values |
| `controlled_context/*.csv` (Experiment 2) | 16 | identical values for RankZephyr-7B and FIRST-7B; the original files also contain rows for a third model (Qwen2.5-7B-Instruct) that is not part of the reported analysis and is not run here |

The `consistency_checks.csv` table, which `analysis/analyze.py` builds from the run settings in the listwise runner
source files of both experiments, is identical as well (all checks pass).

The same pipeline was then run on a fresh data root that contained only the stored model outputs listed in
`model_outputs.sha256` and the qrels and query files written by `experiments/cross_retriever/data/01_qrels_queries.py`:
`scripts/compare_outputs.py` reported 77/77 files matching `results/summary/` (tolerance 1e-9; all differences were 0).

## Addition (2026-11-15): matched set swaps

`analysis/query_agreement/matched_swap.py` compares sets that differ by one reranker (a pointwise reranker replaced by a
listwise one; 8 swaps for k = 0 → 1, 24 for k = 1 → 2) on relevant-set-identical and on all units, with each set's own
untied units and with the units untied for both sets, using the definitions and the bootstrap of `analysis/analyze.py`
(10,000 query resamples, seed 0, shared by all sets). It recomputes the per-set and k-group κ values; all 40 values
compared in `matched_swap_comparison_with_reported.csv` equal the reference values (maximum absolute difference
5.6e-17). Its six output files were identical byte for byte to those of the analysis run before it was added here, and
`scripts/compare_outputs.py` reported 83/83 files matching on a data root that held only the stored model outputs.

## 2. Inputs (level B)

Regenerated from the benchmark files on CPU and compared byte by byte with the original files:

| File | Script | Result |
|---|---|---|
| qrels and queries, six datasets | `data/01_qrels_queries.py` | identical |
| `pointwise/grid.parquet` (26,520 rows) | `pointwise/build_grid.py` | identical values |
| `listwise_e/queries_mix.csv` (set E) | `listwise/build_jobs_e.py` | identical |
| `listwise_e/jobs.jsonl` | `listwise/build_jobs_e.py` | the 1,641 non-swap jobs identical, in the same order (the original file also has 711 swap jobs, see docs/reproduction.md) |
| `listwise_e/runs/*.txt` (30), `grid_mix.parquet` | `listwise/build_grid_e.py` | identical |
| `listwise_f/queries_F.csv` (set F), `F/jobs.jsonl` | `listwise/build_jobs_f.py Fn 742`, `finalize_f.py` | identical |
| `listwise_f/rep*/jobs.jsonl`, query lists | `listwise/build_jobs_f.py rep`, `rep50` | identical |
| `listwise_f/runs/*.txt` (24) | `listwise/make_runs_f.py` | identical |
| `rep/diag_rep.csv`, `rep/t_sec.txt` | `listwise/diag_rep.py`, `compare_rep.py` | identical |
| `controlled/pairs.tsv`, `base_sets.json`, `pools/`, `conditions.jsonl` | `build_pairs.py`, `build_conditions.py` | identical |

## 3. Not verified

- Model inference (first-stage encoders, pointwise and listwise rerankers, Qwen3-Reranker-4B, Experiment 2 control) was
  not rerun. Listwise outputs are known to vary between vLLM engine sessions for RankZephyr-7B (docs/environment.md).
- First-stage retrieval (`03_retrieve.py`), document export and embeddings were not rerun; the stored runs are used.
- The Experiment 2 data preparation under `experiments/controlled_context/data/` was not rerun.

## Changes to the original code

Only paths, imports and host-specific resource handling were changed; metric definitions, sampling, bootstrap,
seeds and decision rules are unchanged.

- Paths are read from `RRT_DATA`, `RRT_CONTROLLED_DATA`, `RRT_OUT` and `RRT_CONTROLLED_RESULTS` instead of fixed
  directories; directory names inside the data roots were made descriptive (`pointwise/`, `listwise_e/`, `listwise_f/`,
  `qwen_reranker/`, `controlled/`).
- The two copies of the listwise runner used for sets E and F (identical except for the job directory argument and
  host-specific memory handling) were merged into `experiments/cross_retriever/listwise/run_listwise.py`.
- GPU memory reservation through the host manager became optional (`src/host_resources.py`); a one-time launch-lock
  bypass option that was never used was removed.
- `build_jobs_e.py` adds the exploratory swap jobs only if their candidate file is present.
- `build_pairs.py` creates its output directories.
- Comments and report headings that referred to internal planning documents were rewritten; `RESULTS.md` headings are
  in English. Category labels inside the output tables were left unchanged (docs/output_labels.md).
