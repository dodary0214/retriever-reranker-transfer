# retriever-reranker-transfer

Code, settings and result summaries for a study of whether reranker comparisons made under one first-stage retriever
still hold under another.

A reranker is usually chosen on a benchmark under one retriever. Retrievers change (lexical to learned sparse or dense,
new embedding models, rebuilt indexes), and with them the candidate lists the reranker sees. A pointwise reranker scores
each document independently, so the order of two documents present in both lists cannot change. A listwise LLM reranker
reads a window of candidates at once, so the order of the same two documents can depend on the other candidates. This
repository measures how far per-query reranker preferences, document-pair orders and evaluation conclusions carry over
between retrievers, and isolates the effect of the surrounding candidates in a controlled experiment.

- **Experiment 1 (cross-retriever).** Top-100 candidates from BM25, SPLADE++ and BGE-base on six datasets (TREC DL19,
  DL20; BEIR NFCorpus, FiQA, SciFact, DBpedia), 1,289 evaluation queries, reranked by four pointwise rerankers
  (MiniLM-L6, bge-reranker-base, bge-reranker-v2-m3, monoT5-base) and two listwise LLM rerankers (RankZephyr-7B,
  FIRST-7B). Analyses: agreement of the per-query top-1 reranker between retrievers (Cohen's κ) for all 15
  four-reranker sets grouped by the number k of listwise rerankers, on all queries and on queries whose retrieved
  relevant documents are identical; sign agreement of pairwise preferences; order reversals of shared document pairs;
  same-input reruns; agreement of significance conclusions, Kendall's τ of system rankings and retention of the
  per-query oracle gain; robustness by tie threshold, metric, judged-only evaluation, margin and dataset.
- **Experiment 2 (controlled candidate context).** 776 document pairs on 97 TREC DL queries, each fixed in slots 3 and
  12 of a 20-document window while the other 18 documents are repeated (R), reordered (C0) or replaced (C1–C5);
  reversal rates against the base call, paired differences from C0 with query-bootstrap intervals and Bonferroni
  correction over C1–C5.

## Repository structure

```
src/                          shared paths and helpers (common.py), optional GPU-memory coordination
experiments/
  cross_retriever/
    data/                     first-stage retrieval: indexes, qrels/queries, runs, documents, validation
    pointwise/                four pointwise rerankers; per-query nDCG@10 grid
    listwise/                 RankZephyr-7B / FIRST-7B runner, query sets E and F, same-input reruns
    qwen_reranker/            Qwen3-Reranker-4B scores (robustness analysis)
  controlled_context/         Experiment 2: pairs, conditions, listwise and control runs, reversal analysis
    data/                     benchmark preparation for Experiment 2
analysis/
  analyze.py                  main analysis: tables, figure values, consistency checks, RESULTS.md
  query_agreement/            15 four-reranker sets and k groups; matched set swaps; within-type Spearman
  pair_reversals/             same-input rerun window diagnostic
  evaluation_transfer/        raw nDCG@10 behind the oracle gain retention
  robustness/                 robustness by k group, Qwen3-Reranker-4B rows, judged-grade-0 flips
scripts/
  run_analysis.sh             CPU analysis from stored outputs, in order
  compare_outputs.py          compares outputs with results/summary
results/
  summary/                    reference values (CSV) produced by the code in this repository
  figures/                    plots of these values (PDF and PNG)
  manifests/                  query sets, Experiment 2 pairs, checksums of stored model outputs, verification record
docs/                         reproduction steps, environment, data sources, data layout, output labels
```

## Requirements

- Python 3.12. CPU analysis: packages in `requirements.txt` (numpy, pandas, scipy, pyarrow, ir_measures).
- Retrieval and pointwise reranking: the same environment plus Java 21 for Pyserini and a CUDA GPU.
- Listwise reranking: a separate environment from `requirements-llm.txt` (vLLM 0.26.0, rank_llm 0.26.0) and a GPU with
  about 24 GB free for a 7B model.

Exact versions and run settings are in [docs/environment.md](docs/environment.md).

## Installation

```bash
git clone https://github.com/dodary0214/retriever-reranker-transfer.git
cd retriever-reranker-transfer
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# for listwise reranking only
python3.12 -m venv .venv-llm && . .venv-llm/bin/activate && pip install -r requirements-llm.txt
```

## Dataset preparation

Benchmark data are not included. The preparation scripts download the Pyserini prebuilt indexes and read queries and
relevance judgments through `ir_datasets`; see [docs/data_sources.md](docs/data_sources.md) for identifiers and
licenses. Data locations are set by environment variables:

```bash
export RRT_DATA=$PWD/data/cross_retriever
export RRT_CONTROLLED_DATA=$PWD/data/controlled_context
export RRT_OUT=$PWD/outputs
mkdir -p $RRT_DATA/common/{qrels,queries,runs,docs,emb}
python experiments/cross_retriever/data/00_download_indexes.py
python experiments/cross_retriever/data/01_qrels_queries.py
```

## Running experiments

The full sequence (retrieval, pointwise and listwise reranking, Qwen3-Reranker-4B, Experiment 2) is listed in
[docs/reproduction.md](docs/reproduction.md). Main entry points:

```bash
python experiments/cross_retriever/pointwise/run_rerankers.py
python experiments/cross_retriever/listwise/run_listwise.py --model zephyr --phases F --dir $RRT_DATA/listwise_f/F
python experiments/controlled_context/run_listwise.py --model first
```

## Reproducing reported results

All analyses run on CPU from the stored model outputs:

```bash
bash scripts/run_analysis.sh
python scripts/compare_outputs.py
```

The stored model outputs (first-stage runs, pointwise scores, listwise rankings, same-input reruns, Qwen3-Reranker-4B
scores, Experiment 2 pairs, conditions and model outputs; no document text) are not part of the git repository. Their
checksums are listed in `results/manifests/model_outputs.sha256`; the archive is available from the corresponding author of
the article on request until it is deposited in a public archive.

What can be reproduced from what:

| Starting point | Reproduced | Verified here |
|---|---|---|
| Stored model outputs + qrels | every CSV in `results/summary/` | yes: identical values (`results/manifests/verification.md`) |
| Benchmark data only, CPU | query sets E and F, job lists, Experiment 2 pairs and conditions, pointwise grid and listwise runs from stored outputs | yes: byte-identical files |
| Benchmark data + GPU | model outputs | not rerun; listwise outputs vary between engine sessions (docs/environment.md) |

## Expected outputs

`scripts/run_analysis.sh` writes to `$RRT_OUT`:

- `tables/*.csv`: candidate overlap, κ by set and retriever pair, sign agreement, replacement flips, same-input
  reruns, significance agreement, Kendall's τ, oracle gain retention, robustness and per-dataset tables.
- `figures/*_values.csv`: the values behind the figures (the PDF plots written next to them are drafts).
- `derived/*.csv`: the 15 four-reranker sets and k groups (`ksets_*.csv`), matched set swaps (`matched_swap_*.csv`),
  robustness by k group, Qwen3-Reranker-4B rows.
- `controlled_context/*.csv`: Experiment 2 reversal rates per condition and model.
- `RESULTS.md`: all tables in one report, with the construction and consistency checks.

Selected values (relevant-set-identical units, 95% query-bootstrap intervals):

| Quantity | Value | File |
|---|---|---|
| κ of the per-query top-1 reranker, k = 0 / 1 / 2 | 0.900 [0.861, 0.935] / 0.755 [0.712, 0.795] / 0.676 [0.618, 0.732] | `derived/ksets_groups.csv` |
| Replacement reversals of shared relevant–non-relevant pairs, RankZephyr-7B / FIRST-7B | 17.1% / 13.7% (same-input rerun: 1.2% / 0%) | `tables/table2_replacement_flips.csv` |
| Experiment 2 reversal after reordering only (C0), RankZephyr-7B / FIRST-7B | 9.0% / 6.3% (repeat R: 0% / 0%) | `figures/fig4_values.csv` |
| Oracle gain retention, k = 0 / 1 / 2 | 0.722 / 0.576 / 0.472 | `derived/ksets_table3_groups.csv` |

## Correspondence with the article

The output files keep the names used while the analysis was developed, so some file names and the headings in
`RESULTS.md` carry figure, table and appendix numbers that differ from the article. "Flip" in file and column names
means the same as "reversal" in the article. `PW4` is the k = 0 set and `MIX4` is the k = 2 set fixed before the
15-set analysis. Paths are relative to `results/summary/`.

| Article | Content | Files |
|---|---|---|
| Table 1(a) | datasets and evaluation queries | `tables/table1a_queries.csv` |
| Table 1(b) | rerankers, mean nDCG@10 and Judged@10 | `tables/table1b_rerankers.csv` |
| Fig. 2(a) | candidate overlap, relevant-set-identical share | `figures/fig2_values.csv` |
| Fig. 2(b), Sec. 5.1 | κ by four-reranker set and k group; all-query κ | `derived/ksets_sets.csv`, `derived/ksets_groups.csv`, `derived/ksets_group_diffs.csv` |
| Sec. 5.1 | winner-margin strata | `derived/ksets_margin_kappa.csv` |
| Sec. 5.2 | sign agreement of pairwise preferences (`RESULTS.md`: "Fig. 3") | `tables/fig3_sign_agreement.csv`, `derived/ksets_robust_gapmatched.csv` |
| Table 2 | pair-order reversals across retrievers and same-input reruns | `tables/table2_replacement_flips.csv`, `tables/appendixD_same_input_flips.csv` |
| Fig. 3 | Experiment 2 controlled reversals (`RESULTS.md`: "Fig. 4") | `figures/fig4_values.csv` |
| Fig. 4, Sec. 7 | significance agreement, Kendall's τ, oracle gain retention; nDCG@10 behind the retention | `derived/ksets_table3_groups.csv`, `derived/ksets_table3_group_diffs.csv`, `derived/ksets_selection_ndcg.csv` |
| Table A1 | sensitivity of the k = 0 − k = 2 difference | `derived/ksets_robust_A3.csv`, `derived/ksets_robust_S3.csv`, `derived/qwen_kappa.csv` |
| Table A2 | agreement by dataset | `derived/ksets_robust_B1.csv` |
| Table A3 | matched-core substitutions | `derived/matched_swap_swaps_summary.csv`, `derived/matched_swap_model_identity.csv` |
| Fig. A1 | reversal rate and top-1 change (`RESULTS.md`: "Fig. 5") | `derived/ksets_fig5_groups.csv` |
| Table A4 | Experiment 2 conditions | `results/manifests/experiment2_pairs.tsv`, `controlled_context/*.csv` |

The plotted figures are in `results/figures/` under descriptive names (see its README).

Some category labels in the output tables are in Korean; [docs/output_labels.md](docs/output_labels.md) translates them.

## License and citation

The code in this repository is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE). Benchmark data
and model checkpoints are not covered by this license and remain subject to their own licenses (docs/data_sources.md).
If you use this code, please cite the accompanying article.
