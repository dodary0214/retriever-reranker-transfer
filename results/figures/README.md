# Figures

Plots of the results in `results/summary/`, as PDF (vector, Liberation Sans embedded) and PNG (600 dpi, rendered from
the PDF). The plotting scripts are not included; every plotted value is in the CSV files listed below.

| File | Content | Values |
|---|---|---|
| `candidate_overlap_and_agreement` | (a) top-100 Jaccard similarity and share of queries with an identical retrieved relevant set per retriever pair; (b) Cohen's κ of the per-query top-1 reranker between retrievers on relevant-set-identical units, per four-reranker set and k group, with the same-input rerun reference | `summary/figures/fig2_values.csv` (a), `summary/derived/ksets_sets.csv`, `ksets_groups.csv` (b) |
| `controlled_context_reversals` | Experiment 2: (a) controlled reversal rate per condition (R, C0–C5) and paired difference from C0 with Bonferroni marks; (b) C1 reversal rate by correctness of the base call | `summary/figures/fig4_values.csv` |
| `evaluation_transfer` | agreement of significance conclusions and oracle gain retention between retrievers, by k group | `summary/derived/ksets_table3_groups.csv` |
| `reversal_rate_vs_top1_change` | share of relevant-set-identical units whose top-1 reranker changes, by the mean shared relevant–non-relevant reversal rate of the two listwise rerankers, by k group (share, 0–1) | `summary/derived/ksets_fig5_groups.csv` |

k = number of listwise rerankers in a four-reranker set (k = 0 blue circles, k = 1 purple squares, k = 2 vermillion
diamonds). Error bars are 95% query-bootstrap intervals (10,000 resamples, seed 0).
