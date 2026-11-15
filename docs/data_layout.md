# Data layout

All scripts read and write two data roots and one output root, set by environment variables (defaults are relative to
the repository root and are ignored by git).

| Variable | Default | Content |
|---|---|---|
| `RRT_DATA` | `data/cross_retriever` | Experiment 1: benchmark files, first-stage runs, reranker outputs |
| `RRT_CONTROLLED_DATA` | `data/controlled_context` | Experiment 2: benchmark files, pairs, conditions, model outputs |
| `RRT_OUT` | `outputs` | analysis outputs |
| `RRT_CONTROLLED_RESULTS` | `$RRT_OUT/controlled_context` | Experiment 2 analysis outputs (read by `analysis/analyze.py`) |

## `$RRT_DATA` (Experiment 1)

| Path | Written by | Content |
|---|---|---|
| `common/qrels/<ds>.tsv`, `common/queries/<ds>.tsv` | `data/01_qrels_queries.py` | judgments (qid, docid, grade), query text |
| `common/emb/<ds>.queries.npy` | `data/02_encode_queries.py` | BGE-base query vectors (also used by the dense retriever) |
| `common/runs/<ds>.<bm25,splade,dense>.top100.txt` | `data/03_retrieve.py` | first-stage runs (TREC format) |
| `common/docs/<ds>.jsonl` | `data/04_docs.py` | text of every candidate (+ every judged DL document), bert-base token counts |
| `common/emb/<ds>.docs.npy` | `data/05_embed_docs.py` | BGE-base document vectors (not used by the analysis) |
| `common/validation.md`, `common/manifest.json` | `data/06_validate.py` and others | first-stage effectiveness against Pyserini references |
| `pointwise/scores/<ds>.<reranker>/part-*.parquet` | `pointwise/run_rerankers.py` | one score per unique (qid, docid) |
| `pointwise/runs/<ds>.<retriever>.<reranker>.txt` | `pointwise/run_rerankers.py` | reranked runs |
| `pointwise/grid.parquet` | `pointwise/build_grid.py` | per-query nDCG@10 for every (dataset, retriever, reranker) |
| `listwise_e/queries_mix.csv`, `listwise_e/jobs.jsonl` | `listwise/build_jobs_e.py` | query set E (547) and its listwise jobs |
| `listwise_e/out/<zephyr,first>.jsonl` | `listwise/run_listwise.py --dir listwise_e` | final rankings of 100 candidates |
| `listwise_e/runs/`, `listwise_e/grid_mix.parquet` | `listwise/build_grid_e.py` | listwise runs and per-query nDCG@10 for set E |
| `listwise_f/queries_F.csv`, `listwise_f/F/jobs.jsonl` | `listwise/build_jobs_f.py Fn 742` | query set F (742) and its jobs |
| `listwise_f/F/out/<zephyr,first>.jsonl` | `listwise/run_listwise.py --dir listwise_f/F` | final rankings for set F; `F/out/B_DONE` marks a complete run (`finalize_f.py`) |
| `listwise_f/runs/` | `listwise/make_runs_f.py` | listwise runs for set F |
| `listwise_f/rep/`, `listwise_f/rep50/` | `build_jobs_f.py rep|rep50`, `run_listwise.py` | same-input reruns (10 + 40 set-E queries) |
| `qwen_reranker/scores/<sample,rest>-<ds>.parquet` | `qwen_reranker/run_qwen.py` | Qwen3-Reranker-4B scores |

`analysis/analyze.py` treats set F as present only if `listwise_f/F/out/B_DONE` exists, and its report lists the
execution record in `listwise_f/orchestrate.log` and `listwise_f/n_recompute_v8.json` when these files are present.

## `$RRT_CONTROLLED_DATA` (Experiment 2)

| Path | Written by | Content |
|---|---|---|
| `common/{qrels,queries,runs,docs,emb}/` | `experiments/controlled_context/data/0*.py` | the same benchmark preparation, run separately for Experiment 2 |
| `controlled/pairs.tsv` | `build_pairs.py` | 776 target pairs (97 queries x 8) |
| `controlled/base_sets.json`, `controlled/pools/` | `build_pairs.py` | base 20-document sets and replacement pools |
| `controlled/conditions.jsonl` | `build_conditions.py` | 19,400 calls: base, R, C0–C5 (C5 with five permutations) |
| `controlled/rank_out.jsonl` | `run_listwise.py --model zephyr` | RankZephyr-7B outputs |
| `controlled/models/rank_out_first.jsonl` | `run_listwise.py --model first` | FIRST-7B outputs |
| `controlled/ce_out.jsonl` | `run_ce.py` | bge-reranker-v2-m3 control scores |
| `controlled/logs/build_{pairs,conditions}.json` | `build_pairs.py`, `build_conditions.py` | construction statistics (read by `analyze.py` for its report) |

Experiment 2 uses its own copy of the benchmark preparation. Its DL19/DL20 qrels, queries, BM25 runs and documents are
identical to those of Experiment 1, but the BGE-base document vectors (used to choose C3/C4 replacements) and the dense
runs of NFCorpus and DBpedia were computed in a separate run and differ slightly; regenerated vectors may therefore select
different C3/C4 replacements. The stored `conditions.jsonl` fixes the reported inputs.

## Stored model outputs

The model outputs and derived inputs that the analysis reads (all paths above except `common/docs`, `common/emb`,
`common/qrels`, `common/queries`) are not in git. Their sha256 checksums are in `results/manifests/model_outputs.sha256`
(paths relative to the two data roots). See the README for how to obtain them.
