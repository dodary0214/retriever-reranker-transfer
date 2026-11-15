# Output labels

`analysis/analyze.py` writes a few category labels in Korean. They are kept as written so that the regenerated tables
are identical to the reference tables in `results/summary/` and so that scripts which filter on them keep working.

| Label | Meaning | Where |
|---|---|---|
| `평가 질의` | evaluation queries (all 1,289) | `scope` column of most tables |
| `실험 2와 같은 질의` | the DL19 + DL20 queries, i.e. the queries of Experiment 2 | `subset` column of flip tables |
| `(합계)` | total | `dataset` column of `table1a_queries.csv` |
| `전체` | all collections | `bundle` column of `table1a_queries.csv` |
| `동일 입력 재실행 (관련–비관련)` | same-input rerun, relevant–non-relevant pairs | `column` of `table2_*` |
| `교체 관련–비관련 <retriever pair>` | replacement flips, relevant–non-relevant pairs, one retriever pair | `column` of `table2_*` |
| `교체 관련–비관련 (전체)` | replacement flips, relevant–non-relevant pairs, retriever pairs pooled | `column` of `table2_*` |
| `교체 관련–관련 (전체)` | replacement flips, relevant–relevant pairs, pooled | `column` of `table2_*` |
| `교체 관련–비관련, DL19+DL20만 (전체)` | as above, DL19 + DL20 queries only | `column` of `table2_*` |
| `유의성 결론 일치율` (`범위`) | agreement of pairwise significance conclusions (range) | `table3_conclusions.csv` |
| `시스템 순위 τ (검색기 쌍 3개)` (`범위`) | Kendall's τ of system rankings for the three retriever pairs (range) | `table3_conclusions.csv` |
| `이득 유지율` (`범위`) | oracle gain retention (range) | `table3_conclusions.csv` |

Other identifiers used in the outputs:

| Identifier | Meaning |
|---|---|
| `minilm`, `bge-base`, `bge-m3`, `monot5` | MiniLM-L6, bge-reranker-base, bge-reranker-v2-m3, monoT5-base (pointwise) |
| `zephyr`, `first` | RankZephyr-7B, FIRST-7B (listwise) |
| `qwen` | Qwen3-Reranker-4B (pointwise, robustness analysis only) |
| `bm25`, `splade`, `dense` | BM25, SPLADE++ (EnsembleDistil), BGE-base-en-v1.5 |
| `PW4` | the set of the four pointwise rerankers (k = 0) |
| `MIX4` | bge-reranker-v2-m3 + monoT5 + RankZephyr + FIRST (the k = 2 set fixed before the 15-set analysis) |
| `same_rel` | (query, retriever pair) units whose retrieved relevant documents are identical under both retrievers |
| set E, set F | the 547 queries reranked first and the 742 queries added later; E ∪ F = the 1,289 evaluation queries |
| R, C0–C5 | Experiment 2 conditions: repeat, reorder, same-grade replacement, any-grade replacement, query-similar replacement, document-A-similar replacement, C1 with reordering |
