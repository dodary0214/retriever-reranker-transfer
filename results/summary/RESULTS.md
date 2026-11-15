# RESULTS — cross-retriever reranker transfer

Generated 2026-11-15 09:17:09. Evaluation queries 1289 (set E 547 + set F 742); queries with runs of all three retrievers 1753. ALL 1753, E 547, F 742, evaluation 1289 (B present) / nDCG@10 recomputed vs grid_mix: 24318 rows, max |diff| 0.00e+00

## Set F execution record (n recomputation, FIRST check)

```
2026-11-11 12:00:55 n = 1058
arg 1058.0 -> n = 1058
2026-11-11 14:01:21 v5 stop: zephyr gen elapsed 856 min, lists done 367; n = 1058
arg 1058.0 -> n = 1058
2026-11-11 14:04:07 v5b recompute: zephyr gen elapsed 103.2 min, lists done 367; n = 788
arg 788.0 -> n = 788
2026-11-11 16:17:38 measured: 1761 s, lists 112, calls 1008 -> 34.34 calls/min, t_RZ 47.17 s/query
2026-11-11 16:17:40 stopped v5b at chunk boundary; zephyr lists 651
2026-11-11 16:17:40 v8 recompute: t_RZ 47.17 s, t_FIRST 50.48 s, done RZ queries 217, n = 742, expected end 2026-11-12T09:49
arg 742.0 -> n = 742
2026-11-11 22:29:17 zephyr done for n = 742
2026-11-11 22:46:47 FIRST check: tF 45.6 s/query, done 20/742, eta Thu Nov 12 07:55:30 2026, decision KEEP n2 872
2026-11-12 05:00:19 F done; total GPU wall 65409s (18.17 h)
n_recompute_v8.json: {"time": "2026-11-11T16:17:40", "t_RZ": 47.17, "ratio": 1.0701754385964912, "t_FIRST": 50.48017543859649, "done_RZ_queries": 217, "n": 742, "eta": "2026-11-12T09:49"}
v8_first_check.txt: tF 45.6 s/query, done 20/742, eta Thu Nov 12 07:55:30 2026, decision KEEP n2 872
```

## Hypotheses H1–H3 (set F only, 742 queries; not used in figures or tables)

| hypothesis | verdict | evidence |
|---|---|---|
| H1 | supported | bm25–splade +0.140 [+0.024, +0.267]; bm25–dense +0.212 [+0.053, +0.382]; splade–dense +0.234 [+0.122, +0.354] → CI lower bound > 0: 3/3 |
| H2 | supported | PP − PL +0.071 [+0.054, +0.087] (all units, original definition) |
| H3 RankZephyr | supported | 0.170 [0.156, 0.184] |
| H3 FIRST | supported | 0.131 [0.118, 0.144] |

## Table 1(a). Datasets

| bundle   | dataset   |   n_all |   n_eval |   mean_rel_per_query | experiment2   |
|:---------|:----------|--------:|---------:|---------------------:|:--------------|
| TREC DL  | dl19      |      43 |       43 |                95.4  | yes           |
| TREC DL  | dl20      |      54 |       54 |                66.78 | yes           |
| TREC DL  | (합계)      |      97 |       97 |                79.46 |               |
| BEIR     | nfcorpus  |     308 |      228 |                43.17 | no            |
| BEIR     | fiqa      |     648 |      476 |                 2.6  | no            |
| BEIR     | scifact   |     300 |      184 |                 1.15 | no            |
| BEIR     | dbpedia   |     400 |      304 |                34.26 | no            |
| BEIR     | (합계)      |    1656 |     1192 |                18.21 |               |
| 전체       | (합계)      |    1753 |     1289 |                22.82 |               |

## Table 1(b). Rerankers

| reranker   | type                          | checkpoint                           | params   | precision   | batch                                                 | max_length          | truncation                                              |   ndcg10_bm25 |   ndcg10_splade |   ndcg10_dense |   judged10_bm25 |   judged10_splade |   judged10_dense |
|:-----------|:------------------------------|:-------------------------------------|:---------|:------------|:------------------------------------------------------|:--------------------|:--------------------------------------------------------|--------------:|----------------:|---------------:|----------------:|------------------:|-----------------:|
| MiniLM     | pointwise (cross-encoder)     | cross-encoder/ms-marco-MiniLM-L-6-v2 | 22.7M    | fp16        | 64 (query, doc) pairs                                 | 512 tokens          | only_second (document cut)                              |         0.44  |           0.453 |          0.461 |           0.338 |             0.318 |            0.321 |
| bge-base   | pointwise (cross-encoder)     | BAAI/bge-reranker-base               | 278.0M   | fp16        | 64 (query, doc) pairs                                 | 512 tokens          | only_second (document cut)                              |         0.422 |           0.427 |          0.434 |           0.317 |             0.301 |            0.3   |
| bge-m3     | pointwise (cross-encoder)     | BAAI/bge-reranker-v2-m3              | 567.8M   | fp16        | 64 (query, doc) pairs                                 | 512 tokens          | only_second (document cut)                              |         0.465 |           0.485 |          0.489 |           0.334 |             0.32  |            0.317 |
| monoT5     | pointwise (seq2seq)           | castorini/monot5-base-msmarco-10k    | 222.9M   | bf16        | 64 (query, doc) pairs                                 | 512 tokens          | document tokens cut to fit the prompt                   |         0.465 |           0.471 |          0.479 |           0.34  |             0.31  |            0.311 |
| RankZephyr | listwise (generative)         | castorini/rank_zephyr_7b_v1_full     | 7.24B    | bf16        | vLLM offline, ~500 window calls; window 20, stride 10 | 4,096-token context | 300 words per passage, cut evenly until the prompt fits |         0.48  |           0.504 |          0.516 |           0.329 |             0.318 |            0.324 |
| FIRST      | listwise (first-token logits) | castorini/first_mistral              | 7.25B    | bf16        | vLLM offline, ~500 window calls; window 20, stride 10 | 8,192-token context | 300 words per passage, cut evenly until the prompt fits |         0.505 |           0.534 |          0.554 |           0.352 |             0.337 |            0.341 |

## Fig. 2(a). Candidate overlap

| scope   | pair         |    n |   jaccard |   jac_lo |   jac_hi |   same_rel |   same_rel_lo |   same_rel_hi |
|:--------|:-------------|-----:|----------:|---------:|---------:|-----------:|--------------:|--------------:|
| 평가 질의   | bm25–splade  | 1289 |     0.241 |    0.233 |    0.248 |      0.49  |         0.463 |         0.517 |
| 평가 질의   | bm25–dense   | 1289 |     0.15  |    0.145 |    0.156 |      0.409 |         0.382 |         0.435 |
| 평가 질의   | splade–dense | 1289 |     0.245 |    0.238 |    0.252 |      0.488 |         0.461 |         0.515 |

## Fig. 2(b) / Appendix A. κ (ties excluded, threshold 0.01, nDCG@10)

| scope   | metric   |   th | subset   | pair         | set      |   n_queries |   n_notie |      po |      pe |   kappa |    lo |    hi |
|:--------|:---------|-----:|:---------|:-------------|:---------|------------:|----------:|--------:|--------:|--------:|------:|------:|
| 평가 질의   | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |        1289 |       505 |   0.802 |   0.266 |   0.73  | 0.682 | 0.776 |
| 평가 질의   | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |        1289 |       466 |   0.672 |   0.261 |   0.556 | 0.497 | 0.614 |
| 평가 질의   | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.175 | 0.107 | 0.242 |
| 평가 질의   | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |        1289 |       492 |   0.754 |   0.264 |   0.666 | 0.615 | 0.715 |
| 평가 질의   | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |        1289 |       464 |   0.606 |   0.266 |   0.463 | 0.399 | 0.523 |
| 평가 질의   | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.203 | 0.13  | 0.276 |
| 평가 질의   | nDCG@10  | 0.01 | all      | splade–dense | PW4      |        1289 |       562 |   0.861 |   0.266 |   0.811 | 0.772 | 0.849 |
| 평가 질의   | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |        1289 |       538 |   0.684 |   0.267 |   0.569 | 0.515 | 0.622 |
| 평가 질의   | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.242 | 0.179 | 0.307 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |         632 |       174 |   0.937 |   0.277 |   0.913 | 0.858 | 0.959 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |         632 |       140 |   0.786 |   0.278 |   0.703 | 0.604 | 0.795 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.209 | 0.109 | 0.316 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |         527 |       150 |   0.9   |   0.277 |   0.862 | 0.79  | 0.923 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |         527 |       119 |   0.731 |   0.292 |   0.62  | 0.499 | 0.727 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.241 | 0.112 | 0.376 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |         629 |       215 |   0.94  |   0.275 |   0.917 | 0.87  | 0.957 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |         629 |       186 |   0.785 |   0.284 |   0.7   | 0.613 | 0.778 |
| 평가 질의   | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.217 | 0.126 | 0.311 |

| set   |   n_queries |   units |   n_notie |    po |    pe |   kappa |    lo |   hi |
|:------|------------:|--------:|----------:|------:|------:|--------:|------:|-----:|
| MIX4  |          50 |     150 |        71 | 0.958 | 0.325 |   0.937 | 0.827 |    1 |

## Fig. 3. Sign agreement vs mean effectiveness gap (relevant-set-identical units; all units in the table files)

| scope   | subset   | metric   |   th | group   | pair                |   units |   agree |   mean_diff |   abs_diff |    lo |    hi |   diff_lo |   diff_hi |
|:--------|:---------|:---------|-----:|:--------|:--------------------|--------:|--------:|------------:|-----------:|------:|------:|----------:|----------:|
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP      | MiniLM–bge-base     |     793 |   0.962 |       0.023 |      0.023 | 0.944 | 0.979 |     0.014 |     0.032 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP      | MiniLM–bge-m3       |     815 |   0.968 |      -0.028 |      0.028 | 0.952 | 0.983 |    -0.038 |    -0.019 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP      | MiniLM–monoT5       |     714 |   0.976 |      -0.02  |      0.02  | 0.961 | 0.989 |    -0.029 |    -0.012 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | MiniLM–RankZephyr   |     771 |   0.874 |      -0.049 |      0.049 | 0.844 | 0.903 |    -0.059 |    -0.038 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | MiniLM–FIRST        |     771 |   0.927 |      -0.08  |      0.08  | 0.903 | 0.949 |    -0.09  |    -0.07  |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP      | bge-base–bge-m3     |     724 |   0.968 |      -0.052 |      0.052 | 0.952 | 0.982 |    -0.06  |    -0.043 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP      | bge-base–monoT5     |     797 |   0.961 |      -0.044 |      0.044 | 0.942 | 0.978 |    -0.053 |    -0.034 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | bge-base–RankZephyr |     823 |   0.87  |      -0.072 |      0.072 | 0.84  | 0.898 |    -0.083 |    -0.06  |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | bge-base–FIRST      |     827 |   0.937 |      -0.103 |      0.103 | 0.916 | 0.957 |    -0.114 |    -0.092 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP      | bge-m3–monoT5       |     704 |   0.963 |       0.008 |      0.008 | 0.945 | 0.979 |     0     |     0.016 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | bge-m3–RankZephyr   |     739 |   0.847 |      -0.02  |      0.02  | 0.814 | 0.88  |    -0.03  |    -0.01  |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | bge-m3–FIRST        |     732 |   0.921 |      -0.051 |      0.051 | 0.896 | 0.943 |    -0.061 |    -0.042 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | monoT5–RankZephyr   |     717 |   0.845 |      -0.028 |      0.028 | 0.812 | 0.878 |    -0.038 |    -0.019 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | monoT5–FIRST        |     725 |   0.908 |      -0.059 |      0.059 | 0.88  | 0.934 |    -0.068 |    -0.05  |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | LL      | RankZephyr–FIRST    |     533 |   0.801 |      -0.031 |      0.031 | 0.759 | 0.842 |    -0.038 |    -0.024 |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6]      |    4547 |   0.966 |     nan     |      0.029 | 0.958 | 0.974 |   nan     |   nan     |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8]      |    6105 |   0.892 |     nan     |      0.058 | 0.877 | 0.907 |   nan     |   nan     |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1]      |     533 |   0.801 |     nan     |      0.031 | 0.759 | 0.842 |   nan     |   nan     |
| 평가 질의   | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]           |     nan |   0.075 |     nan     |    nan     | 0.06  | 0.09  |   nan     |   nan     |

## Table 2. Listwise flips (zero for pointwise rerankers by construction)

| reranker   | 동일 입력 재실행 (관련–비관련)             | 교체 관련–비관련 bm25–splade          | 교체 관련–비관련 bm25–dense           | 교체 관련–비관련 splade–dense         | 교체 관련–비관련 (전체)                 | 교체 관련–관련 (전체)                  | 교체 관련–비관련, DL19+DL20만 (전체)     |
|:-----------|:-------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|
| RankZephyr | 0.012 [0.007, 0.018] (110,252) | 0.174 [0.164, 0.184] (168,339) | 0.175 [0.164, 0.186] (100,948) | 0.166 [0.156, 0.176] (202,863) | 0.171 [0.162, 0.179] (472,150) | 0.238 [0.225, 0.252] (206,646) | 0.158 [0.141, 0.176] (101,344) |
| FIRST      | 0.000 [0.000, 0.000] (19,198)  | 0.142 [0.133, 0.151] (168,339) | 0.149 [0.138, 0.159] (100,948) | 0.128 [0.120, 0.136] (202,863) | 0.137 [0.130, 0.145] (472,150) | 0.196 [0.186, 0.206] (206,646) | 0.145 [0.130, 0.160] (101,344) |

## Fig. 4. Controlled flips (Experiment 2, recomputed here: 10,000 bootstrap resamples)

| panel   | model      | cond   |   rate |     lo |     hi |   excess |   excess_lo |   excess_hi |   excess_bonf_lo |   excess_bonf_hi |   sig_bonferroni |   n_units |   n_pairs | base         |
|:--------|:-----------|:-------|-------:|-------:|-------:|---------:|------------:|------------:|-----------------:|-----------------:|-----------------:|----------:|----------:|:-------------|
| a       | RankZephyr | R      | 0      | 0      | 0      |  -0.0899 |     -0.1049 |     -0.0753 |          -0.1095 |          -0.0706 |                0 |      1552 |       776 | nan          |
| a       | RankZephyr | C0     | 0.0899 | 0.0753 | 0.1049 |   0      |      0      |      0      |           0      |           0      |                0 |      3880 |       776 | nan          |
| a       | RankZephyr | C1     | 0.1022 | 0.0833 | 0.122  |   0.0123 |     -0.0042 |      0.0297 |          -0.0095 |           0.0352 |                0 |      2328 |       776 | nan          |
| a       | RankZephyr | C2     | 0.1121 | 0.0932 | 0.1319 |   0.0222 |      0.0082 |      0.0371 |           0.0039 |           0.0418 |                1 |      2328 |       776 | nan          |
| a       | RankZephyr | C3     | 0.1302 | 0.1057 | 0.1559 |   0.0402 |      0.0183 |      0.0629 |           0.0113 |           0.0701 |                1 |       776 |       776 | nan          |
| a       | RankZephyr | C4     | 0.1057 | 0.0799 | 0.1327 |   0.0157 |     -0.0075 |      0.0392 |          -0.0137 |           0.0466 |                0 |       776 |       776 | nan          |
| a       | RankZephyr | C5     | 0.1121 | 0.0932 | 0.1319 |   0.0222 |      0.0088 |      0.0365 |           0.0051 |           0.0413 |                1 |      2328 |       776 | nan          |
| b       | RankZephyr | C1     | 0.2462 | 0.1848 | 0.3113 | nan      |    nan      |    nan      |         nan      |         nan      |              nan |       333 |       111 | base wrong   |
| b       | RankZephyr | C1     | 0.0467 | 0.0334 | 0.0613 | nan      |    nan      |    nan      |         nan      |         nan      |              nan |      1413 |       471 | base correct |
| a       | FIRST      | R      | 0      | 0      | 0      |  -0.0634 |     -0.0778 |     -0.05   |          -0.0827 |          -0.0459 |                0 |      1552 |       776 | nan          |
| a       | FIRST      | C0     | 0.0634 | 0.05   | 0.0778 |   0      |      0      |      0      |           0      |           0      |                0 |      3880 |       776 | nan          |
| a       | FIRST      | C1     | 0.0666 | 0.0503 | 0.0846 |   0.0032 |     -0.0119 |      0.0185 |          -0.0162 |           0.0241 |                0 |      2328 |       776 | nan          |
| a       | FIRST      | C2     | 0.0735 | 0.0571 | 0.0906 |   0.0101 |     -0.0022 |      0.0226 |          -0.0057 |           0.0262 |                0 |      2328 |       776 | nan          |
| a       | FIRST      | C3     | 0.0709 | 0.0515 | 0.0915 |   0.0075 |     -0.0093 |      0.025  |          -0.0144 |           0.0304 |                0 |       776 |       776 | nan          |
| a       | FIRST      | C4     | 0.0631 | 0.0451 | 0.0838 |  -0.0003 |     -0.0178 |      0.0186 |          -0.0237 |           0.025  |                0 |       776 |       776 | nan          |
| a       | FIRST      | C5     | 0.0653 | 0.0498 | 0.0825 |   0.0019 |     -0.0087 |      0.0128 |          -0.0121 |           0.0157 |                0 |      2328 |       776 | nan          |
| b       | FIRST      | C1     | 0.2299 | 0.1206 | 0.349  | nan      |    nan      |    nan      |         nan      |         nan      |              nan |       174 |        58 | base wrong   |
| b       | FIRST      | C1     | 0.0261 | 0.0154 | 0.0383 | nan      |    nan      |    nan      |         nan      |         nan      |              nan |      1572 |       524 | base correct |

## Fig. 5. Top-1 change rate by pair-flip bin (293 units without a shared relevant–non-relevant pair for either listwise reranker excluded)

| scope   | set   | bin         |   units_in_bin |   units_notie |   n_queries |   change_rate |    lo |    hi |
|:--------|:------|:------------|---------------:|--------------:|------------:|--------------:|------:|------:|
| 평가 질의   | MIX4  | 0           |            685 |            35 |          29 |         0.029 | 0     | 0.098 |
| 평가 질의   | MIX4  | (0, 0.1]    |            459 |           208 |         128 |         0.25  | 0.181 | 0.321 |
| 평가 질의   | MIX4  | (0.1, 0.25] |            286 |           164 |         115 |         0.238 | 0.167 | 0.314 |
| 평가 질의   | MIX4  | > 0.25      |             65 |            38 |          29 |         0.263 | 0.119 | 0.424 |
| 평가 질의   | PW4   | 0           |            685 |            89 |          58 |         0.056 | 0.01  | 0.122 |
| 평가 질의   | PW4   | (0, 0.1]    |            459 |           246 |         144 |         0.049 | 0.022 | 0.081 |
| 평가 질의   | PW4   | (0.1, 0.25] |            286 |           175 |         121 |         0.103 | 0.053 | 0.159 |
| 평가 질의   | PW4   | > 0.25      |             65 |            29 |          23 |         0.138 | 0.029 | 0.281 |

Placement rule: Spearman(bin order, MIX4 change rate) = 0.800; (>0.25) − (0) = 0.235 [0.078, 0.402] → **main text**

## Table 3. Agreement of evaluation conclusions across retrievers (significance, retention: mean of three retriever pairs [95% CI], range; τ: three values and range)

| set   |   n_queries | 유의성 결론 일치율           | 유의성 결론 일치율 범위   | 시스템 순위 τ (검색기 쌍 3개)   | 시스템 순위 τ 범위   | 이득 유지율               | 이득 유지율 범위   |
|:------|------------:|:---------------------|:----------------|:----------------------|:--------------|:---------------------|:------------|
| PW4   |        1289 | 0.796 [0.704, 0.889] | 0.722–0.944     | 0.722 / 0.611 / 0.889 | 0.611–0.889   | 0.722 [0.684, 0.757] | 0.633–0.816 |
| MIX4  |        1289 | 0.796 [0.630, 0.833] | 0.722–0.917     | 0.889 / 0.833 / 0.944 | 0.833–0.944   | 0.491 [0.439, 0.540] | 0.414–0.545 |

Note: τ: four systems, so τ can take only a few distinct values; the three retriever-pair values and their range are shown. Smallest mean nDCG@10 gap between adjacent-ranked models (mean over retrievers): PW4 0.0080, MIX4 0.0080.

Mean nDCG@10 gaps between the models of each set (Table 1(b) values)

| set   | retriever   |   max_minus_min |   min_adjacent_gap |
|:------|:------------|----------------:|-------------------:|
| PW4   | bm25        |          0.0427 |             0.0004 |
| PW4   | splade      |          0.0572 |             0.014  |
| PW4   | dense       |          0.0549 |             0.0097 |
| MIX4  | bm25        |          0.0405 |             0.0004 |
| MIX4  | splade      |          0.063  |             0.014  |
| MIX4  | dense       |          0.0745 |             0.0097 |
| PW4   | mean        |          0.0516 |             0.008  |
| MIX4  | mean        |          0.0593 |             0.008  |

| scope   | set   |   n_queries |   sig_units |   sig_share_nodiff |   tau_units |   retention_pooled6 |   retention_pooled6_lo |   retention_pooled6_hi |   sig_mean |   sig_min |   sig_max |   sig_lo |   sig_hi |   sig_bm25–splade |   sig_bm25–dense |   sig_splade–dense |   tau_mean |   tau_min |   tau_max |   tau_lo |   tau_hi |   tau_bm25–splade |   tau_bm25–dense |   tau_splade–dense |   retention_mean |   retention_min |   retention_max |   retention_lo |   retention_hi |   retention_bm25–splade |   retention_bm25–dense |   retention_splade–dense |
|:--------|:------|------------:|------------:|-------------------:|------------:|--------------------:|-----------------------:|-----------------------:|-----------:|----------:|----------:|---------:|---------:|------------------:|-----------------:|-------------------:|-----------:|----------:|----------:|---------:|---------:|------------------:|-----------------:|-------------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|------------------------:|-----------------------:|-------------------------:|
| 평가 질의   | PW4   |        1289 |         108 |              0.463 |          18 |               0.724 |                  0.687 |                  0.758 |      0.796 |     0.722 |     0.944 |    0.704 |    0.889 |             0.722 |            0.722 |              0.944 |      0.741 |     0.611 |     0.889 |    0.593 |    0.889 |             0.722 |            0.611 |              0.889 |            0.722 |           0.633 |           0.816 |          0.684 |          0.757 |                   0.717 |                  0.633 |                    0.816 |
| 평가 질의   | MIX4  |        1289 |         108 |              0.426 |          18 |               0.491 |                  0.439 |                  0.54  |      0.796 |     0.722 |     0.917 |    0.63  |    0.833 |             0.722 |            0.75  |              0.917 |      0.889 |     0.833 |     0.944 |    0.593 |    0.926 |             0.889 |            0.833 |              0.944 |            0.491 |           0.414 |           0.545 |          0.439 |          0.54  |                   0.515 |                  0.414 |                    0.545 |

## Judgment-gap check

Share of judged documents in the top 10 (Judged@10, mean and range over the three retrievers):

| reranker   |   mean |   min |   max |
|:-----------|-------:|------:|------:|
| MiniLM     |  0.325 | 0.318 | 0.338 |
| bge-base   |  0.306 | 0.3   | 0.317 |
| bge-m3     |  0.324 | 0.317 | 0.334 |
| monoT5     |  0.32  | 0.31  | 0.34  |
| RankZephyr |  0.324 | 0.318 | 0.329 |
| FIRST      |  0.343 | 0.337 | 0.352 |

κ recomputed with nDCG@10 on judged documents only (threshold 0.01):

| scope   | metric      |   th | subset   | pair         | set      |   n_queries |   n_notie |      po |      pe |   kappa |     lo |    hi |
|:--------|:------------|-----:|:---------|:-------------|:---------|------------:|----------:|--------:|--------:|--------:|-------:|------:|
| 평가 질의   | nDCG_judged | 0.01 | all      | bm25–splade  | PW4      |        1289 |       228 |   0.781 |   0.257 |   0.705 |  0.629 | 0.776 |
| 평가 질의   | nDCG_judged | 0.01 | all      | bm25–splade  | MIX4     |        1289 |       218 |   0.638 |   0.258 |   0.512 |  0.424 | 0.598 |
| 평가 질의   | nDCG_judged | 0.01 | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.193 |  0.088 | 0.297 |
| 평가 질의   | nDCG_judged | 0.01 | all      | bm25–dense   | PW4      |        1289 |       205 |   0.737 |   0.259 |   0.645 |  0.561 | 0.723 |
| 평가 질의   | nDCG_judged | 0.01 | all      | bm25–dense   | MIX4     |        1289 |       204 |   0.569 |   0.254 |   0.422 |  0.328 | 0.511 |
| 평가 질의   | nDCG_judged | 0.01 | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.223 |  0.117 | 0.331 |
| 평가 질의   | nDCG_judged | 0.01 | all      | splade–dense | PW4      |        1289 |       230 |   0.83  |   0.259 |   0.771 |  0.705 | 0.832 |
| 평가 질의   | nDCG_judged | 0.01 | all      | splade–dense | MIX4     |        1289 |       227 |   0.639 |   0.257 |   0.514 |  0.424 | 0.594 |
| 평가 질의   | nDCG_judged | 0.01 | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.257 |  0.158 | 0.359 |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | bm25–splade  | PW4      |         632 |        30 |   0.967 |   0.274 |   0.954 |  0.839 | 1     |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | bm25–splade  | MIX4     |         632 |        23 |   0.652 |   0.253 |   0.534 |  0.242 | 0.78  |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.42  |  0.189 | 0.701 |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | bm25–dense   | PW4      |         527 |        16 |   0.938 |   0.316 |   0.909 |  0.662 | 1     |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | bm25–dense   | MIX4     |         527 |        15 |   0.733 |   0.24  |   0.649 |  0.345 | 0.917 |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.259 | -0.082 | 0.591 |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | splade–dense | PW4      |         629 |        31 |   0.903 |   0.282 |   0.865 |  0.69  | 1     |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | splade–dense | MIX4     |         629 |        29 |   0.793 |   0.254 |   0.722 |  0.495 | 0.908 |
| 평가 질의   | nDCG_judged | 0.01 | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.143 | -0.084 | 0.384 |

## Construction and consistency checks

95 checks, 0 failed.

No failed checks.

| check                    | item                                                | exp1                    | exp2                | ok   |
|:-------------------------|:----------------------------------------------------|:------------------------|:--------------------|:-----|
| 7.4 model identity       | checkpoint RankZephyr                               | True                    | True                | True |
| 7.4 model identity       | checkpoint FIRST                                    | True                    | True                | True |
| 7.4 model identity       | RankZephyr prompt (rank_llm, context 4096)          | True                    | True                | True |
| 7.4 model identity       | FIRST prompt (alpha template)                       | True                    | True                | True |
| 7.4 model identity       | FIRST system message                                | True                    | True                | True |
| 7.4 model identity       | FIRST logits                                        | True                    | True                | True |
| 7.4 model identity       | FIRST variable passages                             | True                    | True                | True |
| 7.4 model identity       | FIRST "[" prefix                                    | True                    | True                | True |
| 7.4 model identity       | FIRST sampling (1 token, logprobs 30)               | True                    | True                | True |
| 7.4 model identity       | temperature 0                                       | True                    | True                | True |
| 7.4 model identity       | vLLM enforce_eager False / max_logprobs 30          | True                    | True                | True |
| 7.4 model identity       | vLLM seed                                           | True                    | True                | True |
| 7.4 model identity       | max_model_len from rank_llm                         | True                    | True                | True |
| 7.4 model identity       | vLLM version (both .venv_llm)                       | 0.26.0                  | 0.26.0              | True |
| 7.4 model identity       | window (design difference, reported)                | 20 / stride 10 over 100 | single window of 20 | True |
| 7.5 experiment-2 queries | DL19+DL20 evaluation queries = experiment-2 queries | 97                      | 97                  | True |
| 8.1 evaluation set       | E ∪ F, all six rerankers present                    | 1289                    | 1289                | True |
| 8.1 evaluation set       | bundle TREC DL = sum of its datasets                | 97                      | 97                  | True |
| 8.1 evaluation set       | bundle BEIR = sum of its datasets                   | 1192                    | 1192                | True |

### Query and pair counts

| item                                          |   n_queries |   n_notie |   n_units |   n_pairs |
|:----------------------------------------------|------------:|----------:|----------:|----------:|
| Fig. 2(a)                                     |        1289 |       nan |       nan |       nan |
| Fig. 2(b) all bm25–splade PW4                 |        1289 |       505 |       nan |       nan |
| Fig. 2(b) all bm25–splade MIX4                |        1289 |       466 |       nan |       nan |
| Fig. 2(b) all bm25–dense PW4                  |        1289 |       492 |       nan |       nan |
| Fig. 2(b) all bm25–dense MIX4                 |        1289 |       464 |       nan |       nan |
| Fig. 2(b) all splade–dense PW4                |        1289 |       562 |       nan |       nan |
| Fig. 2(b) all splade–dense MIX4               |        1289 |       538 |       nan |       nan |
| Fig. 2(b) same_rel bm25–splade PW4            |         632 |       174 |       nan |       nan |
| Fig. 2(b) same_rel bm25–splade MIX4           |         632 |       140 |       nan |       nan |
| Fig. 2(b) same_rel bm25–dense PW4             |         527 |       150 |       nan |       nan |
| Fig. 2(b) same_rel bm25–dense MIX4            |         527 |       119 |       nan |       nan |
| Fig. 2(b) same_rel splade–dense PW4           |         629 |       215 |       nan |       nan |
| Fig. 2(b) same_rel splade–dense MIX4          |         629 |       186 |       nan |       nan |
| Fig. 2(b) same-input κ                        |          50 |        71 |       nan |       nan |
| Fig. 3                                        |        1289 |       nan |      6105 |       nan |
| Fig. 4                                        |          97 |       nan |       nan |       776 |
| Fig. 5 MIX4 0                                 |          29 |       nan |        35 |       nan |
| Fig. 5 MIX4 (0, 0.1]                          |         128 |       nan |       208 |       nan |
| Fig. 5 MIX4 (0.1, 0.25]                       |         115 |       nan |       164 |       nan |
| Fig. 5 MIX4 > 0.25                            |          29 |       nan |        38 |       nan |
| Fig. 5 PW4 0                                  |          58 |       nan |        89 |       nan |
| Fig. 5 PW4 (0, 0.1]                           |         144 |       nan |       246 |       nan |
| Fig. 5 PW4 (0.1, 0.25]                        |         121 |       nan |       175 |       nan |
| Fig. 5 PW4 > 0.25                             |          23 |       nan |        29 |       nan |
| Table 1                                       |        1289 |       nan |       nan |       nan |
| Table 3                                       |        1289 |       nan |       nan |       nan |
| Table 2 RankZephyr 동일 입력 재실행 (관련–비관련)         |          50 |       nan |       nan |    110252 |
| Table 2 RankZephyr 교체 관련–비관련 bm25–splade      |        1289 |       nan |       nan |    168339 |
| Table 2 RankZephyr 교체 관련–비관련 bm25–dense       |        1289 |       nan |       nan |    100948 |
| Table 2 RankZephyr 교체 관련–비관련 splade–dense     |        1289 |       nan |       nan |    202863 |
| Table 2 RankZephyr 교체 관련–비관련 (전체)             |        1289 |       nan |       nan |    472150 |
| Table 2 RankZephyr 교체 관련–관련 (전체)              |        1289 |       nan |       nan |    206646 |
| Table 2 RankZephyr 교체 관련–비관련, DL19+DL20만 (전체) |          97 |       nan |       nan |    101344 |
| Table 2 FIRST 동일 입력 재실행 (관련–비관련)              |          10 |       nan |       nan |     19198 |
| Table 2 FIRST 교체 관련–비관련 bm25–splade           |        1289 |       nan |       nan |    168339 |
| Table 2 FIRST 교체 관련–비관련 bm25–dense            |        1289 |       nan |       nan |    100948 |
| Table 2 FIRST 교체 관련–비관련 splade–dense          |        1289 |       nan |       nan |    202863 |
| Table 2 FIRST 교체 관련–비관련 (전체)                  |        1289 |       nan |       nan |    472150 |
| Table 2 FIRST 교체 관련–관련 (전체)                   |        1289 |       nan |       nan |    206646 |
| Table 2 FIRST 교체 관련–비관련, DL19+DL20만 (전체)      |          97 |       nan |       nan |    101344 |
| Appendix C TREC DL                            |          97 |       nan |       nan |       nan |
| Appendix C BEIR                               |        1192 |       nan |       nan |       nan |
| Appendix C dl19                               |          43 |       nan |       nan |       nan |
| Appendix C dl20                               |          54 |       nan |       nan |       nan |
| Appendix C nfcorpus                           |         228 |       nan |       nan |       nan |
| Appendix C fiqa                               |         476 |       nan |       nan |       nan |
| Appendix C scifact                            |         184 |       nan |       nan |       nan |
| Appendix C dbpedia                            |         304 |       nan |       nan |       nan |

## Appendix B. Sensitivity — sign agreement by pair type

| scope   | subset   | metric      |    th | group   | pair           |   units |   agree |   abs_diff |    lo |    hi |
|:--------|:---------|:------------|------:|:--------|:---------------|--------:|--------:|-----------:|------:|------:|
| 평가 질의   | all      | nDCG@10     | 0.005 | PP      | [PP pooled, 6] |   12396 |   0.885 |      0.029 | 0.874 | 0.896 |
| 평가 질의   | all      | nDCG@10     | 0.005 | PL      | [PL pooled, 8] |   17005 |   0.81  |      0.058 | 0.798 | 0.822 |
| 평가 질의   | all      | nDCG@10     | 0.005 | LL      | [LL pooled, 1] |    1781 |   0.712 |      0.031 | 0.686 | 0.738 |
| 평가 질의   | all      | nDCG@10     | 0.005 | PP−PL   | [PP − PL]      |     nan |   0.075 |    nan     | 0.062 | 0.088 |
| 평가 질의   | same_rel | nDCG@10     | 0.005 | PP      | [PP pooled, 6] |    4617 |   0.965 |      0.029 | 0.957 | 0.974 |
| 평가 질의   | same_rel | nDCG@10     | 0.005 | PL      | [PL pooled, 8] |    6224 |   0.887 |      0.058 | 0.871 | 0.902 |
| 평가 질의   | same_rel | nDCG@10     | 0.005 | LL      | [LL pooled, 1] |     546 |   0.791 |      0.031 | 0.749 | 0.832 |
| 평가 질의   | same_rel | nDCG@10     | 0.005 | PP−PL   | [PP − PL]      |     nan |   0.079 |    nan     | 0.064 | 0.094 |
| 평가 질의   | all      | nDCG@10     | 0.02  | PP      | [PP pooled, 6] |   10797 |   0.911 |      0.029 | 0.9   | 0.921 |
| 평가 질의   | all      | nDCG@10     | 0.02  | PL      | [PL pooled, 8] |   15013 |   0.837 |      0.058 | 0.825 | 0.849 |
| 평가 질의   | all      | nDCG@10     | 0.02  | LL      | [LL pooled, 1] |    1473 |   0.746 |      0.031 | 0.719 | 0.773 |
| 평가 질의   | all      | nDCG@10     | 0.02  | PP−PL   | [PP − PL]      |     nan |   0.074 |    nan     | 0.061 | 0.087 |
| 평가 질의   | same_rel | nDCG@10     | 0.02  | PP      | [PP pooled, 6] |    4347 |   0.971 |      0.029 | 0.962 | 0.978 |
| 평가 질의   | same_rel | nDCG@10     | 0.02  | PL      | [PL pooled, 8] |    5876 |   0.9   |      0.058 | 0.885 | 0.914 |
| 평가 질의   | same_rel | nDCG@10     | 0.02  | LL      | [LL pooled, 1] |     491 |   0.817 |      0.031 | 0.773 | 0.858 |
| 평가 질의   | same_rel | nDCG@10     | 0.02  | PP−PL   | [PP − PL]      |     nan |   0.071 |    nan     | 0.056 | 0.086 |
| 평가 질의   | all      | RR@10       | 0.005 | PP      | [PP pooled, 6] |    6227 |   0.946 |      0.031 | 0.934 | 0.956 |
| 평가 질의   | all      | RR@10       | 0.005 | PL      | [PL pooled, 8] |    8268 |   0.893 |      0.068 | 0.879 | 0.907 |
| 평가 질의   | all      | RR@10       | 0.005 | LL      | [LL pooled, 1] |     626 |   0.783 |      0.03  | 0.742 | 0.822 |
| 평가 질의   | all      | RR@10       | 0.005 | PP−PL   | [PP − PL]      |     nan |   0.052 |    nan     | 0.037 | 0.067 |
| 평가 질의   | same_rel | RR@10       | 0.005 | PP      | [PP pooled, 6] |    3297 |   0.979 |      0.031 | 0.971 | 0.986 |
| 평가 질의   | same_rel | RR@10       | 0.005 | PL      | [PL pooled, 8] |    4457 |   0.923 |      0.068 | 0.909 | 0.937 |
| 평가 질의   | same_rel | RR@10       | 0.005 | LL      | [LL pooled, 1] |     345 |   0.806 |      0.03  | 0.753 | 0.856 |
| 평가 질의   | same_rel | RR@10       | 0.005 | PP−PL   | [PP − PL]      |     nan |   0.055 |    nan     | 0.04  | 0.07  |
| 평가 질의   | all      | RR@10       | 0.01  | PP      | [PP pooled, 6] |    6227 |   0.946 |      0.031 | 0.934 | 0.956 |
| 평가 질의   | all      | RR@10       | 0.01  | PL      | [PL pooled, 8] |    8268 |   0.893 |      0.068 | 0.879 | 0.907 |
| 평가 질의   | all      | RR@10       | 0.01  | LL      | [LL pooled, 1] |     626 |   0.783 |      0.03  | 0.742 | 0.822 |
| 평가 질의   | all      | RR@10       | 0.01  | PP−PL   | [PP − PL]      |     nan |   0.052 |    nan     | 0.037 | 0.067 |
| 평가 질의   | same_rel | RR@10       | 0.01  | PP      | [PP pooled, 6] |    3297 |   0.979 |      0.031 | 0.971 | 0.986 |
| 평가 질의   | same_rel | RR@10       | 0.01  | PL      | [PL pooled, 8] |    4457 |   0.923 |      0.068 | 0.909 | 0.937 |
| 평가 질의   | same_rel | RR@10       | 0.01  | LL      | [LL pooled, 1] |     345 |   0.806 |      0.03  | 0.753 | 0.856 |
| 평가 질의   | same_rel | RR@10       | 0.01  | PP−PL   | [PP − PL]      |     nan |   0.055 |    nan     | 0.04  | 0.07  |
| 평가 질의   | all      | RR@10       | 0.02  | PP      | [PP pooled, 6] |    6132 |   0.95  |      0.031 | 0.939 | 0.96  |
| 평가 질의   | all      | RR@10       | 0.02  | PL      | [PL pooled, 8] |    8207 |   0.895 |      0.068 | 0.881 | 0.909 |
| 평가 질의   | all      | RR@10       | 0.02  | LL      | [LL pooled, 1] |     615 |   0.793 |      0.03  | 0.753 | 0.833 |
| 평가 질의   | all      | RR@10       | 0.02  | PP−PL   | [PP − PL]      |     nan |   0.055 |    nan     | 0.04  | 0.069 |
| 평가 질의   | same_rel | RR@10       | 0.02  | PP      | [PP pooled, 6] |    3242 |   0.983 |      0.031 | 0.975 | 0.989 |
| 평가 질의   | same_rel | RR@10       | 0.02  | PL      | [PL pooled, 8] |    4425 |   0.925 |      0.068 | 0.911 | 0.939 |
| 평가 질의   | same_rel | RR@10       | 0.02  | LL      | [LL pooled, 1] |     340 |   0.815 |      0.03  | 0.761 | 0.866 |
| 평가 질의   | same_rel | RR@10       | 0.02  | PP−PL   | [PP − PL]      |     nan |   0.057 |    nan     | 0.043 | 0.072 |
| 평가 질의   | all      | nDCG_judged | 0.005 | PP      | [PP pooled, 6] |    5810 |   0.847 |      0.003 | 0.831 | 0.862 |
| 평가 질의   | all      | nDCG_judged | 0.005 | PL      | [PL pooled, 8] |    7946 |   0.754 |      0.007 | 0.736 | 0.772 |
| 평가 질의   | all      | nDCG_judged | 0.005 | LL      | [LL pooled, 1] |     861 |   0.639 |      0     | 0.602 | 0.677 |
| 평가 질의   | all      | nDCG_judged | 0.005 | PP−PL   | [PP − PL]      |     nan |   0.093 |    nan     | 0.072 | 0.113 |
| 평가 질의   | same_rel | nDCG_judged | 0.005 | PP      | [PP pooled, 6] |     664 |   0.941 |      0.003 | 0.914 | 0.967 |
| 평가 질의   | same_rel | nDCG_judged | 0.005 | PL      | [PL pooled, 8] |     892 |   0.827 |      0.007 | 0.781 | 0.872 |
| 평가 질의   | same_rel | nDCG_judged | 0.005 | LL      | [LL pooled, 1] |      90 |   0.644 |      0     | 0.53  | 0.753 |
| 평가 질의   | same_rel | nDCG_judged | 0.005 | PP−PL   | [PP − PL]      |     nan |   0.114 |    nan     | 0.066 | 0.164 |
| 평가 질의   | all      | nDCG_judged | 0.01  | PP      | [PP pooled, 6] |    5107 |   0.865 |      0.003 | 0.849 | 0.881 |
| 평가 질의   | all      | nDCG_judged | 0.01  | PL      | [PL pooled, 8] |    7157 |   0.768 |      0.007 | 0.748 | 0.787 |
| 평가 질의   | all      | nDCG_judged | 0.01  | LL      | [LL pooled, 1] |     736 |   0.648 |      0     | 0.609 | 0.688 |
| 평가 질의   | all      | nDCG_judged | 0.01  | PP−PL   | [PP − PL]      |     nan |   0.098 |    nan     | 0.076 | 0.119 |
| 평가 질의   | same_rel | nDCG_judged | 0.01  | PP      | [PP pooled, 6] |     615 |   0.943 |      0.003 | 0.914 | 0.97  |
| 평가 질의   | same_rel | nDCG_judged | 0.01  | PL      | [PL pooled, 8] |     835 |   0.838 |      0.007 | 0.79  | 0.883 |
| 평가 질의   | same_rel | nDCG_judged | 0.01  | LL      | [LL pooled, 1] |      77 |   0.675 |      0     | 0.558 | 0.788 |
| 평가 질의   | same_rel | nDCG_judged | 0.01  | PP−PL   | [PP − PL]      |     nan |   0.105 |    nan     | 0.057 | 0.155 |
| 평가 질의   | all      | nDCG_judged | 0.02  | PP      | [PP pooled, 6] |    4222 |   0.897 |      0.003 | 0.88  | 0.913 |
| 평가 질의   | all      | nDCG_judged | 0.02  | PL      | [PL pooled, 8] |    5931 |   0.796 |      0.007 | 0.775 | 0.817 |
| 평가 질의   | all      | nDCG_judged | 0.02  | LL      | [LL pooled, 1] |     568 |   0.681 |      0     | 0.636 | 0.726 |
| 평가 질의   | all      | nDCG_judged | 0.02  | PP−PL   | [PP − PL]      |     nan |   0.101 |    nan     | 0.078 | 0.123 |
| 평가 질의   | same_rel | nDCG_judged | 0.02  | PP      | [PP pooled, 6] |     536 |   0.966 |      0.003 | 0.94  | 0.989 |
| 평가 질의   | same_rel | nDCG_judged | 0.02  | PL      | [PL pooled, 8] |     694 |   0.876 |      0.007 | 0.834 | 0.914 |
| 평가 질의   | same_rel | nDCG_judged | 0.02  | LL      | [LL pooled, 1] |      64 |   0.703 |      0     | 0.579 | 0.824 |
| 평가 질의   | same_rel | nDCG_judged | 0.02  | PP−PL   | [PP − PL]      |     nan |   0.09  |    nan     | 0.053 | 0.132 |

## Appendix B. Sensitivity — κ

| scope   | metric      |    th | subset   | pair         | set      |   n_queries |   n_notie |      po |      pe |   kappa |     lo |    hi |
|:--------|:------------|------:|:---------|:-------------|:---------|------------:|----------:|--------:|--------:|--------:|-------:|------:|
| 평가 질의   | nDCG@10     | 0.005 | all      | bm25–splade  | PW4      |        1289 |       559 |   0.785 |   0.264 |   0.708 |  0.662 | 0.754 |
| 평가 질의   | nDCG@10     | 0.005 | all      | bm25–splade  | MIX4     |        1289 |       515 |   0.66  |   0.26  |   0.541 |  0.484 | 0.597 |
| 평가 질의   | nDCG@10     | 0.005 | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.167 |  0.104 | 0.23  |
| 평가 질의   | nDCG@10     | 0.005 | all      | bm25–dense   | PW4      |        1289 |       547 |   0.729 |   0.263 |   0.633 |  0.583 | 0.682 |
| 평가 질의   | nDCG@10     | 0.005 | all      | bm25–dense   | MIX4     |        1289 |       513 |   0.585 |   0.264 |   0.436 |  0.376 | 0.493 |
| 평가 질의   | nDCG@10     | 0.005 | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.197 |  0.127 | 0.268 |
| 평가 질의   | nDCG@10     | 0.005 | all      | splade–dense | PW4      |        1289 |       620 |   0.84  |   0.262 |   0.784 |  0.744 | 0.822 |
| 평가 질의   | nDCG@10     | 0.005 | all      | splade–dense | MIX4     |        1289 |       596 |   0.658 |   0.264 |   0.535 |  0.482 | 0.586 |
| 평가 질의   | nDCG@10     | 0.005 | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.249 |  0.188 | 0.312 |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | bm25–splade  | PW4      |         632 |       180 |   0.933 |   0.277 |   0.908 |  0.854 | 0.954 |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | bm25–splade  | MIX4     |         632 |       150 |   0.773 |   0.273 |   0.688 |  0.589 | 0.777 |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.22  |  0.124 | 0.322 |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | bm25–dense   | PW4      |         527 |       156 |   0.897 |   0.279 |   0.858 |  0.786 | 0.92  |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | bm25–dense   | MIX4     |         527 |       125 |   0.728 |   0.286 |   0.619 |  0.501 | 0.724 |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.239 |  0.114 | 0.368 |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | splade–dense | PW4      |         629 |       224 |   0.929 |   0.275 |   0.901 |  0.852 | 0.945 |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | splade–dense | MIX4     |         629 |       196 |   0.77  |   0.28  |   0.681 |  0.595 | 0.76  |
| 평가 질의   | nDCG@10     | 0.005 | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.22  |  0.132 | 0.312 |
| 평가 질의   | nDCG@10     | 0.02  | all      | bm25–splade  | PW4      |        1289 |       428 |   0.841 |   0.269 |   0.783 |  0.734 | 0.829 |
| 평가 질의   | nDCG@10     | 0.02  | all      | bm25–splade  | MIX4     |        1289 |       375 |   0.712 |   0.264 |   0.609 |  0.545 | 0.671 |
| 평가 질의   | nDCG@10     | 0.02  | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.174 |  0.103 | 0.245 |
| 평가 질의   | nDCG@10     | 0.02  | all      | bm25–dense   | PW4      |        1289 |       416 |   0.786 |   0.264 |   0.709 |  0.657 | 0.761 |
| 평가 질의   | nDCG@10     | 0.02  | all      | bm25–dense   | MIX4     |        1289 |       393 |   0.623 |   0.269 |   0.485 |  0.416 | 0.55  |
| 평가 질의   | nDCG@10     | 0.02  | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.225 |  0.146 | 0.304 |
| 평가 질의   | nDCG@10     | 0.02  | all      | splade–dense | PW4      |        1289 |       486 |   0.899 |   0.267 |   0.862 |  0.826 | 0.898 |
| 평가 질의   | nDCG@10     | 0.02  | all      | splade–dense | MIX4     |        1289 |       445 |   0.719 |   0.266 |   0.618 |  0.56  | 0.673 |
| 평가 질의   | nDCG@10     | 0.02  | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.245 |  0.18  | 0.311 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | bm25–splade  | PW4      |         632 |       162 |   0.944 |   0.281 |   0.923 |  0.869 | 0.967 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | bm25–splade  | MIX4     |         632 |       123 |   0.846 |   0.282 |   0.785 |  0.689 | 0.869 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.138 |  0.041 | 0.243 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | bm25–dense   | PW4      |         527 |       142 |   0.915 |   0.278 |   0.883 |  0.814 | 0.942 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | bm25–dense   | MIX4     |         527 |       108 |   0.75  |   0.29  |   0.648 |  0.523 | 0.758 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.235 |  0.11  | 0.371 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | splade–dense | PW4      |         629 |       202 |   0.955 |   0.275 |   0.939 |  0.896 | 0.974 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | splade–dense | MIX4     |         629 |       172 |   0.802 |   0.283 |   0.724 |  0.636 | 0.803 |
| 평가 질의   | nDCG@10     | 0.02  | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.214 |  0.125 | 0.309 |
| 평가 질의   | RR@10       | 0.005 | all      | bm25–splade  | PW4      |        1289 |       197 |   0.893 |   0.267 |   0.855 |  0.792 | 0.91  |
| 평가 질의   | RR@10       | 0.005 | all      | bm25–splade  | MIX4     |        1289 |       149 |   0.738 |   0.269 |   0.642 |  0.541 | 0.736 |
| 평가 질의   | RR@10       | 0.005 | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.213 |  0.107 | 0.323 |
| 평가 질의   | RR@10       | 0.005 | all      | bm25–dense   | PW4      |        1289 |       188 |   0.878 |   0.27  |   0.832 |  0.766 | 0.891 |
| 평가 질의   | RR@10       | 0.005 | all      | bm25–dense   | MIX4     |        1289 |       136 |   0.691 |   0.284 |   0.569 |  0.454 | 0.677 |
| 평가 질의   | RR@10       | 0.005 | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.263 |  0.148 | 0.385 |
| 평가 질의   | RR@10       | 0.005 | all      | splade–dense | PW4      |        1289 |       237 |   0.932 |   0.268 |   0.908 |  0.862 | 0.949 |
| 평가 질의   | RR@10       | 0.005 | all      | splade–dense | MIX4     |        1289 |       171 |   0.766 |   0.261 |   0.684 |  0.592 | 0.768 |
| 평가 질의   | RR@10       | 0.005 | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.224 |  0.133 | 0.322 |
| 평가 질의   | RR@10       | 0.005 | same_rel | bm25–splade  | PW4      |         632 |       113 |   0.982 |   0.284 |   0.975 |  0.936 | 1     |
| 평가 질의   | RR@10       | 0.005 | same_rel | bm25–splade  | MIX4     |         632 |        83 |   0.795 |   0.284 |   0.714 |  0.579 | 0.832 |
| 평가 질의   | RR@10       | 0.005 | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.261 |  0.136 | 0.401 |
| 평가 질의   | RR@10       | 0.005 | same_rel | bm25–dense   | PW4      |         527 |       104 |   0.952 |   0.279 |   0.933 |  0.869 | 0.986 |
| 평가 질의   | RR@10       | 0.005 | same_rel | bm25–dense   | MIX4     |         527 |        74 |   0.784 |   0.286 |   0.697 |  0.551 | 0.821 |
| 평가 질의   | RR@10       | 0.005 | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.236 |  0.101 | 0.388 |
| 평가 질의   | RR@10       | 0.005 | same_rel | splade–dense | PW4      |         629 |       136 |   0.985 |   0.276 |   0.98  |  0.947 | 1     |
| 평가 질의   | RR@10       | 0.005 | same_rel | splade–dense | MIX4     |         629 |       106 |   0.84  |   0.283 |   0.776 |  0.67  | 0.869 |
| 평가 질의   | RR@10       | 0.005 | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.203 |  0.104 | 0.315 |
| 평가 질의   | RR@10       | 0.01  | all      | bm25–splade  | PW4      |        1289 |       197 |   0.893 |   0.267 |   0.855 |  0.792 | 0.91  |
| 평가 질의   | RR@10       | 0.01  | all      | bm25–splade  | MIX4     |        1289 |       149 |   0.738 |   0.269 |   0.642 |  0.541 | 0.736 |
| 평가 질의   | RR@10       | 0.01  | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.213 |  0.107 | 0.323 |
| 평가 질의   | RR@10       | 0.01  | all      | bm25–dense   | PW4      |        1289 |       188 |   0.878 |   0.27  |   0.832 |  0.766 | 0.891 |
| 평가 질의   | RR@10       | 0.01  | all      | bm25–dense   | MIX4     |        1289 |       136 |   0.691 |   0.284 |   0.569 |  0.454 | 0.677 |
| 평가 질의   | RR@10       | 0.01  | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.263 |  0.148 | 0.385 |
| 평가 질의   | RR@10       | 0.01  | all      | splade–dense | PW4      |        1289 |       237 |   0.932 |   0.268 |   0.908 |  0.862 | 0.949 |
| 평가 질의   | RR@10       | 0.01  | all      | splade–dense | MIX4     |        1289 |       171 |   0.766 |   0.261 |   0.684 |  0.592 | 0.768 |
| 평가 질의   | RR@10       | 0.01  | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.224 |  0.133 | 0.322 |
| 평가 질의   | RR@10       | 0.01  | same_rel | bm25–splade  | PW4      |         632 |       113 |   0.982 |   0.284 |   0.975 |  0.936 | 1     |
| 평가 질의   | RR@10       | 0.01  | same_rel | bm25–splade  | MIX4     |         632 |        83 |   0.795 |   0.284 |   0.714 |  0.579 | 0.832 |
| 평가 질의   | RR@10       | 0.01  | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.261 |  0.136 | 0.401 |
| 평가 질의   | RR@10       | 0.01  | same_rel | bm25–dense   | PW4      |         527 |       104 |   0.952 |   0.279 |   0.933 |  0.869 | 0.986 |
| 평가 질의   | RR@10       | 0.01  | same_rel | bm25–dense   | MIX4     |         527 |        74 |   0.784 |   0.286 |   0.697 |  0.551 | 0.821 |
| 평가 질의   | RR@10       | 0.01  | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.236 |  0.101 | 0.388 |
| 평가 질의   | RR@10       | 0.01  | same_rel | splade–dense | PW4      |         629 |       136 |   0.985 |   0.276 |   0.98  |  0.947 | 1     |
| 평가 질의   | RR@10       | 0.01  | same_rel | splade–dense | MIX4     |         629 |       106 |   0.84  |   0.283 |   0.776 |  0.67  | 0.869 |
| 평가 질의   | RR@10       | 0.01  | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.203 |  0.104 | 0.315 |
| 평가 질의   | RR@10       | 0.02  | all      | bm25–splade  | PW4      |        1289 |       196 |   0.893 |   0.268 |   0.854 |  0.79  | 0.91  |
| 평가 질의   | RR@10       | 0.02  | all      | bm25–splade  | MIX4     |        1289 |       145 |   0.745 |   0.273 |   0.649 |  0.547 | 0.745 |
| 평가 질의   | RR@10       | 0.02  | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.205 |  0.098 | 0.314 |
| 평가 질의   | RR@10       | 0.02  | all      | bm25–dense   | PW4      |        1289 |       182 |   0.89  |   0.271 |   0.849 |  0.785 | 0.907 |
| 평가 질의   | RR@10       | 0.02  | all      | bm25–dense   | MIX4     |        1289 |       128 |   0.703 |   0.288 |   0.583 |  0.465 | 0.693 |
| 평가 질의   | RR@10       | 0.02  | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.266 |  0.152 | 0.389 |
| 평가 질의   | RR@10       | 0.02  | all      | splade–dense | PW4      |        1289 |       233 |   0.936 |   0.269 |   0.912 |  0.867 | 0.952 |
| 평가 질의   | RR@10       | 0.02  | all      | splade–dense | MIX4     |        1289 |       167 |   0.772 |   0.26  |   0.693 |  0.601 | 0.777 |
| 평가 질의   | RR@10       | 0.02  | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.219 |  0.127 | 0.319 |
| 평가 질의   | RR@10       | 0.02  | same_rel | bm25–splade  | PW4      |         632 |       112 |   0.982 |   0.287 |   0.975 |  0.935 | 1     |
| 평가 질의   | RR@10       | 0.02  | same_rel | bm25–splade  | MIX4     |         632 |        80 |   0.8   |   0.291 |   0.718 |  0.582 | 0.838 |
| 평가 질의   | RR@10       | 0.02  | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.257 |  0.13  | 0.398 |
| 평가 질의   | RR@10       | 0.02  | same_rel | bm25–dense   | PW4      |         527 |       102 |   0.951 |   0.283 |   0.932 |  0.866 | 0.985 |
| 평가 질의   | RR@10       | 0.02  | same_rel | bm25–dense   | MIX4     |         527 |        70 |   0.786 |   0.287 |   0.7   |  0.55  | 0.826 |
| 평가 질의   | RR@10       | 0.02  | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.232 |  0.095 | 0.388 |
| 평가 질의   | RR@10       | 0.02  | same_rel | splade–dense | PW4      |         629 |       134 |   0.985 |   0.279 |   0.979 |  0.946 | 1     |
| 평가 질의   | RR@10       | 0.02  | same_rel | splade–dense | MIX4     |         629 |       104 |   0.846 |   0.28  |   0.786 |  0.681 | 0.879 |
| 평가 질의   | RR@10       | 0.02  | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.193 |  0.095 | 0.304 |
| 평가 질의   | nDCG_judged | 0.005 | all      | bm25–splade  | PW4      |        1289 |       277 |   0.744 |   0.252 |   0.657 |  0.585 | 0.726 |
| 평가 질의   | nDCG_judged | 0.005 | all      | bm25–splade  | MIX4     |        1289 |       252 |   0.619 |   0.261 |   0.484 |  0.403 | 0.564 |
| 평가 질의   | nDCG_judged | 0.005 | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.173 |  0.076 | 0.27  |
| 평가 질의   | nDCG_judged | 0.005 | all      | bm25–dense   | PW4      |        1289 |       267 |   0.7   |   0.254 |   0.598 |  0.522 | 0.671 |
| 평가 질의   | nDCG_judged | 0.005 | all      | bm25–dense   | MIX4     |        1289 |       244 |   0.553 |   0.256 |   0.4   |  0.313 | 0.485 |
| 평가 질의   | nDCG_judged | 0.005 | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.198 |  0.1   | 0.301 |
| 평가 질의   | nDCG_judged | 0.005 | all      | splade–dense | PW4      |        1289 |       286 |   0.804 |   0.256 |   0.737 |  0.675 | 0.796 |
| 평가 질의   | nDCG_judged | 0.005 | all      | splade–dense | MIX4     |        1289 |       271 |   0.616 |   0.258 |   0.483 |  0.402 | 0.559 |
| 평가 질의   | nDCG_judged | 0.005 | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.254 |  0.16  | 0.349 |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | bm25–splade  | PW4      |         632 |        33 |   0.97  |   0.276 |   0.958 |  0.853 | 1     |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | bm25–splade  | MIX4     |         632 |        26 |   0.615 |   0.26  |   0.48  |  0.193 | 0.722 |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.478 |  0.248 | 0.752 |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | bm25–dense   | PW4      |         527 |        17 |   0.941 |   0.304 |   0.915 |  0.697 | 1     |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | bm25–dense   | MIX4     |         527 |        15 |   0.733 |   0.24  |   0.649 |  0.345 | 0.917 |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.266 | -0.072 | 0.597 |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | splade–dense | PW4      |         629 |        36 |   0.889 |   0.275 |   0.847 |  0.68  | 0.965 |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | splade–dense | MIX4     |         629 |        37 |   0.73  |   0.252 |   0.639 |  0.432 | 0.822 |
| 평가 질의   | nDCG_judged | 0.005 | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.208 | -0.025 | 0.448 |
| 평가 질의   | nDCG_judged | 0.01  | all      | bm25–splade  | PW4      |        1289 |       228 |   0.781 |   0.257 |   0.705 |  0.629 | 0.776 |
| 평가 질의   | nDCG_judged | 0.01  | all      | bm25–splade  | MIX4     |        1289 |       218 |   0.638 |   0.258 |   0.512 |  0.424 | 0.598 |
| 평가 질의   | nDCG_judged | 0.01  | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.193 |  0.088 | 0.297 |
| 평가 질의   | nDCG_judged | 0.01  | all      | bm25–dense   | PW4      |        1289 |       205 |   0.737 |   0.259 |   0.645 |  0.561 | 0.723 |
| 평가 질의   | nDCG_judged | 0.01  | all      | bm25–dense   | MIX4     |        1289 |       204 |   0.569 |   0.254 |   0.422 |  0.328 | 0.511 |
| 평가 질의   | nDCG_judged | 0.01  | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.223 |  0.117 | 0.331 |
| 평가 질의   | nDCG_judged | 0.01  | all      | splade–dense | PW4      |        1289 |       230 |   0.83  |   0.259 |   0.771 |  0.705 | 0.832 |
| 평가 질의   | nDCG_judged | 0.01  | all      | splade–dense | MIX4     |        1289 |       227 |   0.639 |   0.257 |   0.514 |  0.424 | 0.594 |
| 평가 질의   | nDCG_judged | 0.01  | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.257 |  0.158 | 0.359 |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | bm25–splade  | PW4      |         632 |        30 |   0.967 |   0.274 |   0.954 |  0.839 | 1     |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | bm25–splade  | MIX4     |         632 |        23 |   0.652 |   0.253 |   0.534 |  0.242 | 0.78  |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.42  |  0.189 | 0.701 |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | bm25–dense   | PW4      |         527 |        16 |   0.938 |   0.316 |   0.909 |  0.662 | 1     |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | bm25–dense   | MIX4     |         527 |        15 |   0.733 |   0.24  |   0.649 |  0.345 | 0.917 |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.259 | -0.082 | 0.591 |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | splade–dense | PW4      |         629 |        31 |   0.903 |   0.282 |   0.865 |  0.69  | 1     |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | splade–dense | MIX4     |         629 |        29 |   0.793 |   0.254 |   0.722 |  0.495 | 0.908 |
| 평가 질의   | nDCG_judged | 0.01  | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.143 | -0.084 | 0.384 |
| 평가 질의   | nDCG_judged | 0.02  | all      | bm25–splade  | PW4      |        1289 |       153 |   0.797 |   0.257 |   0.727 |  0.635 | 0.809 |
| 평가 질의   | nDCG_judged | 0.02  | all      | bm25–splade  | MIX4     |        1289 |       152 |   0.664 |   0.256 |   0.549 |  0.445 | 0.649 |
| 평가 질의   | nDCG_judged | 0.02  | all      | bm25–splade  | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.178 |  0.063 | 0.298 |
| 평가 질의   | nDCG_judged | 0.02  | all      | bm25–dense   | PW4      |        1289 |       147 |   0.789 |   0.265 |   0.713 |  0.62  | 0.8   |
| 평가 질의   | nDCG_judged | 0.02  | all      | bm25–dense   | MIX4     |        1289 |       147 |   0.626 |   0.257 |   0.496 |  0.387 | 0.6   |
| 평가 질의   | nDCG_judged | 0.02  | all      | bm25–dense   | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.217 |  0.099 | 0.335 |
| 평가 질의   | nDCG_judged | 0.02  | all      | splade–dense | PW4      |        1289 |       177 |   0.893 |   0.258 |   0.855 |  0.792 | 0.913 |
| 평가 질의   | nDCG_judged | 0.02  | all      | splade–dense | MIX4     |        1289 |       155 |   0.729 |   0.253 |   0.637 |  0.54  | 0.728 |
| 평가 질의   | nDCG_judged | 0.02  | all      | splade–dense | PW4−MIX4 |        1289 |       nan | nan     | nan     |   0.218 |  0.116 | 0.323 |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | bm25–splade  | PW4      |         632 |        23 |   1     |   0.267 |   1     |  1     | 1     |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | bm25–splade  | MIX4     |         632 |        20 |   0.7   |   0.255 |   0.597 |  0.286 | 0.847 |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | bm25–splade  | PW4−MIX4 |         632 |       nan | nan     | nan     |   0.403 |  0.153 | 0.714 |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | bm25–dense   | PW4      |         527 |        14 |   0.929 |   0.281 |   0.901 |  0.636 | 1     |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | bm25–dense   | MIX4     |         527 |        14 |   0.786 |   0.255 |   0.712 |  0.395 | 1     |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | bm25–dense   | PW4−MIX4 |         527 |       nan | nan     | nan     |   0.188 | -0.161 | 0.537 |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | splade–dense | PW4      |         629 |        25 |   0.92  |   0.283 |   0.888 |  0.705 | 1     |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | splade–dense | MIX4     |         629 |        24 |   0.833 |   0.253 |   0.777 |  0.543 | 0.95  |
| 평가 질의   | nDCG_judged | 0.02  | same_rel | splade–dense | PW4−MIX4 |         629 |       nan | nan     | nan     |   0.112 | -0.098 | 0.344 |

## Appendix C. By collection and dataset

### κ

| scope    | metric   |   th | subset   | pair         | set      |   n_queries |   n_notie |      po |      pe |   kappa |      lo |      hi |
|:---------|:---------|-----:|:---------|:-------------|:---------|------------:|----------:|--------:|--------:|--------:|--------:|--------:|
| TREC DL  | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |          97 |        58 |   0.655 |   0.254 |   0.538 |   0.374 |   0.689 |
| TREC DL  | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |          97 |        55 |   0.618 |   0.294 |   0.459 |   0.275 |   0.629 |
| TREC DL  | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |          97 |       nan | nan     | nan     |   0.079 |  -0.107 |   0.27  |
| TREC DL  | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |          97 |        60 |   0.633 |   0.245 |   0.514 |   0.355 |   0.662 |
| TREC DL  | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |          97 |        62 |   0.661 |   0.279 |   0.53  |   0.362 |   0.683 |
| TREC DL  | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |          97 |       nan | nan     | nan     |  -0.016 |  -0.207 |   0.177 |
| TREC DL  | nDCG@10  | 0.01 | all      | splade–dense | PW4      |          97 |        67 |   0.91  |   0.27  |   0.877 |   0.774 |   0.96  |
| TREC DL  | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |          97 |        63 |   0.683 |   0.27  |   0.565 |   0.399 |   0.713 |
| TREC DL  | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |          97 |       nan | nan     | nan     |   0.312 |   0.128 |   0.507 |
| TREC DL  | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |           4 |         2 |   1     |   0.5   |   1     |   1     |   1     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |           4 |         1 |   1     |   1     | nan     | nan     | nan     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |           4 |       nan | nan     | nan     | nan     | nan     | nan     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |           3 |         1 |   1     |   1     | nan     | nan     | nan     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |           3 |         0 | nan     | nan     | nan     | nan     | nan     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |           3 |       nan | nan     | nan     | nan     | nan     | nan     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |           7 |         3 |   1     |   0.556 |   1     |   1     |   1     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |           7 |         3 |   0.667 |   0.333 |   0.5   |   0     |   1     |
| TREC DL  | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |           7 |       nan | nan     | nan     |   0.5   |   0     |   1     |
| BEIR     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |        1192 |       447 |   0.821 |   0.271 |   0.755 |   0.704 |   0.801 |
| BEIR     | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |        1192 |       411 |   0.679 |   0.262 |   0.565 |   0.5   |   0.624 |
| BEIR     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |        1192 |       nan | nan     | nan     |   0.19  |   0.118 |   0.264 |
| BEIR     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |        1192 |       432 |   0.771 |   0.27  |   0.686 |   0.631 |   0.74  |
| BEIR     | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |        1192 |       402 |   0.597 |   0.267 |   0.45  |   0.383 |   0.513 |
| BEIR     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |        1192 |       nan | nan     | nan     |   0.236 |   0.159 |   0.317 |
| BEIR     | nDCG@10  | 0.01 | all      | splade–dense | PW4      |        1192 |       495 |   0.855 |   0.27  |   0.801 |   0.758 |   0.842 |
| BEIR     | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |        1192 |       475 |   0.684 |   0.269 |   0.568 |   0.508 |   0.627 |
| BEIR     | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |        1192 |       nan | nan     | nan     |   0.233 |   0.164 |   0.303 |
| BEIR     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |         628 |       172 |   0.936 |   0.279 |   0.911 |   0.856 |   0.957 |
| BEIR     | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |         628 |       139 |   0.784 |   0.278 |   0.701 |   0.6   |   0.793 |
| BEIR     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |         628 |       nan | nan     | nan     |   0.21  |   0.109 |   0.32  |
| BEIR     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |         524 |       149 |   0.899 |   0.277 |   0.861 |   0.788 |   0.924 |
| BEIR     | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |         524 |       119 |   0.731 |   0.292 |   0.62  |   0.5   |   0.728 |
| BEIR     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |         524 |       nan | nan     | nan     |   0.24  |   0.114 |   0.376 |
| BEIR     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |         622 |       212 |   0.939 |   0.278 |   0.915 |   0.868 |   0.957 |
| BEIR     | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |         622 |       183 |   0.787 |   0.283 |   0.703 |   0.614 |   0.782 |
| BEIR     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |         622 |       nan | nan     | nan     |   0.212 |   0.121 |   0.306 |
| dl19     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |          43 |        27 |   0.593 |   0.274 |   0.439 |   0.187 |   0.667 |
| dl19     | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |          43 |        24 |   0.583 |   0.255 |   0.441 |   0.154 |   0.695 |
| dl19     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |          43 |       nan | nan     | nan     |  -0.002 |  -0.229 |   0.253 |
| dl19     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |          43 |        28 |   0.643 |   0.27  |   0.51  |   0.26  |   0.731 |
| dl19     | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |          43 |        25 |   0.72  |   0.262 |   0.62  |   0.364 |   0.841 |
| dl19     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |          43 |       nan | nan     | nan     |  -0.11  |  -0.379 |   0.172 |
| dl19     | nDCG@10  | 0.01 | all      | splade–dense | PW4      |          43 |        29 |   0.931 |   0.262 |   0.907 |   0.76  |   1     |
| dl19     | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |          43 |        24 |   0.792 |   0.252 |   0.722 |   0.478 |   0.928 |
| dl19     | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |          43 |       nan | nan     | nan     |   0.185 |  -0.072 |   0.471 |
| dl19     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |           1 |         0 | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |           1 |         0 | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |           1 |       nan | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |           1 |         0 | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |           1 |         0 | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |           1 |       nan | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |           2 |         0 | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |           2 |         0 | nan     | nan     | nan     | nan     | nan     |
| dl19     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |           2 |       nan | nan     | nan     | nan     | nan     | nan     |
| dl20     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |          54 |        31 |   0.71  |   0.241 |   0.617 |   0.407 |   0.812 |
| dl20     | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |          54 |        31 |   0.645 |   0.356 |   0.449 |   0.201 |   0.679 |
| dl20     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |          54 |       nan | nan     | nan     |   0.168 |  -0.109 |   0.447 |
| dl20     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |          54 |        32 |   0.625 |   0.229 |   0.513 |   0.311 |   0.714 |
| dl20     | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |          54 |        37 |   0.622 |   0.291 |   0.467 |   0.243 |   0.668 |
| dl20     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |          54 |       nan | nan     | nan     |   0.047 |  -0.226 |   0.319 |
| dl20     | nDCG@10  | 0.01 | all      | splade–dense | PW4      |          54 |        38 |   0.895 |   0.283 |   0.853 |   0.698 |   0.965 |
| dl20     | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |          54 |        39 |   0.615 |   0.306 |   0.446 |   0.229 |   0.65  |
| dl20     | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |          54 |       nan | nan     | nan     |   0.407 |   0.151 |   0.665 |
| dl20     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |           3 |         2 |   1     |   0.5   |   1     |   1     |   1     |
| dl20     | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |           3 |         1 |   1     |   1     | nan     | nan     | nan     |
| dl20     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |           3 |       nan | nan     | nan     | nan     | nan     | nan     |
| dl20     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |           2 |         1 |   1     |   1     | nan     | nan     | nan     |
| dl20     | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |           2 |         0 | nan     | nan     | nan     | nan     | nan     |
| dl20     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |           2 |       nan | nan     | nan     | nan     | nan     | nan     |
| dl20     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |           5 |         3 |   1     |   0.556 |   1     |   1     |   1     |
| dl20     | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |           5 |         3 |   0.667 |   0.333 |   0.5   |   0     |   1     |
| dl20     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |           5 |       nan | nan     | nan     |   0.5   |   0     |   1     |
| nfcorpus | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |         228 |        97 |   0.814 |   0.275 |   0.744 |   0.633 |   0.845 |
| nfcorpus | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |         228 |        87 |   0.644 |   0.28  |   0.505 |   0.357 |   0.64  |
| nfcorpus | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |         228 |       nan | nan     | nan     |   0.239 |   0.084 |   0.4   |
| nfcorpus | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |         228 |        94 |   0.819 |   0.291 |   0.745 |   0.63  |   0.85  |
| nfcorpus | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |         228 |        88 |   0.511 |   0.276 |   0.325 |   0.178 |   0.465 |
| nfcorpus | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |         228 |       nan | nan     | nan     |   0.42  |   0.256 |   0.586 |
| nfcorpus | nDCG@10  | 0.01 | all      | splade–dense | PW4      |         228 |       100 |   0.83  |   0.28  |   0.764 |   0.658 |   0.861 |
| nfcorpus | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |         228 |        94 |   0.596 |   0.281 |   0.438 |   0.298 |   0.569 |
| nfcorpus | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |         228 |       nan | nan     | nan     |   0.326 |   0.17  |   0.485 |
| nfcorpus | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |          86 |        20 |   0.9   |   0.305 |   0.856 |   0.639 |   1     |
| nfcorpus | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |          86 |        16 |   0.812 |   0.25  |   0.75  |   0.442 |   1     |
| nfcorpus | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |          86 |       nan | nan     | nan     |   0.106 |  -0.144 |   0.411 |
| nfcorpus | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |          60 |        14 |   0.857 |   0.321 |   0.789 |   0.476 |   1     |
| nfcorpus | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |          60 |         9 |   0.778 |   0.235 |   0.71  |   0.278 |   1     |
| nfcorpus | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |          60 |       nan | nan     | nan     |   0.08  |  -0.355 |   0.55  |
| nfcorpus | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |          69 |        16 |   1     |   0.305 |   1     |   1     |   1     |
| nfcorpus | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |          69 |        16 |   0.688 |   0.258 |   0.579 |   0.229 |   0.883 |
| nfcorpus | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |          69 |       nan | nan     | nan     |   0.421 |   0.117 |   0.771 |
| fiqa     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |         476 |       140 |   0.879 |   0.309 |   0.824 |   0.743 |   0.896 |
| fiqa     | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |         476 |       114 |   0.719 |   0.294 |   0.603 |   0.479 |   0.716 |
| fiqa     | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |         476 |       nan | nan     | nan     |   0.222 |   0.097 |   0.355 |
| fiqa     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |         476 |       138 |   0.87  |   0.313 |   0.81  |   0.723 |   0.888 |
| fiqa     | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |         476 |       121 |   0.653 |   0.325 |   0.486 |   0.358 |   0.606 |
| fiqa     | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |         476 |       nan | nan     | nan     |   0.324 |   0.19  |   0.466 |
| fiqa     | nDCG@10  | 0.01 | all      | splade–dense | PW4      |         476 |       174 |   0.931 |   0.309 |   0.9   |   0.842 |   0.95  |
| fiqa     | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |         476 |       157 |   0.752 |   0.294 |   0.648 |   0.549 |   0.74  |
| fiqa     | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |         476 |       nan | nan     | nan     |   0.252 |   0.145 |   0.363 |
| fiqa     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |         317 |        95 |   0.926 |   0.297 |   0.895 |   0.815 |   0.967 |
| fiqa     | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |         317 |        70 |   0.743 |   0.283 |   0.641 |   0.487 |   0.78  |
| fiqa     | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |         317 |       nan | nan     | nan     |   0.254 |   0.099 |   0.422 |
| fiqa     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |         262 |        88 |   0.92  |   0.295 |   0.887 |   0.799 |   0.962 |
| fiqa     | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |         262 |        67 |   0.701 |   0.313 |   0.565 |   0.397 |   0.717 |
| fiqa     | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |         262 |       nan | nan     | nan     |   0.322 |   0.15  |   0.503 |
| fiqa     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |         313 |       127 |   0.945 |   0.304 |   0.921 |   0.858 |   0.975 |
| fiqa     | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |         313 |       103 |   0.806 |   0.303 |   0.721 |   0.607 |   0.825 |
| fiqa     | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |         313 |       nan | nan     | nan     |   0.199 |   0.082 |   0.325 |
| scifact  | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |         184 |        34 |   1     |   0.261 |   1     |   1     |   1     |
| scifact  | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |         184 |        32 |   0.875 |   0.326 |   0.814 |   0.618 |   0.957 |
| scifact  | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |         184 |       nan | nan     | nan     |   0.186 |   0.043 |   0.382 |
| scifact  | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |         184 |        34 |   0.941 |   0.262 |   0.92  |   0.793 |   1     |
| scifact  | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |         184 |        30 |   0.833 |   0.331 |   0.751 |   0.52  |   0.936 |
| scifact  | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |         184 |       nan | nan     | nan     |   0.169 |  -0.065 |   0.428 |
| scifact  | nDCG@10  | 0.01 | all      | splade–dense | PW4      |         184 |        37 |   0.973 |   0.257 |   0.964 |   0.874 |   1     |
| scifact  | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |         184 |        32 |   0.875 |   0.31  |   0.819 |   0.623 |   0.96  |
| scifact  | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |         184 |       nan | nan     | nan     |   0.145 |  -0.033 |   0.355 |
| scifact  | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |         169 |        34 |   1     |   0.261 |   1     |   1     |   1     |
| scifact  | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |         169 |        30 |   0.9   |   0.326 |   0.852 |   0.665 |   1     |
| scifact  | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |         169 |       nan | nan     | nan     |   0.148 |   0     |   0.335 |
| scifact  | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |         166 |        33 |   0.97  |   0.264 |   0.959 |   0.861 |   1     |
| scifact  | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |         166 |        28 |   0.821 |   0.332 |   0.733 |   0.484 |   0.931 |
| scifact  | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |         166 |       nan | nan     | nan     |   0.226 |   0.006 |   0.483 |
| scifact  | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |         173 |        36 |   1     |   0.258 |   1     |   1     |   1     |
| scifact  | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |         173 |        31 |   0.871 |   0.303 |   0.815 |   0.614 |   0.959 |
| scifact  | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |         173 |       nan | nan     | nan     |   0.185 |   0.041 |   0.386 |
| dbpedia  | nDCG@10  | 0.01 | all      | bm25–splade  | PW4      |         304 |       176 |   0.744 |   0.261 |   0.654 |   0.564 |   0.739 |
| dbpedia  | nDCG@10  | 0.01 | all      | bm25–splade  | MIX4     |         304 |       178 |   0.635 |   0.252 |   0.512 |   0.414 |   0.605 |
| dbpedia  | nDCG@10  | 0.01 | all      | bm25–splade  | PW4−MIX4 |         304 |       nan | nan     | nan     |   0.142 |   0.019 |   0.264 |
| dbpedia  | nDCG@10  | 0.01 | all      | bm25–dense   | PW4      |         304 |       166 |   0.627 |   0.257 |   0.497 |   0.398 |   0.593 |
| dbpedia  | nDCG@10  | 0.01 | all      | bm25–dense   | MIX4     |         304 |       163 |   0.558 |   0.253 |   0.408 |   0.301 |   0.508 |
| dbpedia  | nDCG@10  | 0.01 | all      | bm25–dense   | PW4−MIX4 |         304 |       nan | nan     | nan     |   0.089 |  -0.039 |   0.221 |
| dbpedia  | nDCG@10  | 0.01 | all      | splade–dense | PW4      |         304 |       184 |   0.772 |   0.257 |   0.693 |   0.609 |   0.77  |
| dbpedia  | nDCG@10  | 0.01 | all      | splade–dense | MIX4     |         304 |       192 |   0.641 |   0.257 |   0.516 |   0.422 |   0.606 |
| dbpedia  | nDCG@10  | 0.01 | all      | splade–dense | PW4−MIX4 |         304 |       nan | nan     | nan     |   0.177 |   0.057 |   0.293 |
| dbpedia  | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4      |          56 |        23 |   0.913 |   0.272 |   0.881 |   0.698 |   1     |
| dbpedia  | nDCG@10  | 0.01 | same_rel | bm25–splade  | MIX4     |          56 |        23 |   0.739 |   0.289 |   0.633 |   0.346 |   0.872 |
| dbpedia  | nDCG@10  | 0.01 | same_rel | bm25–splade  | PW4−MIX4 |          56 |       nan | nan     | nan     |   0.248 |  -0.059 |   0.573 |
| dbpedia  | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4      |          36 |        14 |   0.643 |   0.281 |   0.504 |   0.113 |   0.837 |
| dbpedia  | nDCG@10  | 0.01 | same_rel | bm25–dense   | MIX4     |          36 |        15 |   0.667 |   0.236 |   0.564 |   0.238 |   0.861 |
| dbpedia  | nDCG@10  | 0.01 | same_rel | bm25–dense   | PW4−MIX4 |          36 |       nan | nan     | nan     |  -0.06  |  -0.559 |   0.419 |
| dbpedia  | nDCG@10  | 0.01 | same_rel | splade–dense | PW4      |          67 |        33 |   0.818 |   0.262 |   0.754 |   0.56  |   0.917 |
| dbpedia  | nDCG@10  | 0.01 | same_rel | splade–dense | MIX4     |          67 |        33 |   0.697 |   0.257 |   0.592 |   0.362 |   0.79  |
| dbpedia  | nDCG@10  | 0.01 | same_rel | splade–dense | PW4−MIX4 |          67 |       nan | nan     | nan     |   0.162 |  -0.096 |   0.429 |

### Sign agreement (pooled pair types)

| scope    | subset   | metric   |   th | group   | pair           |   units |   agree |   mean_diff |   abs_diff |      lo |      hi |   diff_lo |   diff_hi |
|:---------|:---------|:---------|-----:|:--------|:---------------|--------:|--------:|------------:|-----------:|--------:|--------:|----------:|----------:|
| TREC DL  | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |    1243 |   0.853 |         nan |      0.022 |   0.816 |   0.887 |       nan |       nan |
| TREC DL  | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    1748 |   0.791 |         nan |      0.036 |   0.752 |   0.828 |       nan |       nan |
| TREC DL  | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     179 |   0.709 |         nan |      0.012 |   0.63  |   0.789 |       nan |       nan |
| TREC DL  | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.062 |         nan |    nan     |   0.016 |   0.107 |       nan |       nan |
| TREC DL  | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |      58 |   0.983 |         nan |      0.022 |   0.929 |   1     |       nan |       nan |
| TREC DL  | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |      70 |   0.929 |         nan |      0.036 |   0.846 |   1     |       nan |       nan |
| TREC DL  | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |       5 |   1     |         nan |      0.012 |   1     |   1     |       nan |       nan |
| TREC DL  | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.054 |         nan |    nan     |  -0.04  |   0.143 |       nan |       nan |
| BEIR     | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |   10580 |   0.901 |         nan |      0.032 |   0.89  |   0.911 |       nan |       nan |
| BEIR     | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |   14437 |   0.825 |         nan |      0.059 |   0.812 |   0.837 |       nan |       nan |
| BEIR     | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |    1480 |   0.728 |         nan |      0.035 |   0.699 |   0.755 |       nan |       nan |
| BEIR     | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.076 |         nan |    nan     |   0.063 |   0.089 |       nan |       nan |
| BEIR     | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |    4489 |   0.966 |         nan |      0.032 |   0.957 |   0.974 |       nan |       nan |
| BEIR     | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    6035 |   0.891 |         nan |      0.059 |   0.876 |   0.906 |       nan |       nan |
| BEIR     | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     528 |   0.799 |         nan |      0.035 |   0.756 |   0.841 |       nan |       nan |
| BEIR     | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.075 |         nan |    nan     |   0.06  |   0.09  |       nan |       nan |
| dl19     | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |     536 |   0.813 |         nan |      0.017 |   0.748 |   0.875 |       nan |       nan |
| dl19     | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |     722 |   0.737 |         nan |      0.023 |   0.673 |   0.797 |       nan |       nan |
| dl19     | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |      70 |   0.786 |         nan |      0.005 |   0.662 |   0.897 |       nan |       nan |
| dl19     | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.077 |         nan |    nan     |  -0     |   0.156 |       nan |       nan |
| dl19     | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |       8 |   0.875 |         nan |      0.017 |   0.8   |   1     |       nan |       nan |
| dl19     | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |       6 |   1     |         nan |      0.023 |   1     |   1     |       nan |       nan |
| dl19     | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |       0 | nan     |         nan |      0.005 | nan     | nan     |       nan |       nan |
| dl19     | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |  -0.125 |         nan |    nan     |  -0.2   |   0     |       nan |       nan |
| dl20     | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |     707 |   0.883 |         nan |      0.027 |   0.843 |   0.919 |       nan |       nan |
| dl20     | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    1026 |   0.829 |         nan |      0.047 |   0.782 |   0.874 |       nan |       nan |
| dl20     | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     109 |   0.661 |         nan |      0.017 |   0.558 |   0.768 |       nan |       nan |
| dl20     | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.053 |         nan |    nan     |  -0.004 |   0.107 |       nan |       nan |
| dl20     | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |      50 |   1     |         nan |      0.027 |   1     |   1     |       nan |       nan |
| dl20     | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |      64 |   0.922 |         nan |      0.047 |   0.826 |   1     |       nan |       nan |
| dl20     | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |       5 |   1     |         nan |      0.017 |   1     |   1     |       nan |       nan |
| dl20     | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.078 |         nan |    nan     |   0     |   0.174 |       nan |       nan |
| nfcorpus | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |    2060 |   0.896 |         nan |      0.027 |   0.87  |   0.919 |       nan |       nan |
| nfcorpus | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    2662 |   0.81  |         nan |      0.049 |   0.781 |   0.839 |       nan |       nan |
| nfcorpus | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     305 |   0.702 |         nan |      0.01  |   0.639 |   0.763 |       nan |       nan |
| nfcorpus | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.085 |         nan |    nan     |   0.053 |   0.117 |       nan |       nan |
| nfcorpus | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |     396 |   0.98  |         nan |      0.027 |   0.959 |   0.997 |       nan |       nan |
| nfcorpus | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |     478 |   0.858 |         nan |      0.049 |   0.798 |   0.914 |       nan |       nan |
| nfcorpus | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |      51 |   0.686 |         nan |      0.01  |   0.536 |   0.839 |       nan |       nan |
| nfcorpus | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.122 |         nan |    nan     |   0.073 |   0.177 |       nan |       nan |
| fiqa     | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |    3766 |   0.942 |         nan |      0.054 |   0.928 |   0.954 |       nan |       nan |
| fiqa     | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    5185 |   0.874 |         nan |      0.078 |   0.856 |   0.891 |       nan |       nan |
| fiqa     | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     470 |   0.781 |         nan |      0.065 |   0.733 |   0.827 |       nan |       nan |
| fiqa     | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.068 |         nan |    nan     |   0.049 |   0.087 |       nan |       nan |
| fiqa     | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |    2580 |   0.968 |         nan |      0.054 |   0.957 |   0.978 |       nan |       nan |
| fiqa     | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    3436 |   0.9   |         nan |      0.078 |   0.882 |   0.917 |       nan |       nan |
| fiqa     | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     304 |   0.806 |         nan |      0.065 |   0.749 |   0.86  |       nan |       nan |
| fiqa     | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.068 |         nan |    nan     |   0.05  |   0.086 |       nan |       nan |
| scifact  | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |    1031 |   0.981 |         nan |      0.038 |   0.962 |   0.995 |       nan |       nan |
| scifact  | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    1462 |   0.922 |         nan |      0.075 |   0.889 |   0.951 |       nan |       nan |
| scifact  | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     106 |   0.896 |         nan |      0.015 |   0.812 |   0.968 |       nan |       nan |
| scifact  | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.059 |         nan |    nan     |   0.028 |   0.093 |       nan |       nan |
| scifact  | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |     999 |   0.983 |         nan |      0.038 |   0.966 |   0.996 |       nan |       nan |
| scifact  | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    1422 |   0.923 |         nan |      0.075 |   0.889 |   0.952 |       nan |       nan |
| scifact  | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     100 |   0.89  |         nan |      0.015 |   0.802 |   0.966 |       nan |       nan |
| scifact  | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.06  |         nan |    nan     |   0.029 |   0.095 |       nan |       nan |
| dbpedia  | all      | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |    3723 |   0.84  |         nan |      0.018 |   0.819 |   0.861 |       nan |       nan |
| dbpedia  | all      | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |    5128 |   0.754 |         nan |      0.028 |   0.731 |   0.777 |       nan |       nan |
| dbpedia  | all      | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |     599 |   0.669 |         nan |      0.018 |   0.624 |   0.715 |       nan |       nan |
| dbpedia  | all      | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.086 |         nan |    nan     |   0.059 |   0.111 |       nan |       nan |
| dbpedia  | same_rel | nDCG@10  | 0.01 | PP      | [PP pooled, 6] |     514 |   0.914 |         nan |      0.018 |   0.879 |   0.948 |       nan |       nan |
| dbpedia  | same_rel | nDCG@10  | 0.01 | PL      | [PL pooled, 8] |     699 |   0.807 |         nan |      0.028 |   0.753 |   0.859 |       nan |       nan |
| dbpedia  | same_rel | nDCG@10  | 0.01 | LL      | [LL pooled, 1] |      73 |   0.726 |         nan |      0.018 |   0.608 |   0.844 |       nan |       nan |
| dbpedia  | same_rel | nDCG@10  | 0.01 | PP−PL   | [PP − PL]      |     nan |   0.108 |         nan |    nan     |   0.056 |   0.161 |       nan |       nan |

### Replacement flips (listwise, retriever pairs pooled)

| scope    | reranker   | kind   | pair_type   | retriever_pair   |   n_pairs |   flips |   flip_rate |     lo |     hi |
|:---------|:-----------|:-------|:------------|:-----------------|----------:|--------:|------------:|-------:|-------:|
| TREC DL  | RankZephyr | L      | rel-nonrel  | pooled           |    101344 |   16046 |      0.1583 | 0.1409 | 0.1759 |
| TREC DL  | RankZephyr | L      | rel-rel     | pooled           |    111493 |   25556 |      0.2292 | 0.2066 | 0.2516 |
| TREC DL  | FIRST      | L      | rel-nonrel  | pooled           |    101344 |   14716 |      0.1452 | 0.1302 | 0.1598 |
| TREC DL  | FIRST      | L      | rel-rel     | pooled           |    111493 |   21770 |      0.1953 | 0.1791 | 0.2113 |
| BEIR     | RankZephyr | L      | rel-nonrel  | pooled           |    370806 |   64543 |      0.1741 | 0.164  | 0.1837 |
| BEIR     | RankZephyr | L      | rel-rel     | pooled           |     95153 |   23636 |      0.2484 | 0.2353 | 0.2613 |
| BEIR     | FIRST      | L      | rel-nonrel  | pooled           |    370806 |   50110 |      0.1351 | 0.1261 | 0.144  |
| BEIR     | FIRST      | L      | rel-rel     | pooled           |     95153 |   18735 |      0.1969 | 0.1858 | 0.2083 |
| dl19     | RankZephyr | L      | rel-nonrel  | pooled           |     44088 |    7048 |      0.1599 | 0.1321 | 0.1878 |
| dl19     | RankZephyr | L      | rel-rel     | pooled           |     57296 |   13332 |      0.2327 | 0.1921 | 0.2679 |
| dl19     | FIRST      | L      | rel-nonrel  | pooled           |     44088 |    6033 |      0.1368 | 0.115  | 0.1575 |
| dl19     | FIRST      | L      | rel-rel     | pooled           |     57296 |   11574 |      0.202  | 0.1748 | 0.226  |
| dl20     | RankZephyr | L      | rel-nonrel  | pooled           |     57256 |    8998 |      0.1572 | 0.1342 | 0.1794 |
| dl20     | RankZephyr | L      | rel-rel     | pooled           |     54197 |   12224 |      0.2255 | 0.2027 | 0.2469 |
| dl20     | FIRST      | L      | rel-nonrel  | pooled           |     57256 |    8683 |      0.1517 | 0.131  | 0.1708 |
| dl20     | FIRST      | L      | rel-rel     | pooled           |     54197 |   10196 |      0.1881 | 0.1707 | 0.2058 |
| nfcorpus | RankZephyr | L      | rel-nonrel  | pooled           |    112277 |   22482 |      0.2002 | 0.1813 | 0.2186 |
| nfcorpus | RankZephyr | L      | rel-rel     | pooled           |     39765 |    9487 |      0.2386 | 0.2133 | 0.2628 |
| nfcorpus | FIRST      | L      | rel-nonrel  | pooled           |    112277 |   17650 |      0.1572 | 0.1421 | 0.1718 |
| nfcorpus | FIRST      | L      | rel-rel     | pooled           |     39765 |    7095 |      0.1784 | 0.1633 | 0.1916 |
| fiqa     | RankZephyr | L      | rel-nonrel  | pooled           |     58306 |    6563 |      0.1126 | 0.0992 | 0.1265 |
| fiqa     | RankZephyr | L      | rel-rel     | pooled           |      1338 |     267 |      0.1996 | 0.1649 | 0.2312 |
| fiqa     | FIRST      | L      | rel-nonrel  | pooled           |     58306 |    2796 |      0.048  | 0.0412 | 0.0553 |
| fiqa     | FIRST      | L      | rel-rel     | pooled           |      1338 |     134 |      0.1001 | 0.0694 | 0.1365 |
| scifact  | RankZephyr | L      | rel-nonrel  | pooled           |     21698 |     802 |      0.037  | 0.0238 | 0.0519 |
| scifact  | RankZephyr | L      | rel-rel     | pooled           |       110 |      14 |      0.1273 | 0.029  | 0.25   |
| scifact  | FIRST      | L      | rel-nonrel  | pooled           |     21698 |     446 |      0.0206 | 0.0138 | 0.0283 |
| scifact  | FIRST      | L      | rel-rel     | pooled           |       110 |      16 |      0.1455 | 0.0396 | 0.2523 |
| dbpedia  | RankZephyr | L      | rel-nonrel  | pooled           |    178525 |   34696 |      0.1943 | 0.1799 | 0.2085 |
| dbpedia  | RankZephyr | L      | rel-rel     | pooled           |     53940 |   13868 |      0.2571 | 0.2421 | 0.2712 |
| dbpedia  | FIRST      | L      | rel-nonrel  | pooled           |    178525 |   29218 |      0.1637 | 0.1513 | 0.1769 |
| dbpedia  | FIRST      | L      | rel-rel     | pooled           |     53940 |   11490 |      0.213  | 0.1982 | 0.2282 |

### Table 3

| scope    | set   |   n_queries |   sig_units |   sig_share_nodiff |   tau_units |   retention_pooled6 |   retention_pooled6_lo |   retention_pooled6_hi |   sig_mean |   sig_min |   sig_max |   sig_lo |   sig_hi |   sig_bm25–splade |   sig_bm25–dense |   sig_splade–dense |   tau_mean |   tau_min |   tau_max |   tau_lo |   tau_hi |   tau_bm25–splade |   tau_bm25–dense |   tau_splade–dense |   retention_mean |   retention_min |   retention_max |   retention_lo |   retention_hi |   retention_bm25–splade |   retention_bm25–dense |   retention_splade–dense |
|:---------|:------|------------:|------------:|-------------------:|------------:|--------------------:|-----------------------:|-----------------------:|-----------:|----------:|----------:|---------:|---------:|------------------:|-----------------:|-------------------:|-----------:|----------:|----------:|---------:|---------:|------------------:|-----------------:|-------------------:|-----------------:|----------------:|----------------:|---------------:|---------------:|------------------------:|-----------------------:|-------------------------:|
| TREC DL  | PW4   |          97 |          36 |              0.722 |           6 |               0.629 |                  0.487 |                  0.741 |      0.722 |     0.583 |     1     |    0.556 |    0.889 |             0.583 |            0.583 |              1     |      0.444 |     0.167 |     0.833 |    0.111 |    0.889 |             0.333 |            0.167 |              0.833 |            0.626 |           0.486 |           0.854 |          0.485 |          0.738 |                   0.486 |                  0.538 |                    0.854 |
| TREC DL  | MIX4  |          97 |          36 |              0.583 |           6 |               0.484 |                  0.287 |                  0.639 |      0.722 |     0.583 |     0.917 |    0.5   |    0.833 |             0.583 |            0.667 |              0.917 |      1     |     1     |     1     |    0.333 |    1     |             1     |            1     |              1     |            0.485 |           0.367 |           0.613 |          0.292 |          0.639 |                   0.367 |                  0.476 |                    0.613 |
| BEIR     | PW4   |        1192 |          72 |              0.333 |          12 |               0.73  |                  0.691 |                  0.765 |      0.833 |     0.792 |     0.917 |    0.75  |    0.944 |             0.792 |            0.792 |              0.917 |      0.889 |     0.833 |     0.917 |    0.722 |    1     |             0.917 |            0.833 |              0.917 |            0.728 |           0.639 |           0.813 |          0.688 |          0.764 |                   0.731 |                  0.639 |                    0.813 |
| BEIR     | MIX4  |        1192 |          72 |              0.347 |          12 |               0.492 |                  0.438 |                  0.541 |      0.833 |     0.792 |     0.917 |    0.667 |    0.889 |             0.792 |            0.792 |              0.917 |      0.833 |     0.75  |     0.917 |    0.611 |    0.944 |             0.833 |            0.75  |              0.917 |            0.492 |           0.409 |           0.541 |          0.437 |          0.541 |                   0.525 |                  0.409 |                    0.541 |
| dl19     | PW4   |          43 |          18 |              0.778 |           3 |               0.602 |                  0.343 |                  0.797 |      0.778 |     0.667 |     1     |    0.444 |    1     |             0.667 |            0.667 |              1     |      0.333 |     0     |     1     |   -0.111 |    1     |             0     |            0     |              1     |            0.597 |           0.386 |           0.818 |          0.334 |          0.793 |                   0.386 |                  0.586 |                    0.818 |
| dl19     | MIX4  |          43 |          18 |              0.667 |           3 |               0.497 |                  0.214 |                  0.693 |      0.667 |     0.5   |     1     |    0.444 |    0.889 |             0.5   |            0.5   |              1     |      1     |     1     |     1     |   -0.111 |    1     |             1     |            1     |              1     |            0.501 |           0.335 |           0.633 |          0.216 |          0.692 |                   0.335 |                  0.535 |                    0.633 |
| dl20     | PW4   |          54 |          18 |              0.667 |           3 |               0.649 |                  0.489 |                  0.769 |      0.667 |     0.5   |     1     |    0.444 |    0.889 |             0.5   |            0.5   |              1     |      0.556 |     0.333 |     0.667 |   -0.111 |    1     |             0.667 |            0.333 |              0.667 |            0.646 |           0.5   |           0.881 |          0.487 |          0.766 |                   0.556 |                  0.5   |                    0.881 |
| dl20     | MIX4  |          54 |          18 |              0.5   |           3 |               0.473 |                  0.169 |                  0.695 |      0.778 |     0.667 |     0.833 |    0.444 |    0.889 |             0.667 |            0.833 |              0.833 |      1     |     1     |     1     |    0.333 |    1     |             1     |            1     |              1     |            0.473 |           0.395 |           0.598 |          0.175 |          0.694 |                   0.395 |                  0.427 |                    0.598 |
| nfcorpus | PW4   |         228 |          18 |              0.389 |           3 |               0.746 |                  0.663 |                  0.813 |      0.889 |     0.833 |     1     |    0.667 |    1     |             0.833 |            1     |              0.833 |      1     |     1     |     1     |    0.556 |    1     |             1     |            1     |              1     |            0.744 |           0.711 |           0.787 |          0.659 |          0.812 |                   0.735 |                  0.711 |                    0.787 |
| nfcorpus | MIX4  |         228 |          18 |              0.222 |           3 |               0.464 |                  0.346 |                  0.565 |      0.667 |     0.667 |     0.667 |    0.556 |    1     |             0.667 |            0.667 |              0.667 |      1     |     1     |     1     |    0.556 |    1     |             1     |            1     |              1     |            0.462 |           0.421 |           0.524 |          0.345 |          0.564 |                   0.524 |                  0.421 |                    0.441 |
| fiqa     | PW4   |         476 |          18 |              0.167 |           3 |               0.755 |                  0.692 |                  0.809 |      1     |     1     |     1     |    0.889 |    1     |             1     |            1     |              1     |      0.778 |     0.667 |     1     |    0.778 |    1     |             0.667 |            0.667 |              1     |            0.753 |           0.657 |           0.855 |          0.689 |          0.807 |                   0.746 |                  0.657 |                    0.855 |
| fiqa     | MIX4  |         476 |          18 |              0.5   |           3 |               0.496 |                  0.397 |                  0.581 |      1     |     1     |     1     |    0.667 |    1     |             1     |            1     |              1     |      0.778 |     0.667 |     1     |    0.333 |    1     |             0.667 |            0.667 |              1     |            0.492 |           0.38  |           0.591 |          0.391 |          0.579 |                   0.505 |                  0.38  |                    0.591 |
| scifact  | PW4   |         184 |          18 |              0.444 |           3 |               0.802 |                  0.692 |                  0.895 |      0.667 |     0.5   |     0.833 |    0.667 |    1     |             0.667 |            0.5   |              0.833 |      0.778 |     0.667 |     1     |    0.556 |    1     |             1     |            0.667 |              0.667 |            0.802 |           0.732 |           0.878 |          0.688 |          0.895 |                   0.796 |                  0.732 |                    0.878 |
| scifact  | MIX4  |         184 |          18 |              0.333 |           3 |               0.617 |                  0.476 |                  0.727 |      1     |     1     |     1     |    0.556 |    1     |             1     |            1     |              1     |      0.778 |     0.667 |     1     |    0.556 |    1     |             1     |            0.667 |              0.667 |            0.617 |           0.613 |           0.62  |          0.476 |          0.727 |                   0.62  |                  0.613 |                    0.619 |
| dbpedia  | PW4   |         304 |          18 |              0.333 |           3 |               0.635 |                  0.563 |                  0.698 |      0.778 |     0.667 |     1     |    0.444 |    1     |             0.667 |            0.667 |              1     |      1     |     1     |     1     |    0.556 |    1     |             1     |            1     |              1     |            0.632 |           0.511 |           0.718 |          0.56  |          0.696 |                   0.667 |                  0.511 |                    0.718 |
| dbpedia  | MIX4  |         304 |          18 |              0.333 |           3 |               0.429 |                  0.338 |                  0.511 |      0.667 |     0.5   |     1     |    0.333 |    0.889 |             0.5   |            0.5   |              1     |      0.778 |     0.667 |     1     |    0.333 |    1     |             0.667 |            0.667 |              1     |            0.43  |           0.322 |           0.496 |          0.339 |          0.511 |                   0.496 |                  0.322 |                    0.471 |

### Candidate overlap

| scope    | pair         |    n |   jaccard |   jac_lo |   jac_hi |   same_rel |   same_rel_lo |   same_rel_hi |
|:---------|:-------------|-----:|----------:|---------:|---------:|-----------:|--------------:|--------------:|
| TREC DL  | bm25–splade  |   97 |     0.23  |    0.2   |    0.26  |      0.041 |         0.01  |         0.082 |
| TREC DL  | bm25–dense   |   97 |     0.181 |    0.158 |    0.205 |      0.031 |         0     |         0.072 |
| TREC DL  | splade–dense |   97 |     0.389 |    0.358 |    0.421 |      0.072 |         0.021 |         0.124 |
| BEIR     | bm25–splade  | 1192 |     0.241 |    0.234 |    0.249 |      0.527 |         0.499 |         0.555 |
| BEIR     | bm25–dense   | 1192 |     0.148 |    0.142 |    0.153 |      0.44  |         0.412 |         0.468 |
| BEIR     | splade–dense | 1192 |     0.233 |    0.227 |    0.239 |      0.522 |         0.494 |         0.55  |
| dl19     | bm25–splade  |   43 |     0.238 |    0.192 |    0.288 |      0.023 |         0     |         0.07  |
| dl19     | bm25–dense   |   43 |     0.19  |    0.155 |    0.228 |      0.023 |         0     |         0.07  |
| dl19     | splade–dense |   43 |     0.397 |    0.352 |    0.44  |      0.047 |         0     |         0.116 |
| dl20     | bm25–splade  |   54 |     0.224 |    0.187 |    0.26  |      0.056 |         0     |         0.13  |
| dl20     | bm25–dense   |   54 |     0.174 |    0.146 |    0.203 |      0.037 |         0     |         0.093 |
| dl20     | splade–dense |   54 |     0.383 |    0.339 |    0.427 |      0.093 |         0.019 |         0.185 |
| nfcorpus | bm25–splade  |  228 |     0.267 |    0.244 |    0.291 |      0.377 |         0.316 |         0.439 |
| nfcorpus | bm25–dense   |  228 |     0.163 |    0.147 |    0.178 |      0.263 |         0.206 |         0.32  |
| nfcorpus | splade–dense |  228 |     0.232 |    0.215 |    0.251 |      0.303 |         0.241 |         0.36  |
| fiqa     | bm25–splade  |  476 |     0.232 |    0.222 |    0.241 |      0.666 |         0.624 |         0.708 |
| fiqa     | bm25–dense   |  476 |     0.133 |    0.126 |    0.14  |      0.55  |         0.504 |         0.595 |
| fiqa     | splade–dense |  476 |     0.232 |    0.223 |    0.24  |      0.658 |         0.616 |         0.7   |
| scifact  | bm25–splade  |  184 |     0.276 |    0.26  |    0.291 |      0.918 |         0.875 |         0.957 |
| scifact  | bm25–dense   |  184 |     0.193 |    0.18  |    0.206 |      0.902 |         0.859 |         0.94  |
| scifact  | splade–dense |  184 |     0.241 |    0.228 |    0.253 |      0.94  |         0.902 |         0.973 |
| dbpedia  | bm25–splade  |  304 |     0.216 |    0.2   |    0.233 |      0.184 |         0.141 |         0.23  |
| dbpedia  | bm25–dense   |  304 |     0.131 |    0.12  |    0.142 |      0.118 |         0.082 |         0.158 |
| dbpedia  | splade–dense |  304 |     0.231 |    0.218 |    0.244 |      0.22  |         0.174 |         0.27  |

## Appendix D. Same-input reruns

| reranker   | pair_type   |   n_queries |   n_lists |   identical_lists |   n_pairs |   flips |   flip_rate |     lo |     hi |
|:-----------|:------------|------------:|----------:|------------------:|----------:|--------:|------------:|-------:|-------:|
| RankZephyr | rel-nonrel  |          50 |       150 |                36 |    110252 |    1327 |      0.012  | 0.0066 | 0.0184 |
| RankZephyr | rel-rel     |          50 |       150 |                36 |     52502 |     603 |      0.0115 | 0.0043 | 0.0246 |
| FIRST      | rel-nonrel  |          10 |        30 |                30 |     19198 |       0 |      0      | 0      | 0      |
| FIRST      | rel-rel     |          10 |        30 |                30 |     30694 |       0 |      0      | 0      | 0      |

(Per-dataset κ threshold sensitivity: tables/appendixB_kappa_by_dataset_threshold.csv, 324 rows)

Total time 66 s