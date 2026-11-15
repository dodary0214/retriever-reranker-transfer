# Data sources

No benchmark documents, queries or relevance judgments are included in this repository. They are downloaded from their
official distributions by the preparation scripts (`experiments/*/data/`), which read them through `ir_datasets` and the
Pyserini prebuilt indexes. Check the license of each collection before using or redistributing derived files.

## Collections

| Code | Collection | `ir_datasets` id | Pyserini index (BM25 / SPLADE++ / BGE-base) | Queries used |
|---|---|---|---|---|
| `dl19` | TREC Deep Learning 2019, passage task (MS MARCO v1 passages) | `msmarco-passage/trec-dl-2019/judged` | `msmarco-v1-passage`, `.splade-pp-ed`, `.bge-base-en-v1.5` | 43 |
| `dl20` | TREC Deep Learning 2020, passage task | `msmarco-passage/trec-dl-2020/judged` | same as `dl19` | 54 |
| `nfcorpus` | BEIR NFCorpus, test | `beir/nfcorpus/test` | `beir-v1.0.0-nfcorpus.{flat,splade-pp-ed,bge-base-en-v1.5}` | 228 of 308 |
| `fiqa` | BEIR FiQA-2018, test | `beir/fiqa/test` | `beir-v1.0.0-fiqa.*` | 476 of 648 |
| `scifact` | BEIR SciFact, test | `beir/scifact/test` | `beir-v1.0.0-scifact.*` | 184 of 300 |
| `dbpedia` | BEIR DBpedia-Entity, test | `beir/dbpedia-entity/test` | `beir-v1.0.0-dbpedia-entity.*` | 304 of 400 |

- Queries used = evaluation queries (sets E and F, 1,289 in total). Of the 1,753 queries with candidates from all
  three retrievers, the 547 queries of set E were chosen first (all DL19/DL20 queries and seed-0 samples of 200 FiQA,
  150 DBpedia and 100 NFCorpus queries; `build_jobs_e.py`), and set F (742 queries) was drawn from the remaining 1,206
  queries by dataset share with a seed-0 permutation (`build_jobs_f.py Fn 742`). The size of F was set by the GPU time
  available for the listwise runs. The query lists are in `results/manifests/`.
- Experiment 2 uses the 97 DL19 and DL20 queries and their judged documents.
- `results/manifests/` lists query and document identifiers only. The relevance grades of the Experiment 2 pairs are
  not included; `build_pairs.py` reads them from the qrels.
- MS MARCO and the TREC DL judgments are distributed under the MS MARCO terms (non-commercial research use). The BEIR
  datasets carry the licenses of their original sources; see the BEIR repository.

## Models

| Role | Name used in the code | Checkpoint (Hugging Face) |
|---|---|---|
| Dense retriever, Experiment 2 embeddings | BGE-base | `BAAI/bge-base-en-v1.5` |
| Learned sparse retriever | SPLADE++ | `naver/splade-cocondenser-ensembledistil` (query encoder; Pyserini index `splade-pp-ed`) |
| Pointwise reranker | `minilm` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Pointwise reranker | `bge-base` | `BAAI/bge-reranker-base` |
| Pointwise reranker | `bge-m3` | `BAAI/bge-reranker-v2-m3` |
| Pointwise reranker | `monot5` | `castorini/monot5-base-msmarco-10k` |
| Listwise reranker | `zephyr` | `castorini/rank_zephyr_7b_v1_full` |
| Listwise reranker | `first` | `castorini/first_mistral` (FIRST objective on Mistral-7B-Instruct-v0.3, trained on the RankZephyr GPT-4 ranking data; Chen et al., SIGIR 2025) |
| Pointwise reranker (robustness only) | `qwen` | `Qwen/Qwen3-Reranker-4B` |
| Experiment 2 control | — | `BAAI/bge-reranker-v2-m3` |

Listwise prompts, passage truncation and output parsing come from `rank_llm` 0.26.0 (`RankListwiseOSLLM`); the
Qwen3-Reranker-4B prompt follows its model card.
