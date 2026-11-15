"""docs/<ds>.jsonl = every docid in the three top-100 runs (+ all judged docids for dl19/dl20).
Fields: docid, title, text (= title + ' ' + body, as Pyserini indexes BEIR), n_tokens (bert-base-uncased, no special tokens)."""
import os
import sys
import argparse
import json

from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import DATASETS, EXP, RETRIEVERS, index_name, read_run, start_watchdog, update_manifest

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
args = ap.parse_args()
start_watchdog()
tok = AutoTokenizer.from_pretrained('bert-base-uncased')

stats = {}
for ds in args.datasets:
    ids = set()
    for r in RETRIEVERS:
        ids |= {d for hs in read_run(f'{EXP}/common/runs/{ds}.{r}.top100.txt').values() for d, _ in hs}
    n_cand = len(ids)
    if ds in ('dl19', 'dl20'):
        ids |= {l.split('\t')[1] for l in open(f'{EXP}/common/qrels/{ds}.tsv')}
    searcher = LuceneSearcher.from_prebuilt_index(index_name(ds, 'bm25'))
    rows, missing = [], []
    for docid in sorted(ids):
        doc = searcher.doc(docid)
        if doc is None:
            missing.append(docid)
            continue
        raw = json.loads(doc.raw())
        if 'contents' in raw:  # msmarco-v1-passage
            title, text = '', raw['contents']
        else:  # BEIR flat
            title, text = raw.get('title', ''), raw.get('text', '')
            text = f'{title} {text}'.strip()
        rows.append({'docid': docid, 'title': title, 'text': text})
    lens = [len(x) for x in tok([r['text'] for r in rows], add_special_tokens=False)['input_ids']]
    with open(f'{EXP}/common/docs/{ds}.jsonl', 'w') as f:
        for r, n in zip(rows, lens):
            r['n_tokens'] = n
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    stats[ds] = {'n_docs': len(rows), 'n_candidate_docs': n_cand, 'n_missing_in_index': len(missing),
                 'missing_examples': missing[:5]}
    print(ds, stats[ds], flush=True)
    searcher.close()
update_manifest({'docs': stats})
