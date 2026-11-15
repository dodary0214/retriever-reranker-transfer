"""F/out/<model>.jsonl -> runs/<dataset>.<retriever>.<model>.txt (TREC, score 1000 - rank), same format as listwise_e/runs."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import LISTWISE_F

P = LISTWISE_F
os.makedirs(f'{P}/runs', exist_ok=True)
for m in ('zephyr', 'first'):
    files = defaultdict(list)
    for l in open(f'{P}/F/out/{m}.jsonl'):
        x = json.loads(l)
        _, d, r, q = x['job_id'].split('|')
        files[(d, r)] += [f'{q} Q0 {doc} {i + 1} {1000 - i} {m}\n' for i, doc in enumerate(x['ranking'])]
    for (d, r), lines in files.items():
        open(f'{P}/runs/{d}.{r}.{m}.txt', 'w').writelines(lines)
    print(m, {k: len(v) // 100 for k, v in files.items()})
