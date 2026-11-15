"""Keep only F queries that have all 6 lists (2 models x 3 retrievers); write queries_F.csv (final) and
queries_F_missing.csv. The planned set is read from queries_F.planned.csv."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import LISTWISE_F
P = LISTWISE_F
plan = pd.read_csv(f'{P}/queries_F.planned.csv', dtype={'qid': str})
have = {}
for m in ('zephyr', 'first'):
    have[m] = {tuple(json.loads(l)['job_id'].split('|')[1:]) for l in open(f'{P}/F/out/{m}.jsonl')} \
        if os.path.exists(f'{P}/F/out/{m}.jsonl') else set()
R = ('bm25', 'splade', 'dense')
ok = []
for d, q in plan[['dataset', 'qid']].values:
    ok.append(all((d, r, q) in have[m] for m in have for r in R))
plan['complete'] = ok
plan[plan.complete][['dataset', 'qid']].to_csv(f'{P}/queries_F.csv', index=False)
plan[~plan.complete][['dataset', 'qid']].to_csv(f'{P}/queries_F_missing.csv', index=False)
print('planned', len(plan), 'complete', int(plan.complete.sum()), 'missing', int((~plan.complete).sum()))
print(plan[plan.complete].groupby('dataset').size().to_dict())
