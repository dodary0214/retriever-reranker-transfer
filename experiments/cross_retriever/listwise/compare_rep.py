"""Replication check: rep/out/<model>.jsonl rankings compared with listwise_e/out/<model>.jsonl (exact equality).
Also prints t = (zephyr + first generation minutes) / 10 queries from the run logs."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
from common import LISTWISE_E, LISTWISE_F

P = LISTWISE_F
MIX = f'{LISTWISE_E}/out'
ok_all = True
gen = {}
for m in ('zephyr', 'first'):
    old = {json.loads(l)['job_id']: json.loads(l)['ranking'] for l in open(f'{MIX}/{m}.jsonl')}
    new = [json.loads(l) for l in open(f'{P}/rep/out/{m}.jsonl')]
    same = [x['ranking'] == old[x['job_id']] for x in new]
    diffpos = [sum(a != b for a, b in zip(x['ranking'], old[x['job_id']])) for x in new]
    print(f'{m}: {sum(same)}/{len(same)} lists identical; differing positions per list: {diffpos}; '
          f'n_bad {sum(x["n_bad"] for x in new)}')
    ok_all &= all(same)
    mins = [float(x) for x in re.findall(r'calls, ([\d.]+) min', open(f'{P}/rep/logs/run_{m}.log').read())]
    gen[m] = mins[-1] * 60
t = sum(gen.values()) / 10
print(f'generation sec: {gen}; t = {t:.1f} s per query (3 retrievers, both models)')
print('REPLICATION', 'PASS' if ok_all else 'FAIL')
open(f'{P}/rep/t_sec.txt', 'w').write(f'{t:.2f}\n')
sys.exit(0 if ok_all else 1)
