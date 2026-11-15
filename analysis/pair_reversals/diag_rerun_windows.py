"""Diagnostic: where do same-input rerun outputs of RankZephyr-7B first differ from
the original outputs? Sliding window 20 / stride 10 from the bottom: windows start at 80, 70, ..., 0 (9 calls).
Window starting at s fixes final positions [s+10, s+20) (all of [0, 20) for s = 0), and passes up the set
final[0, s+10) minus initial[0, s). A window is 'observably different' if its fixed segment (order) or its passed-up set
differs between the original and the rerun. Per-window outputs were not stored, so this is a lower bound on the first
divergence: an earlier window may differ only in the order of the passed-up documents. CPU only, reads outputs only."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from common import EXP, LISTWISE_E, LISTWISE_F, OUT, read_run
MIX, B = LISTWISE_E, LISTWISE_F
orig = {json.loads(l)['job_id']: json.loads(l)['ranking'] for l in open(f'{MIX}/out/zephyr.jsonl')}
rer = {}
for sub in ('rep', 'rep50'):
    for l in open(f'{B}/{sub}/out/zephyr.jsonl'):
        x = json.loads(l); rer[x['job_id']] = x['ranking']
runs = {}
rows = []
for job, b in rer.items():
    _, d, r, q = job.split('|')
    if (d, r) not in runs:
        runs[(d, r)] = read_run(f'{EXP}/common/runs/{d}.{r}.top100.txt')
    init = [doc for doc, _ in runs[(d, r)][q]][:100]
    a = orig[job]
    assert set(a) == set(b) == set(init)
    if len(init) < 100:  # fewer windows; excluded from the window analysis, counted separately
        rows.append(dict(job=job, dataset=d, retriever=r, identical=a == b, short=True)); continue
    diffs = []
    for w, s in enumerate(range(80, -1, -10), 1):
        lo, hi = (s + 10, s + 20) if s > 0 else (0, 20)
        seg = a[lo:hi] != b[lo:hi]
        carried = s > 0 and (set(a[:s + 10]) - set(init[:s])) != (set(b[:s + 10]) - set(init[:s]))
        diffs.append(bool(seg or carried))
    first = next((i + 1 for i, v in enumerate(diffs) if v), None)
    rows.append(dict(job=job, dataset=d, retriever=r, identical=a == b, first_window=first,
                     n_diff_windows=sum(diffs), diff_after_first=sum(diffs[first:]) if first else 0,
                     windows_after_first=9 - first if first else 0, top10_differs=a[:10] != b[:10],
                     top10_set_differs=set(a[:10]) != set(b[:10]), short=False))
t = pd.DataFrame(rows)
os.makedirs(f'{OUT}/derived', exist_ok=True)
t.to_csv(f'{OUT}/derived/diag_rerun_windows.csv', index=False)
print('lists', len(t), 'identical', int(t.identical.sum()), 'differing', int((~t.identical).sum()))
print('short lists (<100 candidates):', int(t.short.sum()), 'identical among them:', int(t[t.short].identical.sum()))
dd = t[~t.identical & ~t.short]
print('first observably different window (1 = bottom window, 9 = top window):')
print(dd.first_window.value_counts().sort_index().to_string())
print('of differing lists: later windows also different (share of windows after the first):',
      round(dd.diff_after_first.sum() / max(dd.windows_after_first.sum(), 1), 3),
      '; lists with all later windows different:', int((dd.diff_after_first == dd.windows_after_first).sum()),
      '; top-10 order differs:', int(dd.top10_differs.sum()), '; top-10 set differs:', int(dd.top10_set_differs.sum()))
print(dd.groupby('dataset').size().to_string())
