"""Download all prebuilt Pyserini indexes (resumable: skips finished ones)."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import DATASETS, bm25_index, splade_index, dense_index
from pyserini.util import download_prebuilt_index

corpora = sorted({c for _, c in DATASETS.values()}, key=lambda c: c != "msmarco-v1-passage")
names = [f(c) for f in (bm25_index, splade_index) for c in corpora] + [dense_index(c) for c in corpora]
for n in names:
    t = time.time()
    path = download_prebuilt_index(n, verbose=True)
    print(f"DONE {n} -> {path} ({time.time()-t:.0f}s)", flush=True)
print("ALL_DONE", flush=True)
