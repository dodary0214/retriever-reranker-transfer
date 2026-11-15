"""Download only the Faiss bge-base indexes (run in parallel with 00_download.py, which is stopped before its dense phase)."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import DATASETS, dense_index
from pyserini.util import download_prebuilt_index

for c in ["msmarco-v1-passage", "beir-v1.0.0-dbpedia-entity", "beir-v1.0.0-fiqa", "beir-v1.0.0-nfcorpus", "beir-v1.0.0-scifact"]:
    t = time.time()
    print(f"DONE {dense_index(c)} -> {download_prebuilt_index(dense_index(c), verbose=True)} ({time.time()-t:.0f}s)", flush=True)
print("ALL_DONE", flush=True)
