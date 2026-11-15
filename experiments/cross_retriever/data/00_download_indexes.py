"""Download all Pyserini prebuilt indexes used by Experiment 1 (resumable: skips finished ones)."""
from pyserini.util import download_prebuilt_index

NAMES = []
for ds in ['scifact', 'nfcorpus', 'fiqa', 'dbpedia-entity']:
    NAMES += [f'beir-v1.0.0-{ds}.flat', f'beir-v1.0.0-{ds}.splade-pp-ed', f'beir-v1.0.0-{ds}.bge-base-en-v1.5']
NAMES += ['msmarco-v1-passage', 'msmarco-v1-passage.splade-pp-ed', 'msmarco-v1-passage.bge-base-en-v1.5']

for n in NAMES:
    print('==', n, flush=True)
    print(download_prebuilt_index(n, verbose=True), flush=True)
print('ALL_DONE', flush=True)
